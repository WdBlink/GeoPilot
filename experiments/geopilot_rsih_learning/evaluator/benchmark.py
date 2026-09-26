#!/usr/bin/env python3
"""Strict UseGeo RGB-oriented triangle-mesh evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import laspy
import numpy as np
import scipy
from scipy.spatial import KDTree, cKDTree

ROOT = Path(__file__).resolve().parents[3]
FROZEN_CODE = ROOT / "out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code"
sys.path.insert(0, str(FROZEN_CODE))
from usegeo_benchmark import benchmark as pointcloud_v1


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol_v1.json"
SCORE_SCHEMA = "usegeo-mesh-score-1.0"
RUN_SCHEMA = "usegeo-mesh-run-manifest-1.0"
AGGREGATE_SCHEMA = "usegeo-mesh-aggregate-1.0"
SUBMISSION_KEYS = {
    "schema_version", "protocol_version", "track", "scene_id", "product",
    "geometry", "configuration_id", "method_metadata",
}
HASH_BINDINGS = {
    "protocol_sha256", "bundle_lock_sha256", "bundle_manifest_sha256",
    "input_manifest_sha256", "reference_manifest_sha256",
    "submission_contract_sha256", "mesh_sha256",
}
PLY_HEADER = (
    b"ply\nformat binary_little_endian 1.0\n"
    b"element vertex {vertices}\nproperty double x\nproperty double y\n"
    b"property double z\nelement face {triangles}\n"
    b"property list uchar int vertex_indices\nend_header\n"
)


class InvalidInput(Exception):
    """A deterministic invalid bundle or submission (CLI exit 2)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class EvaluatorFailure(Exception):
    """An evaluator or runtime failure (CLI exit 1)."""


_PROGRESS: Path | None = None


def progress(stage: str, **values: Any) -> None:
    """Append bounded-cost progress outside immutable score output; never metric input."""
    if _PROGRESS is not None:
        with _PROGRESS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                     "pid": os.getpid(), "stage": stage, **values}, allow_nan=False) + "\n")


def load_protocol() -> dict[str, Any]:
    """Load the frozen mesh protocol."""
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def runtime_versions(*, enforce: bool = True) -> dict[str, str]:
    """Return, and normally enforce, every frozen runtime version."""
    observed = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "laspy", "open3d")},
    }
    expected = {key: load_protocol()["runtime"][key] for key in observed}
    if enforce and observed != expected:
        raise EvaluatorFailure(f"frozen runtime mismatch: expected {expected}, observed {observed}")
    return observed


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _strict_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise InvalidInput(reason) from exc
    if not isinstance(value, dict):
        raise InvalidInput(reason)
    return value


def _regular(path: Path, reason: str) -> Path:
    try:
        if path.is_symlink() or not path.is_file():
            raise InvalidInput(reason)
    except OSError as exc:
        raise InvalidInput(reason) from exc
    return path


def _contained(root: Path, relative: str, reason: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise InvalidInput(reason)
    root = root.resolve()
    path = root.joinpath(*posix.parts)
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise InvalidInput(reason) from exc
    return _regular(path, reason)


def _snapshot(source: Path, destination: Path, limit: int, reason: str) -> str:
    """Copy a no-follow regular file, proving stable size/hash during the copy."""
    _regular(source, reason)
    before = source.stat()
    if before.st_size > limit:
        raise InvalidInput("GEOMETRY_TOO_LARGE" if source.name.endswith(".ply") else reason)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise InvalidInput(reason) from exc
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or destination.stat().st_size != before.st_size:
        raise InvalidInput(reason, "source changed during snapshot")
    return digest.hexdigest()


def verify_bundle(bundle: Path, bundle_lock: Path) -> dict[str, Any]:
    """Reuse the accepted point-cloud authority verifier unchanged."""
    try:
        return pointcloud_v1.verify_bundle(bundle, bundle_lock)
    except pointcloud_v1.InvalidInput as exc:
        raise InvalidInput(exc.reason, exc.detail) from exc
    except pointcloud_v1.EvaluatorFailure as exc:
        raise EvaluatorFailure(str(exc)) from exc


def _contract(path: Path, scene: str) -> dict[str, Any]:
    protocol = load_protocol()
    value = _strict_json(path, "MALFORMED_CONTRACT")
    if set(value) != SUBMISSION_KEYS:
        raise InvalidInput("UNKNOWN_FIELD")
    expected = {
        "schema_version": protocol["submission_schema_version"],
        "protocol_version": protocol["protocol_version"],
        "track": protocol["track"],
        "scene_id": scene,
        "product": protocol["product"],
        "geometry": protocol["geometry"],
    }
    codes = {
        "track": "WRONG_TRACK", "scene_id": "WRONG_SCENE",
        "product": "WRONG_PRODUCT", "geometry": "UNSAFE_PATH",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise InvalidInput(codes.get(key, "MALFORMED_CONTRACT"))
    config = value.get("configuration_id")
    if not isinstance(config, str) or not pointcloud_v1.CONFIGURATION_ID.fullmatch(config):
        raise InvalidInput("MALFORMED_CONTRACT")
    metadata = value.get("method_metadata")
    if not isinstance(metadata, dict) or not all(isinstance(k, str) for k in metadata):
        raise InvalidInput("MALFORMED_CONTRACT")
    return value


def snapshot_submission(submission: Path, destination: Path, scene: str) -> dict[str, Any]:
    """Create and bind a private immutable submission snapshot."""
    protocol = load_protocol()
    if submission.is_symlink() or not submission.is_dir():
        raise InvalidInput("GEOMETRY_NOT_REGULAR")
    contract_sha = _snapshot(
        submission / "submission.json", destination / "submission.json",
        int(protocol["limits"]["max_contract_bytes"]), "MALFORMED_CONTRACT",
    )
    contract = _contract(destination / "submission.json", scene)
    mesh_sha = _snapshot(
        _contained(submission, contract["geometry"], "UNSAFE_PATH"),
        destination / "mesh.ply", int(protocol["limits"]["max_mesh_bytes"]),
        "GEOMETRY_NOT_REGULAR",
    )
    return {"contract": contract, "contract_sha256": contract_sha, "mesh_sha256": mesh_sha}


def _read_header(handle: Any) -> tuple[int, int, int]:
    lines: list[bytes] = []
    total = 0
    for _ in range(10):
        line = handle.readline(256)
        total += len(line)
        if not line or len(line) >= 256 or not line.endswith(b"\n"):
            raise InvalidInput("PLY_SCHEMA_INVALID")
        lines.append(line)
        if line == b"end_header\n":
            break
    if len(lines) != 9 or lines[-1] != b"end_header\n":
        raise InvalidInput("PLY_FORMAT_UNSUPPORTED")
    try:
        vertex_count = int(lines[2].decode().removeprefix("element vertex ").strip())
        triangle_count = int(lines[6].decode().removeprefix("element face ").strip())
    except (UnicodeError, ValueError) as exc:
        raise InvalidInput("PLY_SCHEMA_INVALID") from exc
    if b"".join(lines) != PLY_HEADER.replace(b"{vertices}", str(vertex_count).encode()).replace(
        b"{triangles}", str(triangle_count).encode()
    ):
        raise InvalidInput("PLY_FORMAT_UNSUPPORTED")
    return vertex_count, triangle_count, total


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the protocol's sole accepted binary triangle PLY representation."""
    protocol = load_protocol()
    _regular(path, "GEOMETRY_NOT_REGULAR")
    if path.stat().st_size > int(protocol["limits"]["max_mesh_bytes"]):
        raise InvalidInput("GEOMETRY_TOO_LARGE")
    with path.open("rb") as handle:
        vertices_count, triangles_count, header_size = _read_header(handle)
        if vertices_count < 3 or triangles_count < 1:
            raise InvalidInput("EMPTY_GEOMETRY")
        if vertices_count > int(protocol["limits"]["max_vertices"]) or triangles_count > int(
            protocol["limits"]["max_triangles"]
        ):
            raise InvalidInput("COUNT_LIMIT_EXCEEDED")
        expected = header_size + vertices_count * 24 + triangles_count * 13
        if path.stat().st_size != expected:
            raise InvalidInput("PLY_SIZE_MISMATCH")
        vertices = np.fromfile(handle, dtype="<f8", count=vertices_count * 3).reshape(-1, 3)
        face_dtype = np.dtype([("count", "u1"), ("index", "<i4", (3,))], align=False)
        records = np.fromfile(handle, dtype=face_dtype, count=triangles_count)
    if not np.isfinite(vertices).all():
        raise InvalidInput("NONFINITE_GEOMETRY")
    if not np.all(records["count"] == 3):
        raise InvalidInput("PLY_SCHEMA_INVALID")
    triangles = np.asarray(records["index"], dtype=np.int64)
    if np.any(triangles < 0) or np.any(triangles >= vertices_count):
        raise InvalidInput("INDEX_OUT_OF_RANGE")
    if np.any((triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2]) |
              (triangles[:, 0] == triangles[:, 2])):
        raise InvalidInput("DEGENERATE_TRIANGLE")
    areas = triangle_areas(vertices, triangles)
    if np.any(~np.isfinite(areas)):
        raise InvalidInput("NONFINITE_GEOMETRY")
    if np.any(areas <= float(protocol["limits"]["degenerate_area_m2"])):
        raise InvalidInput("DEGENERATE_TRIANGLE")
    return vertices, triangles


def write_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Write the one frozen PLY representation."""
    vertices = np.asarray(vertices, dtype="<f8")
    triangles = np.asarray(triangles, dtype="<i4")
    header = PLY_HEADER.replace(b"{vertices}", str(len(vertices)).encode()).replace(
        b"{triangles}", str(len(triangles)).encode()
    )
    face_dtype = np.dtype([("count", "u1"), ("index", "<i4", (3,))], align=False)
    faces = np.empty(len(triangles), dtype=face_dtype)
    faces["count"] = 3
    faces["index"] = triangles
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(header)
        vertices.tofile(handle)
        faces.tofile(handle)


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    sides_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    sides_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    return np.linalg.norm(np.cross(sides_a, sides_b), axis=1) * 0.5


def structure_diagnostics(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Compute exact-index topology and boundary diagnostics."""
    areas = triangle_areas(vertices, triangles)
    edge_triangles = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, edge_triangles = edges[order], edge_triangles[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(edges)]
    incidence = ends - starts
    boundary = edges[starts[incidence == 1]]
    parent = np.arange(len(triangles), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for start, end in zip(starts, ends):
        base = int(edge_triangles[start])
        for other in edge_triangles[start + 1:end]:
            left, right = find(base), find(int(other))
            if left != right:
                parent[right] = left
    used = np.unique(triangles)
    return {
        "surface_area_m2": float(areas.sum(dtype=np.float64)),
        "degenerate_triangle_count": int(np.count_nonzero(areas <= 1e-12)),
        "unreferenced_vertex_count": int(len(vertices) - len(used)),
        "boundary_edge_count": int(np.count_nonzero(incidence == 1)),
        "boundary_length_m": float(np.linalg.norm(vertices[boundary[:, 0]] - vertices[boundary[:, 1]], axis=1).sum()) if len(boundary) else 0.0,
        "nonmanifold_edge_count": int(np.count_nonzero(incidence > 2)),
        "connected_component_count": len({find(i) for i in range(len(triangles))}),
    }


def sample_surface(
    vertices: np.ndarray, triangles: np.ndarray, count: int, seed: int,
    check: Callable[[], None] = lambda: None,
) -> np.ndarray:
    """Generate deterministic area-uniform samples with one PCG64 stream."""
    areas = triangle_areas(vertices, triangles)
    total = float(areas.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise InvalidInput("DEGENERATE_TRIANGLE")
    rng = np.random.Generator(np.random.PCG64(seed))
    output = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, 100000):
        check()
        size = min(100000, count - start)
        selected = rng.choice(len(triangles), size=size, p=areas / total)
        u, v = rng.random(size), rng.random(size)
        root = np.sqrt(u)
        bary = np.column_stack((1 - root, root * (1 - v), root * v))
        output[start:start + size] = np.einsum(
            "ni,nij->nj", bary, vertices[triangles[selected]], optimize=True
        )
    return output


def raycast_distances(
    vertices: np.ndarray, triangles: np.ndarray, queries: np.ndarray,
    origin: np.ndarray, chunk: int = 100000, check: Callable[[], None] = lambda: None,
) -> np.ndarray:
    """Return exact-candidate Float64 point-to-triangle distances.

    Triangle AABB spheres provide a conservative lower bound.  The nearest
    centre triangle gives an exact upper bound, so every triangle capable of
    improving it is included before Float64 refinement; the tree is only an
    acceleration and never decides the nearest surface.
    """
    return exact_triangle_queries(vertices, triangles, queries, origin, chunk, check)[0]


def exact_triangle_queries(
    vertices: np.ndarray, triangles: np.ndarray, queries: np.ndarray,
    origin: np.ndarray, chunk: int = 100000, check: Callable[[], None] = lambda: None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-candidate Float64 distances and primitive IDs."""
    del origin  # retained in the API/report as the fixed, non-fitting frame marker
    progress("reverse_tree_build_start", points=len(queries), triangles=len(triangles))
    faces = np.asarray(vertices[triangles], dtype=np.float64)
    lower = faces.min(axis=1)
    upper = faces.max(axis=1)
    centres = (lower + upper) * 0.5
    radii = np.linalg.norm(upper - lower, axis=1) * 0.5
    tree = cKDTree(centres)
    radius_bins = np.ceil(np.log2(radii)).astype(np.int64)
    groups = []
    for exponent in np.unique(radius_bins):
        ids = np.flatnonzero(radius_bins == exponent)
        groups.append((cKDTree(centres[ids]), ids, float(radii[ids].max())))
    max_candidate_pairs = int(load_protocol()["raycast"]["max_materialized_candidate_pairs"])
    output = np.empty(len(queries), dtype=np.float64)
    primitive_ids = np.empty(len(queries), dtype=np.int64)
    start = 0
    accepted_chunk = chunk
    progress("reverse_query_start", points=len(queries), triangles=len(triangles))
    while start < len(queries):
        check()
        size = min(accepted_chunk, len(queries) - start)
        while True:
            block = np.asarray(queries[start:start + size], dtype=np.float64)
            # The sphere test subtracts UTM-sized Float64 coordinates.  Each
            # coordinate can have accumulated rounding from min/max, centre,
            # subtraction and norm; gamma(64) * max(|coordinate|) is an
            # outward error bound for those operations, rather than a one-ulp
            # guess at the final comparison.
            scale = max(1.0, float(np.max(np.abs(block))), float(np.max(np.abs(centres))))
            envelope = 64.0 * np.finfo(np.float64).eps * scale
            _, seeds = tree.query(block, k=1, workers=1)
            seeds = np.atleast_1d(seeds).astype(np.int64, copy=False)
            best = _point_triangle_distances(block, faces[seeds])
            pair_count = sum(int(np.sum(group_tree.query_ball_point(
                block, best + group_radius + envelope, return_length=True
            ))) for group_tree, _, group_radius in groups)
            if pair_count <= max_candidate_pairs or size == 1:
                break
            progress("reverse_batch_shrink", start=start, size=size, candidate_pairs=pair_count)
            size = max(1, size // 2)
        if pair_count > max_candidate_pairs:
            raise EvaluatorFailure("single-query candidate pairs exceed materialization limit")
        accepted_chunk = size
        candidate_parts, query_parts = [], []
        for group_tree, ids, group_radius in groups:
            candidate_lists = group_tree.query_ball_point(block, best + group_radius + envelope)
            counts = np.fromiter(map(len, candidate_lists), dtype=np.int64, count=len(block))
            query_parts.append(np.repeat(np.arange(len(block), dtype=np.int64), counts))
            local = np.concatenate(candidate_lists).astype(np.int64, copy=False)
            candidate_parts.append(ids[local])
        query_ids = np.concatenate(query_parts)
        candidates = np.concatenate(candidate_parts)
        keep = (np.linalg.norm(centres[candidates] - block[query_ids], axis=1)
                - radii[candidates] <= best[query_ids] + envelope)
        candidates, query_ids = candidates[keep], query_ids[keep]
        exact = _point_triangle_distances(block[query_ids], faces[candidates])
        order = np.lexsort((candidates, exact, query_ids))
        ordered_queries = query_ids[order]
        first = np.r_[True, ordered_queries[1:] != ordered_queries[:-1]]
        winners = order[first]
        if len(winners) != len(block):
            raise EvaluatorFailure("exact triangle candidate set incomplete")
        values = exact[winners]
        primitive_ids[start:start + len(block)] = candidates[winners]
        if not np.isfinite(values).all():
            raise EvaluatorFailure("non-finite raycast distance")
        output[start:start + len(values)] = values
        start += size
        progress("reverse_query_progress", completed=start, total=len(queries), batch=size, candidate_pairs=pair_count)
    return output, primitive_ids


def _point_triangle_distances(point: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vectorized Float64 distance from one point to nondegenerate triangles."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normal, normal)
    ap = point - a
    plane_scale = np.einsum("ij,ij->i", ap, normal) / normal_sq
    projected = point - plane_scale[:, None] * normal
    relative = projected - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", relative, ab)
    d21 = np.einsum("ij,ij->i", relative, ac)
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
    plane_distance = np.abs(np.einsum("ij,ij->i", ap, normal)) / np.sqrt(normal_sq)

    def segment_distance(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        direction = end - start
        fraction = np.einsum("ij,ij->i", point - start, direction) / np.einsum(
            "ij,ij->i", direction, direction
        )
        closest = start + np.clip(fraction, 0.0, 1.0)[:, None] * direction
        return np.linalg.norm(point - closest, axis=1)

    edges = np.minimum.reduce((
        segment_distance(a, b), segment_distance(b, c), segment_distance(c, a)
    ))
    return np.where(inside, plane_distance, edges)


def reference_origin(refined: np.ndarray) -> np.ndarray:
    """Derive the frozen numerical origin solely from the refined reference AABB."""
    if refined.ndim != 2 or refined.shape[1] != 3 or not len(refined) or not np.isfinite(refined).all():
        raise EvaluatorFailure("invalid refined reference")
    return np.floor(((refined.min(axis=0) + refined.max(axis=0)) * 0.5) / 100.0) * 100.0


def metrics_from_arrays(
    vertices: np.ndarray, triangles: np.ndarray, full_lidar: np.ndarray,
    refined_lidar: np.ndarray, *, sample_count: int = 1000000, seed: int = 20260916,
    check: Callable[[], None] = lambda: None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute formal derived metrics for in-memory geometry/reference arrays."""
    if not np.isfinite(full_lidar).all() or not len(full_lidar):
        raise EvaluatorFailure("invalid full LiDAR")
    progress("sample_surface_start", count=sample_count)
    samples = sample_surface(vertices, triangles, sample_count, seed, check)
    progress("full_lidar_tree_start", points=len(full_lidar))
    tree = KDTree(full_lidar)
    progress("forward_distances_start", count=sample_count)
    distances = np.empty(sample_count, dtype=np.float64)
    chunk = min(100000, sample_count)
    for start in range(0, sample_count, chunk):
        check()
        values = tree.query(samples[start:start + chunk], workers=1)[0]
        if not np.isfinite(values).all():
            raise EvaluatorFailure("non-finite KD-tree distance")
        distances[start:start + len(values)] = values
    best_count = math.ceil(0.9 * sample_count)
    best = distances[np.lexsort((np.arange(sample_count), distances))[:best_count]]
    progress("forward_distances_complete", count=sample_count)
    origin = reference_origin(refined_lidar)
    reverse = raycast_distances(vertices, triangles, refined_lidar, origin, check=check)
    metrics = {
        "accuracy_l1_m": float(np.mean(distances)),
        "accuracy_rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "accuracy_best90_l1_m": float(np.mean(best)),
        "accuracy_best90_rmse_m": float(np.sqrt(np.mean(np.square(best)))),
        "accuracy_best90_count": best_count,
        "precision_0_20": float(np.mean(distances <= 0.20)),
        "completeness_0_20": float(np.mean(reverse <= 0.20)),
    }
    if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, float)):
        raise EvaluatorFailure("non-finite metric")
    return metrics, origin


def _las_points(path: Path, check: Callable[[], None] = lambda: None) -> np.ndarray:
    parts: list[np.ndarray] = []
    try:
        with laspy.open(path) as reader:
            for block in reader.chunk_iterator(100000):
                check()
                values = np.column_stack((block.x, block.y, block.z)).astype(np.float64)
                if not np.isfinite(values).all():
                    raise EvaluatorFailure("non-finite LAS reference")
                parts.append(values)
    except EvaluatorFailure:
        raise
    except Exception as exc:
        raise EvaluatorFailure(f"cannot read LAS: {path.name}") from exc
    if not parts:
        raise EvaluatorFailure(f"empty LAS: {path.name}")
    return np.concatenate(parts)


def _bindings(bundle: Path, bundle_lock: Path, scene: str, snapshot: dict[str, Any]) -> dict[str, str]:
    manifest = _strict_json(bundle / "manifest.json", "BUNDLE_INVALID")
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "bundle_lock_sha256": sha256_file(bundle_lock),
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "input_manifest_sha256": manifest["scenes"][scene]["input_manifest_sha256"]["rgb-oriented"],
        "reference_manifest_sha256": manifest["scenes"][scene]["reference_manifest_sha256"],
        "submission_contract_sha256": snapshot["contract_sha256"],
        "mesh_sha256": snapshot["mesh_sha256"],
        "runtime_implementation_id": load_protocol()["runtime"]["implementation_id"],
    }


def validate_submission(bundle: Path, bundle_lock: Path, scene: str, submission: Path) -> dict[str, Any]:
    """Verify bundle, snapshot bytes, and validate the strict mesh."""
    runtime_versions()
    verify_bundle(bundle, bundle_lock)
    with tempfile.TemporaryDirectory(prefix=".usegeo-mesh-validate-", dir=submission.parent) as temporary:
        snapshot = snapshot_submission(submission, Path(temporary), scene)
        vertices, triangles = read_ply(Path(temporary) / "mesh.ply")
        return {
            "status": "valid", "scene_id": scene, "track": "rgb-oriented",
            "configuration_id": snapshot["contract"]["configuration_id"],
            "counts": {"vertices": len(vertices), "triangles": len(triangles)},
            "bindings": _bindings(bundle, bundle_lock, scene, snapshot),
        }


def _peak_rss_bytes(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
            check=False, timeout=2,
        )
        value = result.stdout.strip()
        return int(value) * 1024 if result.returncode == 0 and value else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _check_worker_limits(deadline: float, cancel: Path) -> None:
    if cancel.is_file():
        raise EvaluatorFailure(f"worker cancelled: {cancel.read_text(encoding='utf-8').strip()}")
    if time.monotonic() > deadline:
        raise EvaluatorFailure("worker deadline exceeded")


def _watch_process(
    process: subprocess.Popen[str], started: float, timeout: float, rss_limit: int,
    period: float, cancel: Path,
) -> tuple[int, str, str]:
    """Observe RSS fail-closed and signal the worker at its next chunk boundary."""
    peak = observations = failures = 0
    failure = ""
    while process.poll() is None:
        rss = _peak_rss_bytes(process.pid)
        if rss is None:
            failures += 1
            if failures >= int(load_protocol()["limits"]["max_rss_measurement_failures"]):
                failure = "worker RSS measurement unavailable"
        else:
            observations += 1
            failures = 0
            peak = max(peak, rss)
            if peak > rss_limit:
                failure = "worker RSS limit exceeded"
        if time.monotonic() - started > timeout:
            failure = "worker scene timeout exceeded"
        if failure:
            _write_json(cancel, {"reason": failure})
            try:
                process.wait(timeout=max(2.0, period * 4))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout, stderr = process.communicate()
            exc = EvaluatorFailure(failure)
            exc.peak_rss_bytes = peak if observations else None
            progress("watchdog_failure", reason=failure, peak_rss_bytes=exc.peak_rss_bytes, stdout=stdout, stderr=stderr)
            raise exc
        time.sleep(period)
    stdout, stderr = process.communicate()
    if observations == 0:
        raise EvaluatorFailure("worker RSS was never observed")
    return peak, stdout, stderr


def _worker(work_order: Path) -> int:
    global _PROGRESS
    order = _strict_json(work_order, "WORK_ORDER_INVALID")
    token = os.environ.get("USEGEO_MESH_WORKER_TOKEN")
    if not token or token != order.get("token") or order.get("parent_pid") != os.getppid():
        raise EvaluatorFailure("score-worker requires a parent-issued work order")
    _PROGRESS = Path(order["progress"])
    deadline = float(order["deadline"])
    cancel = Path(order["cancel"])

    def check() -> None:
        _check_worker_limits(deadline, cancel)

    mesh = Path(order["mesh"])
    progress("read_mesh_start")
    vertices, triangles = read_ply(mesh)
    progress("read_mesh_complete", vertices=len(vertices), triangles=len(triangles))
    reference = Path(order["bundle"]) / "evaluator-only" / order["scene"]
    progress("read_full_lidar_start")
    full = _las_points(reference / "full_lidar.las", check)
    progress("read_full_lidar_complete", points=len(full))
    progress("read_refined_lidar_start")
    refined = _las_points(reference / "refined_lidar.las", check)
    progress("read_refined_lidar_complete", points=len(refined))
    metrics, origin = metrics_from_arrays(vertices, triangles, full, refined, check=check)
    progress("structure_diagnostics_start")
    result = {
        "counts": {
            "vertices": len(vertices), "triangles": len(triangles),
            "surface_samples": 1000000, "full_lidar_points": len(full),
            "refined_lidar_points": len(refined),
        },
        "surface_metrics": metrics,
        "structure_diagnostics": structure_diagnostics(vertices, triangles),
        "raycast_origin_m": origin.tolist(),
    }
    _write_json(Path(order["result"]), result)
    progress("worker_complete")
    return 0


def _publish_directory(output: Path, files: dict[str, Any]) -> None:
    """Atomically create, or exactly verify, an immutable output directory."""
    if output.exists():
        if output.is_symlink() or not output.is_dir() or set(p.name for p in output.iterdir()) != set(files):
            raise EvaluatorFailure(f"existing output differs: {output}")
        if any((output / name).read_bytes() != _json_bytes(value) for name, value in files.items()):
            raise EvaluatorFailure(f"existing output differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, value in files.items():
            _write_json(temporary / name, value)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def score_submission(
    bundle: Path, bundle_lock: Path, scene: str, submission: Path, output: Path,
    cli_arguments: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run formal metrics in a bounded child and immutably publish its result."""
    global _PROGRESS
    protocol = load_protocol()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS = output.with_name(output.name + ".progress.jsonl")
    with _PROGRESS.open("x"):
        pass
    progress("score_start", implementation_id=protocol["runtime"]["implementation_id"], source_sha256=sha256_file(Path(__file__)))
    bindings = {"protocol_sha256": sha256_file(PROTOCOL_PATH), "runtime_implementation_id": protocol["runtime"]["implementation_id"]}
    peak = None
    scratch: Path | None = None
    process: subprocess.Popen[str] | None = None
    try:
        versions = runtime_versions()
        progress("verify_bundle_start")
        verify_bundle(bundle, bundle_lock)
        progress("verify_bundle_complete")
        scratch = Path(tempfile.mkdtemp(prefix=".usegeo-mesh-score-", dir=output.parent))
        snap = snapshot_submission(submission, scratch / "submission", scene)
        bindings = _bindings(bundle, bundle_lock, scene, snap)
        vertices, triangles = read_ply(scratch / "submission" / "mesh.ply")
        progress("submission_verified", bindings=bindings)
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        order = {
            "progress": str(_PROGRESS.resolve()),
            "token": token, "parent_pid": os.getpid(), "bundle": str(bundle.resolve()),
            "scene": scene, "mesh": str((scratch / "submission" / "mesh.ply").resolve()),
            "result": str((scratch / "worker-result.json").resolve()),
            "cancel": str((scratch / "cancel.json").resolve()),
            "deadline": time.monotonic() + float(protocol["limits"]["scene_timeout_seconds"]),
        }
        _write_json(scratch / "work-order.json", order)
        environment = os.environ.copy()
        environment["USEGEO_MESH_WORKER_TOKEN"] = token
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "score-worker",
             "--work-order", str(scratch / "work-order.json")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
        )
        peak, stdout, stderr = _watch_process(
            process, started, float(protocol["limits"]["scene_timeout_seconds"]),
            int(protocol["limits"]["max_peak_rss_bytes"]),
            float(protocol["limits"]["watchdog_period_seconds"]), scratch / "cancel.json",
        )
        if process.returncode != 0:
            raise EvaluatorFailure(f"score worker failed ({process.returncode}): {stderr.strip() or stdout.strip()}")
        worker = _strict_json(scratch / "worker-result.json", "WORKER_RESULT_INVALID")
        score = {
            "schema_version": SCORE_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "valid", "track": protocol["track"], "scene_id": scene,
            "configuration_id": snap["contract"]["configuration_id"],
            "alignment": {"kind": "identity", "raycast_translation_kind": protocol["raycast"]["origin_rule"],
                          "raycast_origin_m": worker["raycast_origin_m"]},
            "counts": worker["counts"], "surface_metrics": worker["surface_metrics"],
            "structure_diagnostics": worker["structure_diagnostics"], "bindings": bindings,
        }
        score_hash = hashlib.sha256(_json_bytes(score)).hexdigest()
        run = {
            "schema_version": RUN_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "valid", "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_bytes": peak,
            "watchdog": {"period_seconds": protocol["limits"]["watchdog_period_seconds"],
                         "timeout_seconds": protocol["limits"]["scene_timeout_seconds"],
                         "rss_limit_bytes": protocol["limits"]["max_peak_rss_bytes"], "exit_code": 0},
            "sampling": protocol["sampling"], "python": sys.version, "platform": platform.platform(),
            "packages": versions,
            "cli_arguments": list(cli_arguments or []), "method_metadata": snap["contract"]["method_metadata"],
            "bindings": bindings, "output_sha256": {"score.json": score_hash},
        }
        if output.exists():
            existing_score, existing_run = _load_result(output)
            if existing_score != score or existing_run.get("bindings") != bindings:
                raise EvaluatorFailure(f"existing output differs: {output}")
            return 0, existing_score
        _publish_directory(output, {"score.json": score, "run_manifest.json": run})
        return 0, score
    except InvalidInput as exc:
        invalid = {
            "schema_version": SCORE_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "invalid", "track": protocol["track"], "scene_id": scene,
            "configuration_id": None, "invalid_reasons": [{"code": exc.reason, "detail": exc.detail}],
            "bindings": dict(bindings),
        }
        run = {
            "schema_version": RUN_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "invalid", "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_bytes": getattr(exc, "peak_rss_bytes", peak),
            "watchdog": {"exit_code": None}, "sampling": protocol["sampling"],
            "python": sys.version, "platform": platform.platform(), "packages": {},
            "cli_arguments": list(cli_arguments or []), "method_metadata": None,
            "bindings": invalid["bindings"],
            "output_sha256": {"score.json": hashlib.sha256(_json_bytes(invalid)).hexdigest()},
        }
        _publish_directory(output, {"score.json": invalid, "run_manifest.json": run})
        return 2, invalid
    except (EvaluatorFailure, OSError, ValueError) as exc:
        error = {
            "schema_version": SCORE_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "error", "track": protocol["track"], "scene_id": scene,
            "configuration_id": None, "error": type(exc).__name__, "detail": str(exc),
            "bindings": dict(bindings),
        }
        run = {
            "schema_version": RUN_SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "error", "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_bytes": getattr(exc, "peak_rss_bytes", peak),
            "watchdog": {"exit_code": 1}, "sampling": protocol["sampling"],
            "python": sys.version, "platform": platform.platform(),
            "packages": runtime_versions(enforce=False), "cli_arguments": list(cli_arguments or []),
            "method_metadata": None, "bindings": error["bindings"],
            "output_sha256": {"score.json": hashlib.sha256(_json_bytes(error)).hexdigest()},
        }
        _publish_directory(output, {"score.json": error, "run_manifest.json": run})
        return 1, error
    finally:
        progress("score_end")
        if process is not None and process.poll() is None:
            process.kill()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


METRIC_NAMES = (
    "accuracy_l1_m", "accuracy_rmse_m", "accuracy_best90_l1_m",
    "accuracy_best90_rmse_m", "precision_0_20", "completeness_0_20",
)


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_dir() or {p.name for p in path.iterdir()} != {"score.json", "run_manifest.json"}:
        raise InvalidInput("RESULT_FILE_SET_MISMATCH")
    score = _strict_json(path / "score.json", "RESULT_SCHEMA_INVALID")
    run = _strict_json(path / "run_manifest.json", "RESULT_RUN_MANIFEST_INVALID")
    score_keys = {
        "schema_version", "protocol_version", "status", "track", "scene_id",
        "configuration_id", "alignment", "counts", "surface_metrics",
        "structure_diagnostics", "bindings",
    }
    binding_keys = HASH_BINDINGS | {"runtime_implementation_id"}
    if set(score) != score_keys or set(score.get("bindings", {})) != binding_keys:
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    protocol = load_protocol()
    if (score.get("schema_version") != SCORE_SCHEMA
            or score.get("protocol_version") != load_protocol()["protocol_version"]
            or score.get("status") != "valid" or run.get("status") != "valid"
            or run.get("protocol_version") != score["protocol_version"]
            or run.get("bindings") != score["bindings"]):
        raise InvalidInput("AGGREGATE_INVALID_SCENE")
    if (not isinstance(score.get("configuration_id"), str)
            or not pointcloud_v1.CONFIGURATION_ID.fullmatch(score["configuration_id"])):
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    expected_hash = run.get("output_sha256", {}).get("score.json")
    if expected_hash != sha256_file(path / "score.json"):
        raise InvalidInput("RESULT_OUTPUT_HASH_MISMATCH")
    if set(run) != {"schema_version", "protocol_version", "status", "started_at", "finished_at",
                    "elapsed_seconds", "peak_worker_rss_bytes", "watchdog", "sampling", "python",
                    "platform", "packages", "cli_arguments", "method_metadata", "bindings",
                    "output_sha256"} or run.get("schema_version") != RUN_SCHEMA:
        raise InvalidInput("RESULT_RUN_MANIFEST_INVALID")
    if run.get("packages") != runtime_versions() or run.get("sampling") != protocol["sampling"]:
        raise InvalidInput("RESULT_RUNTIME_MISMATCH")
    if (not isinstance(run.get("python"), str) or not isinstance(run.get("platform"), str)
            or not isinstance(run.get("started_at"), str) or not isinstance(run.get("finished_at"), str)
            or not _nonnegative_number(run.get("elapsed_seconds"))
            or not _positive_integer(run.get("peak_worker_rss_bytes"))
            or run["peak_worker_rss_bytes"] > protocol["limits"]["max_peak_rss_bytes"]
            or run["elapsed_seconds"] > protocol["limits"]["scene_timeout_seconds"]
            or not isinstance(run.get("cli_arguments"), list)
            or not all(isinstance(value, str) for value in run["cli_arguments"])
            or not isinstance(run.get("method_metadata"), dict)
            or set(run.get("output_sha256", {})) != {"score.json"}):
        raise InvalidInput("RESULT_RUN_MANIFEST_INVALID")
    watchdog = run.get("watchdog")
    if (not isinstance(watchdog, dict)
            or set(watchdog) != {"period_seconds", "timeout_seconds", "rss_limit_bytes", "exit_code"}
            or watchdog != {"period_seconds": protocol["limits"]["watchdog_period_seconds"],
                            "timeout_seconds": protocol["limits"]["scene_timeout_seconds"],
                            "rss_limit_bytes": protocol["limits"]["max_peak_rss_bytes"], "exit_code": 0}):
        raise InvalidInput("RESULT_RUN_MANIFEST_INVALID")
    alignment = score.get("alignment")
    if (not isinstance(alignment, dict)
            or set(alignment) != {"kind", "raycast_translation_kind", "raycast_origin_m"}
            or alignment.get("kind") != "identity"
            or alignment.get("raycast_translation_kind") != protocol["raycast"]["origin_rule"]
            or not _finite_vector(alignment.get("raycast_origin_m"), 3)):
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    counts = score.get("counts")
    expected_count_keys = {"vertices", "triangles", "surface_samples", "full_lidar_points",
                           "refined_lidar_points"}
    if (not isinstance(counts, dict) or set(counts) != expected_count_keys
            or not all(_positive_integer(value) for value in counts.values())
            or counts["surface_samples"] != protocol["sampling"]["count"]
            or counts["vertices"] > protocol["limits"]["max_vertices"]
            or counts["triangles"] > protocol["limits"]["max_triangles"]):
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    diagnostics = score.get("structure_diagnostics")
    if (not isinstance(diagnostics, dict) or set(diagnostics) != {
            "surface_area_m2", "degenerate_triangle_count", "unreferenced_vertex_count",
            "boundary_edge_count", "boundary_length_m", "nonmanifold_edge_count",
            "connected_component_count"}
            or not _nonnegative_number(diagnostics["surface_area_m2"])
            or not _nonnegative_number(diagnostics["boundary_length_m"])
            or not all(_nonnegative_integer(diagnostics[name]) for name in diagnostics
                       if name not in {"surface_area_m2", "boundary_length_m"})
            or diagnostics["surface_area_m2"] <= 0 or diagnostics["connected_component_count"] <= 0
            or diagnostics["degenerate_triangle_count"] != 0):
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    metrics = score.get("surface_metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES) | {"accuracy_best90_count"} or any(
        not isinstance(metrics.get(name), (int, float)) or isinstance(metrics.get(name), bool)
        or not math.isfinite(metrics[name]) for name in METRIC_NAMES
    ):
        raise InvalidInput("RESULT_NONFINITE_METRIC")
    if (not _positive_integer(metrics["accuracy_best90_count"])
            or metrics["accuracy_best90_count"] != protocol["sampling"]["trim_count"]
            or any(metrics[name] < 0 for name in METRIC_NAMES[:4])
            or any(not 0 <= metrics[name] <= 1 for name in METRIC_NAMES[4:])):
        raise InvalidInput("RESULT_SCHEMA_INVALID")
    for name, value in score["bindings"].items():
        if (name in HASH_BINDINGS and not _sha256(value)) or (
                name == "runtime_implementation_id" and value != protocol["runtime"]["implementation_id"]):
            raise InvalidInput("RESULT_SCHEMA_INVALID")
    return score, run


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_vector(value: Any, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(item) for item in value))


def aggregate_scores(
    bundle: Path, bundle_lock: Path, configuration_id: str,
    results: list[Path], output: Path,
) -> dict[str, Any]:
    """Verify and macro-average exactly one valid result per frozen scene."""
    protocol = load_protocol()
    runtime_versions()
    manifest = verify_bundle(bundle, bundle_lock)
    loaded = [(_load_result(path), path) for path in results]
    scores = [pair[0][0] for pair in loaded]
    if len(scores) != 3 or {s.get("scene_id") for s in scores} != set(protocol["scenes"]):
        raise InvalidInput("AGGREGATE_SCENE_SET_MISMATCH")
    if any(s.get("track") != "rgb-oriented" for s in scores):
        raise InvalidInput("AGGREGATE_TRACK_MISMATCH")
    if any(s.get("configuration_id") != configuration_id for s in scores):
        raise InvalidInput("AGGREGATE_CONFIGURATION_MISMATCH")
    for score in scores:
        scene = score["scene_id"]
        expected = {
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "bundle_lock_sha256": sha256_file(bundle_lock),
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "input_manifest_sha256": manifest["scenes"][scene]["input_manifest_sha256"]["rgb-oriented"],
            "reference_manifest_sha256": manifest["scenes"][scene]["reference_manifest_sha256"],
            "runtime_implementation_id": protocol["runtime"]["implementation_id"],
        }
        if any(score["bindings"].get(key) != value for key, value in expected.items()):
            raise InvalidInput("AGGREGATE_BINDING_MISMATCH")
    bindings = {(s["bindings"]["protocol_sha256"], s["bindings"]["bundle_lock_sha256"],
                 s["bindings"]["bundle_manifest_sha256"],
                 s["bindings"]["runtime_implementation_id"]) for s in scores}
    if len(bindings) != 1:
        raise InvalidInput("AGGREGATE_BINDING_MISMATCH")
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA, "protocol_version": protocol["protocol_version"],
        "status": "valid", "track": "rgb-oriented", "configuration_id": configuration_id,
        "scene_ids": sorted(protocol["scenes"]),
        "macro_mean": {name: float(np.mean([s["surface_metrics"][name] for s in scores])) for name in METRIC_NAMES},
        "result_sha256": {s["scene_id"]: {"score.json": sha256_file(path / "score.json"),
                                             "run_manifest.json": sha256_file(path / "run_manifest.json")}
                          for ((s, _), path) in loaded},
    }
    if output.exists():
        if output.is_symlink() or output.read_bytes() != _json_bytes(aggregate):
            raise EvaluatorFailure(f"existing aggregate differs: {output}")
    else:
        _write_json(output, aggregate)
    return aggregate


def _iter_las(path: Path) -> Iterable[np.ndarray]:
    with laspy.open(path) as reader:
        for block in reader.chunk_iterator(100000):
            values = np.column_stack((block.x, block.y, block.z)).astype(np.float64)
            if not np.isfinite(values).all():
                raise EvaluatorFailure("non-finite publisher MVS point")
            yield values


def _fixture_scene(source: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = 0
    minimum = np.array([np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf])
    for points in _iter_las(source):
        count += len(points)
        minimum = np.minimum(minimum, points[:, :2].min(axis=0))
        maximum = np.maximum(maximum, points[:, :2].max(axis=0))
    if not count:
        raise EvaluatorFailure("empty publisher MVS LAS")
    origin = np.floor(minimum)
    cells: dict[tuple[int, int], list[float]] = {}
    for points in _iter_las(source):
        indices = np.floor(points[:, :2] - origin).astype(np.int64)
        unique, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
        sums = np.column_stack([
            np.bincount(inverse, weights=points[:, axis], minlength=len(unique)) for axis in range(3)
        ])
        for (ix, iy), cell_count, xyz_sum in zip(unique, counts, sums):
            item = cells.setdefault((int(ix), int(iy)), [0.0, 0.0, 0.0, 0.0])
            item[0] += int(cell_count)
            item[1] += float(xyz_sum[0]); item[2] += float(xyz_sum[1]); item[3] += float(xyz_sum[2])
    keys = sorted(cells)
    lookup = {key: index for index, key in enumerate(keys)}
    vertices = np.asarray([[v[1] / v[0], v[2] / v[0], v[3] / v[0]] for v in (cells[k] for k in keys)])
    triangles: list[tuple[int, int, int]] = []
    for x, y in keys:
        square = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
        if all(key in lookup for key in square):
            lower_left, lower_right, upper_left, upper_right = (lookup[key] for key in square)
            triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    if not triangles:
        raise EvaluatorFailure("fixture has no complete occupied 2x2 cells")
    return vertices, np.asarray(triangles, dtype=np.int64), {
        "publisher_mvs_point_count": count, "xy_bounds": [minimum.tolist(), maximum.tolist()],
        "grid_origin_m": origin.tolist(), "nonempty_cell_count": len(cells),
    }


def generate_real_fixtures(
    bundle: Path, bundle_lock: Path, output: Path, configuration_id: str,
    cli_arguments: list[str] | None = None,
) -> dict[str, Any]:
    """Generate serial full-publisher-MVS 2.5D triangle fixtures."""
    protocol = load_protocol()
    runtime_versions()
    manifest = verify_bundle(bundle, bundle_lock)
    if configuration_id != protocol["fixture"]["configuration_id"]:
        raise InvalidInput("MALFORMED_CONTRACT")
    if output.exists():
        # Adjacent hashes authenticate only the caller-controlled tree.  Build
        # the frozen source derivation independently and compare its stable
        # mesh/contract/scene records before allowing reuse.
        actual = _validate_fixture_manifest(bundle, bundle_lock, output, configuration_id)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-verify-", dir=output.parent))
        try:
            expected = _generate_fixture_tree(bundle, bundle_lock, temporary, configuration_id,
                                              cli_arguments)
            if (actual["scenes"] != expected["scenes"]
                    or any((output / scene / name).read_bytes() != (temporary / scene / name).read_bytes()
                           for scene in protocol["scenes"] for name in ("mesh.ply", "submission.json"))):
                raise InvalidInput("FIXTURE_DERIVATION_MISMATCH")
            return actual
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        result = _generate_fixture_tree(bundle, bundle_lock, temporary, configuration_id, cli_arguments)
        os.replace(temporary, output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _generate_fixture_tree(
    bundle: Path, bundle_lock: Path, temporary: Path, configuration_id: str,
    cli_arguments: list[str] | None,
) -> dict[str, Any]:
    """Write one deterministic fixture derivation into an empty private directory."""
    protocol = load_protocol()
    records: dict[str, Any] = {}
    for scene in sorted(protocol["scenes"]):
        source = bundle / "evaluator-only" / scene / "publisher_mvs.las"
        vertices, triangles, stats = _fixture_scene(source)
        scene_dir = temporary / scene
        scene_dir.mkdir()
        write_ply(scene_dir / "mesh.ply", vertices, triangles)
        read_ply(scene_dir / "mesh.ply")
        contract = {
                "schema_version": protocol["submission_schema_version"],
                "protocol_version": protocol["protocol_version"], "track": "rgb-oriented",
                "scene_id": scene, "product": "mesh", "geometry": "mesh.ply",
                "configuration_id": configuration_id,
                "method_metadata": {"name": protocol["fixture"]["label"], "family": "non-method"},
        }
        _write_json(scene_dir / "submission.json", contract)
        source_record = _strict_json(bundle / "evaluator-only" / scene / "reference_manifest.json", "REFERENCE_MANIFEST_INVALID")["files"]["publisher_mvs"]
        records[scene] = {
                **stats, "source_archive_sha256": protocol["archive_sha256"][protocol["scenes"][scene]["archive"]],
                "publisher_mvs_sha256": source_record["sha256"], "grid_resolution_m": 1.0,
                "vertex_count": len(vertices), "triangle_count": len(triangles),
                "mesh_sha256": sha256_file(scene_dir / "mesh.ply"),
                "contract_sha256": sha256_file(scene_dir / "submission.json"),
                "algorithm_id": protocol["fixture"]["algorithm_id"], "policy": protocol["fixture"]["policy"],
                "label": protocol["fixture"]["label"],
        }
    result = {
        "schema_version": "usegeo-mesh-fixture-manifest-1.0",
        "protocol_version": protocol["protocol_version"], "configuration_id": configuration_id,
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "bundle_lock_sha256": sha256_file(bundle_lock), "algorithm_id": protocol["fixture"]["algorithm_id"],
        "generation_cli": list(cli_arguments or []), "runtime": protocol["runtime"], "scenes": records,
    }
    _write_json(temporary / "fixture_manifest.json", result)
    return result


def _validate_fixture_manifest(
    bundle: Path, bundle_lock: Path, output: Path, configuration_id: str,
) -> dict[str, Any]:
    """Accept reuse only for the exact current immutable three-scene fixture."""
    protocol = load_protocol()
    if (output.is_symlink() or not output.is_dir()
            or {item.name for item in output.iterdir()} != set(protocol["scenes"]) | {"fixture_manifest.json"}):
        raise InvalidInput("FIXTURE_FILE_SET_MISMATCH")
    value = _strict_json(_regular(output / "fixture_manifest.json", "FIXTURE_MANIFEST_INVALID"),
                         "FIXTURE_MANIFEST_INVALID")
    if (set(value) != {"schema_version", "protocol_version", "configuration_id",
                       "bundle_manifest_sha256", "bundle_lock_sha256", "algorithm_id",
                       "generation_cli", "runtime", "scenes"}
            or value.get("schema_version") != "usegeo-mesh-fixture-manifest-1.0"
            or value.get("protocol_version") != protocol["protocol_version"]
            or value.get("configuration_id") != configuration_id
            or value.get("bundle_manifest_sha256") != sha256_file(bundle / "manifest.json")
            or value.get("bundle_lock_sha256") != sha256_file(bundle_lock)
            or value.get("algorithm_id") != protocol["fixture"]["algorithm_id"]
            or value.get("runtime") != protocol["runtime"]
            or not isinstance(value.get("generation_cli"), list)
            or not all(isinstance(item, str) for item in value["generation_cli"])
            or not isinstance(value.get("scenes"), dict)
            or set(value["scenes"]) != set(protocol["scenes"])):
        raise InvalidInput("FIXTURE_MANIFEST_INVALID")
    record_keys = {
        "publisher_mvs_point_count", "xy_bounds", "grid_origin_m", "nonempty_cell_count",
        "source_archive_sha256", "publisher_mvs_sha256", "grid_resolution_m", "vertex_count",
        "triangle_count", "mesh_sha256", "contract_sha256", "algorithm_id", "policy", "label",
    }
    for scene, record in value["scenes"].items():
        scene_dir = output / scene
        if (scene_dir.is_symlink() or not scene_dir.is_dir()
                or {item.name for item in scene_dir.iterdir()} != {"mesh.ply", "submission.json"}
                or not isinstance(record, dict) or set(record) != record_keys):
            raise InvalidInput("FIXTURE_FILE_SET_MISMATCH")
        mesh_path = _regular(scene_dir / "mesh.ply", "FIXTURE_FILE_SET_MISMATCH")
        contract_path = _regular(scene_dir / "submission.json", "FIXTURE_FILE_SET_MISMATCH")
        vertices, triangles = read_ply(mesh_path)
        contract = _contract(contract_path, scene)
        reference = _strict_json(bundle / "evaluator-only" / scene / "reference_manifest.json",
                                 "REFERENCE_MANIFEST_INVALID")["files"]["publisher_mvs"]
        expected = {
            "source_archive_sha256": protocol["archive_sha256"][protocol["scenes"][scene]["archive"]],
            "publisher_mvs_sha256": reference["sha256"], "grid_resolution_m": 1.0,
            "vertex_count": len(vertices), "triangle_count": len(triangles),
            "mesh_sha256": sha256_file(mesh_path), "contract_sha256": sha256_file(contract_path),
            "algorithm_id": protocol["fixture"]["algorithm_id"], "policy": protocol["fixture"]["policy"],
            "label": protocol["fixture"]["label"],
        }
        if (contract["configuration_id"] != configuration_id
                or contract["method_metadata"] != {"name": protocol["fixture"]["label"],
                                                    "family": "non-method"}
                or any(record.get(key) != expected_value for key, expected_value in expected.items())
                or not _positive_integer(record.get("publisher_mvs_point_count"))
                or not _positive_integer(record.get("nonempty_cell_count"))
                or not _finite_vector(record.get("grid_origin_m"), 2)
                or not isinstance(record.get("xy_bounds"), list)
                or len(record["xy_bounds"]) != 2
                or not all(_finite_vector(item, 2) for item in record["xy_bounds"])):
            raise InvalidInput("FIXTURE_MANIFEST_INVALID")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UseGeo strict RGB-oriented mesh benchmark v1")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-submission", "score"):
        item = commands.add_parser(name)
        item.add_argument("--bundle", type=Path, required=True)
        item.add_argument("--bundle-lock", type=Path, required=True)
        item.add_argument("--scene", choices=["Dataset-1", "Dataset-2", "Dataset-3"], required=True)
        item.add_argument("--submission", type=Path, required=True)
        if name == "score":
            item.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--bundle", type=Path, required=True)
    aggregate.add_argument("--bundle-lock", type=Path, required=True)
    aggregate.add_argument("--configuration-id", required=True)
    aggregate.add_argument("--results", type=Path, nargs=3, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    fixture = commands.add_parser("generate-real-fixtures")
    fixture.add_argument("--bundle", type=Path, required=True)
    fixture.add_argument("--bundle-lock", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--configuration-id", required=True)
    worker = commands.add_parser("score-worker", help=argparse.SUPPRESS)
    worker.add_argument("--work-order", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "score-worker":
            return _worker(arguments.work_order)
        if arguments.command == "validate-submission":
            result = validate_submission(arguments.bundle, arguments.bundle_lock, arguments.scene, arguments.submission)
        elif arguments.command == "score":
            code, result = score_submission(arguments.bundle, arguments.bundle_lock, arguments.scene,
                                              arguments.submission, arguments.output,
                                              list(sys.argv if argv is None else argv))
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return code
        elif arguments.command == "aggregate":
            result = aggregate_scores(arguments.bundle, arguments.bundle_lock, arguments.configuration_id,
                                      arguments.results, arguments.output)
        else:
            result = generate_real_fixtures(arguments.bundle, arguments.bundle_lock, arguments.output,
                                            arguments.configuration_id, list(sys.argv if argv is None else argv))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except InvalidInput as exc:
        print(json.dumps({"status": "invalid", "invalid_reasons": [{"code": exc.reason, "detail": exc.detail}]},
                         sort_keys=True), file=sys.stderr)
        return 2
    except (EvaluatorFailure, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__, "detail": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
