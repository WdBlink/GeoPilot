#!/usr/bin/env python3
"""UseGeo point-cloud benchmark v1 preparation, validation, and scoring CLI."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import stat
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import laspy
import numpy as np
import scipy
from PIL import ExifTags, Image, __version__ as pillow_version
from scipy.spatial import KDTree


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol_v1.json"
SCHEMA_VERSION = "usegeo-score-1.0"
BUNDLE_LOCK_SCHEMA_VERSION = "usegeo-bundle-lock-1.0"
RUN_SCHEMA_VERSION = "usegeo-run-manifest-1.0"
AGGREGATE_SCHEMA_VERSION = "usegeo-track-aggregate-1.0"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CONFIGURATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
ALLOWED_SUBMISSION_KEYS = {
    "schema_version",
    "protocol_version",
    "track",
    "scene_id",
    "product",
    "geometry",
    "camera_centers",
    "configuration_id",
    "method_metadata",
}
FORBIDDEN_INPUT_MARKERS = (
    "depth_resized",
    "lidar_dataset",
    "mvs_dataset",
    "_obs.txt",
    "trajectory",
    "lidar_odm_files",
    "refined_lidar",
    "full_lidar",
    "publisher_mvs",
)


class InvalidInput(Exception):
    """A deterministic bundle or submission rejection (CLI exit 2)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class EvaluatorFailure(Exception):
    """An evaluator/environment failure (CLI exit 1)."""


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the frozen JSON protocol."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_vendor_sources(protocol: dict[str, Any]) -> None:
    """Fail if a byte-pinned upstream source artifact has changed."""
    vendor = HERE / "vendor" / "usegeo-aa689753"
    for name, expected in protocol["upstream"]["files_sha256"].items():
        path = _regular_file(vendor / name, "UPSTREAM_SOURCE_MISSING")
        if sha256_file(path) != expected:
            raise EvaluatorFailure(f"vendored upstream source changed: {name}")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return SHA-256 without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_loads(data: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InvalidInput(reason) from exc
    if not isinstance(value, dict):
        raise InvalidInput(reason)
    return value


def _expect_exact_keys(value: dict[str, Any], keys: set[str], reason: str) -> None:
    if set(value) != keys:
        raise InvalidInput(reason)


def _write_json(path: Path, value: Any) -> None:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_immutable_json(path: Path, value: dict[str, Any]) -> None:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise EvaluatorFailure(f"existing authority file differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular_file(path: Path, reason: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise InvalidInput(reason, f"missing file: {path.name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise InvalidInput(reason, f"not a contained regular file: {path.name}")
    return path


def _contained_regular(root: Path, relative: str, reason: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise InvalidInput(reason, f"unsafe relative path: {relative}")
    root = root.resolve()
    candidate = root.joinpath(*posix.parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InvalidInput(reason, f"path escapes root: {relative}") from exc
    return _regular_file(candidate, reason)


def _snapshot_regular(
    source: Path,
    destination: Path,
    max_bytes: int,
    reason: str,
    too_large_reason: str | None = None,
) -> Path:
    """Copy one no-follow regular file; all later reads use the private copy."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise InvalidInput(reason, f"cannot open regular file: {source.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise InvalidInput(reason, f"not a regular file: {source.name}")
        if metadata.st_size > max_bytes:
            raise InvalidInput(
                too_large_reason or reason,
                f"file exceeds {max_bytes} bytes: {source.name}",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle, destination.open(
            "xb"
        ) as target:
            while chunk := handle.read(8 * 1024 * 1024):
                copied += len(chunk)
                if copied > max_bytes:
                    raise InvalidInput(
                        too_large_reason or reason,
                        f"file grew beyond limit: {source.name}",
                    )
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        return destination
    finally:
        os.close(descriptor)


def _safe_zip_infos(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        unix_mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise EvaluatorFailure(f"unsafe ZIP member: {name}")
        if name in infos:
            raise EvaluatorFailure(f"duplicate ZIP member: {name}")
        if stat.S_ISLNK(unix_mode):
            raise EvaluatorFailure(f"symlink ZIP member: {name}")
        infos[name] = info
    return infos


def _copy_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive.open(info, "r") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    except (zipfile.BadZipFile, EOFError) as exc:
        raise EvaluatorFailure(f"ZIP CRC/read failure: {info.filename}") from exc


def _parse_orientations(data: bytes) -> tuple[list[str], dict[str, np.ndarray]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvaluatorFailure("orientation table is not UTF-8") from exc
    names: list[str] = []
    centers: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 15:
            raise EvaluatorFailure(f"malformed orientation row {line_number}")
        image_id = fields[0]
        if image_id in centers:
            raise EvaluatorFailure(f"duplicate orientation ID: {image_id}")
        try:
            values = np.asarray([float(item) for item in fields[1:]], dtype=np.float64)
        except ValueError as exc:
            raise EvaluatorFailure(f"non-numeric orientation row {line_number}") from exc
        if not np.isfinite(values).all():
            raise EvaluatorFailure(f"non-finite orientation row {line_number}")
        names.append(image_id)
        centers[image_id] = values[:3]
    return names, centers


def _image_metadata(path: Path, expected_dimensions: tuple[int, int]) -> list[str]:
    try:
        with Image.open(path) as image:
            if image.size != expected_dimensions:
                raise EvaluatorFailure(
                    f"unexpected image dimensions for {path.name}: {image.size}"
                )
            exif = image.getexif()
    except (OSError, ValueError) as exc:
        raise EvaluatorFailure(f"malformed JPEG: {path.name}") from exc
    if 34853 in exif:
        raise EvaluatorFailure(f"GPSInfo present in method input: {path.name}")
    return sorted(str(ExifTags.TAGS.get(tag, tag)) for tag in exif.keys())


def _write_centers(path: Path, image_ids: Iterable[str], centers: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "x", "y", "z"])
        for image_id in sorted(image_ids):
            writer.writerow([image_id, *[format(float(v), ".17g") for v in centers[image_id]]])


def _load_las_xyz(path: Path, dimensions: int = 3, chunk_points: int = 1_000_000) -> np.ndarray:
    """Decode a LAS into one contiguous float64 array with bounded read chunks."""
    try:
        with laspy.open(path) as reader:
            count = int(reader.header.point_count)
            if count < 1:
                raise InvalidInput("EMPTY_GEOMETRY")
            result = np.empty((count, dimensions), dtype=np.float64)
            offset = 0
            for points in reader.chunk_iterator(chunk_points):
                size = len(points)
                result[offset : offset + size, 0] = np.asarray(points.x, dtype=np.float64)
                result[offset : offset + size, 1] = np.asarray(points.y, dtype=np.float64)
                if dimensions == 3:
                    result[offset : offset + size, 2] = np.asarray(points.z, dtype=np.float64)
                offset += size
            if offset != count or not np.isfinite(result).all():
                raise InvalidInput("NONFINITE_GEOMETRY")
            return result
    except InvalidInput:
        raise
    except Exception as exc:
        raise InvalidInput("MALFORMED_LAS", str(exc)) from exc


def _iter_las_xyz(path: Path, chunk_points: int) -> Iterable[np.ndarray]:
    try:
        with laspy.open(path) as reader:
            for points in reader.chunk_iterator(chunk_points):
                xyz = np.column_stack((points.x, points.y, points.z)).astype(
                    np.float64, copy=False
                )
                if not np.isfinite(xyz).all():
                    raise InvalidInput("NONFINITE_GEOMETRY")
                yield xyz
    except InvalidInput:
        raise
    except Exception as exc:
        raise InvalidInput("MALFORMED_LAS", str(exc)) from exc


def _validate_las(path: Path, protocol: dict[str, Any]) -> int:
    limits = protocol["limits"]
    _regular_file(path, "GEOMETRY_NOT_REGULAR")
    size = path.stat().st_size
    if size > int(limits["max_submission_bytes"]):
        raise InvalidInput("GEOMETRY_TOO_LARGE")
    try:
        with laspy.open(path) as reader:
            count = int(reader.header.point_count)
            if count < 1:
                raise InvalidInput("EMPTY_GEOMETRY")
            if count > int(limits["max_submission_points"]):
                raise InvalidInput("POINT_LIMIT_EXCEEDED")
            observed = 0
            for points in reader.chunk_iterator(int(limits["las_query_chunk_points"])):
                xyz = np.column_stack((points.x, points.y, points.z))
                if not np.isfinite(xyz).all():
                    raise InvalidInput("NONFINITE_GEOMETRY")
                observed += len(points)
            if observed != count:
                raise InvalidInput("MALFORMED_LAS")
            return count
    except InvalidInput:
        raise
    except Exception as exc:
        raise InvalidInput("MALFORMED_LAS", str(exc)) from exc


def _build_refined_lidar(
    full_lidar: Path,
    publisher_mvs: Path,
    destination: Path,
    threshold: float,
    chunk_points: int,
    workers: int,
) -> int:
    mvs_xy = _load_las_xyz(publisher_mvs, dimensions=2, chunk_points=chunk_points)
    tree = KDTree(mvs_xy, leafsize=20, copy_data=False)
    kept = 0
    try:
        with laspy.open(full_lidar) as reader, laspy.open(
            destination, mode="w", header=reader.header
        ) as writer:
            for points in reader.chunk_iterator(chunk_points):
                xy = np.column_stack((points.x, points.y)).astype(np.float64, copy=False)
                distances, _ = tree.query(xy, workers=workers)
                mask = distances <= threshold
                writer.write_points(points[mask])
                kept += int(np.count_nonzero(mask))
    finally:
        del tree, mvs_xy
        gc.collect()
    if kept < 1:
        raise EvaluatorFailure("fixed refined LiDAR is empty")
    return kept


def _prepare_archives(
    archives: Path,
    output: Path,
    protocol: dict[str, Any],
    *,
    min_free_bytes: int | None = None,
) -> dict[str, Any]:
    """Internal parameterized preparation helper used by production and fixtures."""
    archives = archives.resolve()
    output = output.resolve()
    target = output / "v1"
    protocol_bytes = (
        PROTOCOL_PATH.read_bytes() if protocol == load_protocol() else _json_bytes(protocol)
    )
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    archive_hashes: dict[str, str] = {}
    for scene in sorted(protocol["scenes"]):
        archive_name = protocol["scenes"][scene]["archive"]
        archive_path = _regular_file(archives / archive_name, "ARCHIVE_MISSING")
        observed = sha256_file(archive_path)
        expected = protocol["archive_sha256"][archive_name]
        if observed != expected:
            raise EvaluatorFailure(f"archive SHA-256 mismatch: {archive_name}")
        archive_hashes[archive_name] = observed

    if target.exists():
        existing = _regular_file(target / "manifest.json", "BUNDLE_INVALID")
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_sha256") == protocol_sha
            and manifest.get("archive_sha256") == archive_hashes
        ):
            _verify_bundle_manifest(
                target,
                protocol,
                trusted_manifest_sha256=sha256_file(existing),
                expected_protocol_sha=protocol_sha,
            )
            return manifest
        raise EvaluatorFailure(f"existing bundle differs: {target}")

    required = (
        int(protocol["limits"]["free_space_required_bytes"])
        if min_free_bytes is None
        else min_free_bytes
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < required:
        raise EvaluatorFailure(f"at least {required} free bytes required")

    temporary = Path(tempfile.mkdtemp(prefix=".usegeo-v1-", dir=output))
    file_hashes: dict[str, str] = {}
    scene_manifests: dict[str, Any] = {}
    forbidden_scan: dict[str, list[str]] = {}
    try:
        (temporary / "protocol_v1.json").write_bytes(protocol_bytes)
        file_hashes["protocol_v1.json"] = protocol_sha
        for scene in sorted(protocol["scenes"]):
            spec = protocol["scenes"][scene]
            archive_path = archives / spec["archive"]
            with zipfile.ZipFile(archive_path) as archive:
                infos = _safe_zip_infos(archive)
                forbidden_scan[scene] = sorted(
                    name
                    for name in infos
                    if any(marker in name.lower() for marker in FORBIDDEN_INPUT_MARKERS)
                )
                for required_member in (
                    spec["orientation_member"],
                    spec["full_lidar_member"],
                    spec["publisher_mvs_member"],
                ):
                    if required_member not in infos:
                        raise EvaluatorFailure(f"missing ZIP member: {required_member}")
                orientation_data = archive.read(infos[spec["orientation_member"]])
                orientation_ids, centers = _parse_orientations(orientation_data)
                image_infos = {
                    PurePosixPath(name).name: info
                    for name, info in infos.items()
                    if name.startswith(spec["image_prefix"])
                    and name.lower().endswith(".jpg")
                }
                selected = sorted(set(image_infos).intersection(orientation_ids))
                if len(selected) != int(spec["image_count"]):
                    raise EvaluatorFailure(
                        f"{scene}: expected {spec['image_count']} matched images, got {len(selected)}"
                    )

                local_root = temporary / "inputs" / "rgb-local" / scene
                oriented_root = temporary / "inputs" / "rgb-oriented" / scene
                evaluator_root = temporary / "evaluator-only" / scene
                exif_names: set[str] = set()
                image_records: list[dict[str, Any]] = []
                for image_id in selected:
                    local_image = local_root / "images" / image_id
                    _copy_zip_member(archive, image_infos[image_id], local_image)
                    exif_names.update(
                        _image_metadata(
                            local_image, tuple(protocol["input_image_dimensions"])
                        )
                    )
                    digest = sha256_file(local_image)
                    oriented_image = oriented_root / "images" / image_id
                    oriented_image.parent.mkdir(parents=True, exist_ok=True)
                    os.link(local_image, oriented_image)
                    for path in (local_image, oriented_image):
                        relative = path.relative_to(temporary).as_posix()
                        file_hashes[relative] = digest
                    image_records.append({"image_id": image_id, "sha256": digest})

                orientation_path = oriented_root / PurePosixPath(
                    spec["orientation_member"]
                ).name
                orientation_path.parent.mkdir(parents=True, exist_ok=True)
                orientation_path.write_bytes(orientation_data)
                orientation_sha = sha256_file(orientation_path)
                file_hashes[orientation_path.relative_to(temporary).as_posix()] = orientation_sha

                centers_path = evaluator_root / "publisher_camera_centers.csv"
                _write_centers(centers_path, selected, centers)
                file_hashes[centers_path.relative_to(temporary).as_posix()] = sha256_file(
                    centers_path
                )
                full_lidar = evaluator_root / "full_lidar.las"
                publisher_mvs = evaluator_root / "publisher_mvs.las"
                _copy_zip_member(archive, infos[spec["full_lidar_member"]], full_lidar)
                _copy_zip_member(archive, infos[spec["publisher_mvs_member"]], publisher_mvs)

            refined_lidar = evaluator_root / "refined_lidar.las"
            refined_count = _build_refined_lidar(
                full_lidar,
                publisher_mvs,
                refined_lidar,
                float(protocol["reference_support"]["xy_threshold_m"]),
                int(protocol["limits"]["las_query_chunk_points"]),
                int(protocol["limits"]["query_workers"]),
            )
            reference_files = {}
            for name, path in {
                "full_lidar": full_lidar,
                "publisher_mvs": publisher_mvs,
                "refined_lidar": refined_lidar,
                "publisher_camera_centers": centers_path,
            }.items():
                digest = sha256_file(path)
                relative = path.relative_to(temporary).as_posix()
                file_hashes[relative] = digest
                reference_files[name] = {"path": path.name, "sha256": digest}

            base_input = {
                "schema_version": "usegeo-input-manifest-1.0",
                "protocol_version": protocol["protocol_version"],
                "scene_id": scene,
                "image_count": len(selected),
                "image_dimensions": protocol["input_image_dimensions"],
                "images": image_records,
                "observed_exif_tags": sorted(exif_names),
                "gps_info_present": False,
            }
            local_manifest = {**base_input, "track": "rgb-local", "permissions": ["images"]}
            oriented_manifest = {
                **base_input,
                "track": "rgb-oriented",
                "permissions": ["images", "publisher orientation table"],
                "orientation_sha256": orientation_sha,
            }
            for root, value in ((local_root, local_manifest), (oriented_root, oriented_manifest)):
                path = root / "input_manifest.json"
                _write_json(path, value)
                file_hashes[path.relative_to(temporary).as_posix()] = sha256_file(path)

            reference_manifest = {
                "schema_version": "usegeo-reference-manifest-1.0",
                "protocol_version": protocol["protocol_version"],
                "scene_id": scene,
                "accuracy_reference": "full publisher LiDAR",
                "completeness_support": protocol["reference_support"],
                "files": reference_files,
                "refined_lidar_point_count": refined_count,
            }
            reference_manifest_path = evaluator_root / "reference_manifest.json"
            _write_json(reference_manifest_path, reference_manifest)
            reference_manifest_sha = sha256_file(reference_manifest_path)
            file_hashes[
                reference_manifest_path.relative_to(temporary).as_posix()
            ] = reference_manifest_sha
            scene_manifests[scene] = {
                "image_count": len(selected),
                "input_manifest_sha256": {
                    "rgb-local": sha256_file(local_root / "input_manifest.json"),
                    "rgb-oriented": sha256_file(oriented_root / "input_manifest.json"),
                },
                "reference_manifest_sha256": reference_manifest_sha,
                "refined_lidar_point_count": refined_count,
            }
            gc.collect()

        manifest = {
            "schema_version": "usegeo-bundle-manifest-1.0",
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol_sha,
            "archive_sha256": archive_hashes,
            "scenes": scene_manifests,
            "files": dict(sorted(file_hashes.items())),
            "forbidden_member_scan": forbidden_scan,
            "dataset_license": "CC BY-NC-SA 4.0; do not publish or relicense this bundle",
        }
        _write_json(temporary / "manifest.json", manifest)
        _verify_bundle_manifest(
            temporary,
            protocol,
            trusted_manifest_sha256=sha256_file(temporary / "manifest.json"),
            expected_protocol_sha=protocol_sha,
        )
        os.replace(temporary, target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare(archives: Path, output: Path) -> dict[str, Any]:
    """Prepare all three frozen UseGeo scenes."""
    protocol = load_protocol()
    verify_vendor_sources(protocol)
    manifest = _prepare_archives(archives, output, protocol)
    bundle = output.resolve() / "v1"
    lock_path = output.resolve() / "v1.lock.json"
    lock = _bundle_lock_payload(bundle, manifest, protocol)
    _publish_immutable_json(lock_path, lock)
    return {
        "status": "prepared",
        "protocol_version": protocol["protocol_version"],
        "bundle": str(bundle),
        "bundle_lock": str(lock_path),
        "bundle_manifest_sha256": lock["bundle_manifest_sha256"],
    }


def _bundle_lock_payload(
    bundle: Path, manifest: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_LOCK_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "archive_sha256": protocol["archive_sha256"],
        "upstream_files_sha256": protocol["upstream"]["files_sha256"],
        "scenes": manifest["scenes"],
    }


def _load_bundle_lock(
    bundle: Path, bundle_lock: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    if bundle_lock.is_symlink():
        raise InvalidInput("BUNDLE_LOCK_NOT_REGULAR")
    lock_path = _regular_file(bundle_lock, "BUNDLE_LOCK_NOT_REGULAR").resolve()
    bundle_root = bundle.resolve()
    try:
        lock_path.relative_to(bundle_root)
    except ValueError:
        pass
    else:
        raise InvalidInput("BUNDLE_LOCK_INSIDE_BUNDLE")
    lock = _strict_json_loads(lock_path.read_bytes(), "BUNDLE_LOCK_MALFORMED")
    _expect_exact_keys(
        lock,
        {
            "schema_version",
            "protocol_version",
            "protocol_sha256",
            "bundle_manifest_sha256",
            "archive_sha256",
            "upstream_files_sha256",
            "scenes",
        },
        "BUNDLE_LOCK_SCHEMA_MISMATCH",
    )
    expected = {
        "schema_version": BUNDLE_LOCK_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "archive_sha256": protocol["archive_sha256"],
        "upstream_files_sha256": protocol["upstream"]["files_sha256"],
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise InvalidInput("BUNDLE_LOCK_AUTHORITY_MISMATCH")
    if not isinstance(lock.get("bundle_manifest_sha256"), str) or not HEX_SHA256.fullmatch(
        lock["bundle_manifest_sha256"]
    ):
        raise InvalidInput("BUNDLE_LOCK_SCHEMA_MISMATCH")
    return lock


def _verify_input_manifest(
    bundle: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    track: str,
    scene: str,
) -> tuple[list[str], str]:
    spec = protocol["scenes"][scene]
    root = bundle / "inputs" / track / scene
    manifest_path = _regular_file(root / "input_manifest.json", "INPUT_MANIFEST_INVALID")
    value = _strict_json_loads(manifest_path.read_bytes(), "INPUT_MANIFEST_INVALID")
    common_keys = {
        "schema_version",
        "protocol_version",
        "scene_id",
        "image_count",
        "image_dimensions",
        "images",
        "observed_exif_tags",
        "gps_info_present",
        "track",
        "permissions",
    }
    expected_keys = common_keys | ({"orientation_sha256"} if track == "rgb-oriented" else set())
    _expect_exact_keys(value, expected_keys, "INPUT_MANIFEST_SCHEMA_MISMATCH")
    expected_permissions = (
        ["images", "publisher orientation table"]
        if track == "rgb-oriented"
        else ["images"]
    )
    if (
        value["schema_version"] != "usegeo-input-manifest-1.0"
        or value["protocol_version"] != protocol["protocol_version"]
        or value["scene_id"] != scene
        or value["track"] != track
        or value["image_count"] != int(spec["image_count"])
        or value["image_dimensions"] != protocol["input_image_dimensions"]
        or value["permissions"] != expected_permissions
        or value["gps_info_present"] is not False
        or not isinstance(value["observed_exif_tags"], list)
        or not all(isinstance(tag, str) for tag in value["observed_exif_tags"])
    ):
        raise InvalidInput("INPUT_MANIFEST_RELATION_MISMATCH", f"{track}/{scene}")
    images = value["images"]
    if not isinstance(images, list) or len(images) != int(spec["image_count"]):
        raise InvalidInput("IMAGE_COUNT_MISMATCH", f"{track}/{scene}")
    image_ids: list[str] = []
    for item in images:
        if not isinstance(item, dict) or set(item) != {"image_id", "sha256"}:
            raise InvalidInput("INPUT_IMAGE_RECORD_INVALID", f"{track}/{scene}")
        image_id, expected_hash = item["image_id"], item["sha256"]
        if (
            not isinstance(image_id, str)
            or PurePosixPath(image_id).name != image_id
            or not isinstance(expected_hash, str)
            or not HEX_SHA256.fullmatch(expected_hash)
        ):
            raise InvalidInput("INPUT_IMAGE_RECORD_INVALID", f"{track}/{scene}")
        image = _contained_regular(root / "images", image_id, "INPUT_IMAGE_INVALID")
        relative = image.relative_to(bundle).as_posix()
        if manifest["files"].get(relative) != expected_hash or sha256_file(image) != expected_hash:
            raise InvalidInput("INPUT_IMAGE_HASH_MISMATCH", f"{track}/{scene}/{image_id}")
        image_ids.append(image_id)
    if image_ids != sorted(set(image_ids)):
        raise InvalidInput("INPUT_IMAGE_ID_SET_MISMATCH", f"{track}/{scene}")
    actual_ids = sorted(path.name for path in (root / "images").iterdir() if path.is_file())
    if actual_ids != image_ids:
        raise InvalidInput("INPUT_IMAGE_ID_SET_MISMATCH", f"{track}/{scene}")
    if track == "rgb-oriented":
        orientation = _regular_file(
            root / PurePosixPath(spec["orientation_member"]).name,
            "ORIENTATION_FILE_INVALID",
        )
        orientation_hash = sha256_file(orientation)
        if (
            value["orientation_sha256"] != orientation_hash
            or manifest["files"].get(orientation.relative_to(bundle).as_posix())
            != orientation_hash
        ):
            raise InvalidInput("ORIENTATION_HASH_MISMATCH", scene)
    manifest_hash = sha256_file(manifest_path)
    if manifest["files"].get(manifest_path.relative_to(bundle).as_posix()) != manifest_hash:
        raise InvalidInput("INPUT_MANIFEST_HASH_MISMATCH", f"{track}/{scene}")
    return image_ids, manifest_hash


def _verify_reference_manifest(
    bundle: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    scene: str,
    expected_ids: list[str],
) -> tuple[dict[str, Any], str]:
    root = bundle / "evaluator-only" / scene
    path = _regular_file(root / "reference_manifest.json", "REFERENCE_MANIFEST_INVALID")
    value = _strict_json_loads(path.read_bytes(), "REFERENCE_MANIFEST_INVALID")
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "protocol_version",
            "scene_id",
            "accuracy_reference",
            "completeness_support",
            "files",
            "refined_lidar_point_count",
        },
        "REFERENCE_MANIFEST_SCHEMA_MISMATCH",
    )
    expected_names = {
        "full_lidar": "full_lidar.las",
        "publisher_mvs": "publisher_mvs.las",
        "refined_lidar": "refined_lidar.las",
        "publisher_camera_centers": "publisher_camera_centers.csv",
    }
    if (
        value["schema_version"] != "usegeo-reference-manifest-1.0"
        or value["protocol_version"] != protocol["protocol_version"]
        or value["scene_id"] != scene
        or value["accuracy_reference"] != "full publisher LiDAR"
        or value["completeness_support"] != protocol["reference_support"]
        or not isinstance(value["files"], dict)
        or set(value["files"]) != set(expected_names)
        or not isinstance(value["refined_lidar_point_count"], int)
        or isinstance(value["refined_lidar_point_count"], bool)
        or value["refined_lidar_point_count"] < 1
    ):
        raise InvalidInput("REFERENCE_MANIFEST_RELATION_MISMATCH", scene)
    for role, filename in expected_names.items():
        record = value["files"][role]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise InvalidInput("REFERENCE_FILE_RECORD_INVALID", f"{scene}/{role}")
        reference = _regular_file(root / filename, "REFERENCE_FILE_INVALID")
        observed_hash = sha256_file(reference)
        relative = reference.relative_to(bundle).as_posix()
        if (
            record["path"] != filename
            or record["sha256"] != observed_hash
            or manifest["files"].get(relative) != observed_hash
        ):
            raise InvalidInput("REFERENCE_FILE_HASH_MISMATCH", f"{scene}/{role}")
    try:
        with laspy.open(root / "refined_lidar.las") as reader:
            refined_count = int(reader.header.point_count)
    except Exception as exc:
        raise InvalidInput("REFERENCE_FILE_INVALID", f"{scene}/refined_lidar") from exc
    if refined_count != value["refined_lidar_point_count"]:
        raise InvalidInput("REFERENCE_POINT_COUNT_MISMATCH", scene)
    _read_camera_centers(root / "publisher_camera_centers.csv", expected_ids, "REFERENCE_CAMERA")
    observed_hash = sha256_file(path)
    if manifest["files"].get(path.relative_to(bundle).as_posix()) != observed_hash:
        raise InvalidInput("REFERENCE_MANIFEST_HASH_MISMATCH", scene)
    return value, observed_hash


def _verify_bundle_manifest(
    bundle: Path,
    protocol: dict[str, Any],
    *,
    trusted_manifest_sha256: str,
    expected_protocol_sha: str,
) -> dict[str, Any]:
    if bundle.is_symlink():
        raise InvalidInput("BUNDLE_SYMLINK")
    bundle = bundle.resolve()
    manifest_path = _regular_file(bundle / "manifest.json", "BUNDLE_INVALID")
    if sha256_file(manifest_path) != trusted_manifest_sha256:
        raise InvalidInput("BUNDLE_AUTHORITY_MISMATCH")
    manifest = _strict_json_loads(manifest_path.read_bytes(), "BUNDLE_INVALID")
    _expect_exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_version",
            "protocol_sha256",
            "archive_sha256",
            "scenes",
            "files",
            "forbidden_member_scan",
            "dataset_license",
        },
        "BUNDLE_MANIFEST_SCHEMA_MISMATCH",
    )
    protocol_file = _regular_file(bundle / "protocol_v1.json", "BUNDLE_INVALID")
    protocol_sha = sha256_file(protocol_file)
    if (
        manifest["schema_version"] != "usegeo-bundle-manifest-1.0"
        or protocol_sha != expected_protocol_sha
        or manifest["protocol_sha256"] != protocol_sha
        or manifest["protocol_version"] != protocol["protocol_version"]
        or manifest["archive_sha256"] != protocol["archive_sha256"]
        or not isinstance(manifest["files"], dict)
        or set(manifest["scenes"]) != set(protocol["scenes"])
        or set(manifest["forbidden_member_scan"]) != set(protocol["scenes"])
    ):
        raise InvalidInput("BUNDLE_MANIFEST_RELATION_MISMATCH")
    for relative, expected_hash in manifest["files"].items():
        if not isinstance(expected_hash, str) or not HEX_SHA256.fullmatch(expected_hash):
            raise InvalidInput("BUNDLE_FILE_RECORD_INVALID", str(relative))
        file = _contained_regular(bundle, relative, "BUNDLE_FILE_INVALID")
        if sha256_file(file) != expected_hash:
            raise InvalidInput("BUNDLE_HASH_MISMATCH", relative)
    actual_files: set[str] = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise InvalidInput("BUNDLE_SYMLINK", path.name)
        if path.is_file():
            actual_files.add(path.relative_to(bundle).as_posix())
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise InvalidInput("BUNDLE_FILE_SET_MISMATCH")

    for scene, spec in protocol["scenes"].items():
        scene_record = manifest["scenes"].get(scene)
        if not isinstance(scene_record, dict) or set(scene_record) != {
            "image_count",
            "input_manifest_sha256",
            "reference_manifest_sha256",
            "refined_lidar_point_count",
        }:
            raise InvalidInput("SCENE_MANIFEST_SCHEMA_MISMATCH", scene)
        local_ids, local_hash = _verify_input_manifest(
            bundle, manifest, protocol, "rgb-local", scene
        )
        oriented_ids, oriented_hash = _verify_input_manifest(
            bundle, manifest, protocol, "rgb-oriented", scene
        )
        if local_ids != oriented_ids:
            raise InvalidInput("TRACK_IMAGE_SET_MISMATCH", scene)
        for image_id in local_ids:
            local = bundle / "inputs" / "rgb-local" / scene / "images" / image_id
            oriented = bundle / "inputs" / "rgb-oriented" / scene / "images" / image_id
            if sha256_file(local) != sha256_file(oriented):
                raise InvalidInput("TRACK_IMAGE_HASH_MISMATCH", f"{scene}/{image_id}")
        _, reference_hash = _verify_reference_manifest(
            bundle, manifest, protocol, scene, local_ids
        )
        if (
            scene_record["image_count"] != int(spec["image_count"])
            or scene_record["input_manifest_sha256"]
            != {"rgb-local": local_hash, "rgb-oriented": oriented_hash}
            or scene_record["reference_manifest_sha256"] != reference_hash
            or scene_record["refined_lidar_point_count"]
            != _strict_json_loads(
                (bundle / "evaluator-only" / scene / "reference_manifest.json").read_bytes(),
                "REFERENCE_MANIFEST_INVALID",
            )["refined_lidar_point_count"]
        ):
            raise InvalidInput("SCENE_MANIFEST_RELATION_MISMATCH", scene)
    return manifest


def verify_bundle(bundle: Path, bundle_lock: Path) -> dict[str, Any]:
    """Verify a production bundle against an external organizer authority lock."""
    protocol = load_protocol()
    verify_vendor_sources(protocol)
    lock = _load_bundle_lock(bundle, bundle_lock, protocol)
    manifest = _verify_bundle_manifest(
        bundle,
        protocol,
        trusted_manifest_sha256=lock["bundle_manifest_sha256"],
        expected_protocol_sha=lock["protocol_sha256"],
    )
    if lock["scenes"] != manifest["scenes"]:
        raise InvalidInput("BUNDLE_LOCK_SCENE_MISMATCH")
    return manifest


def _read_camera_centers(
    path: Path, expected_ids: list[str], reason_prefix: str = "CAMERA"
) -> np.ndarray:
    _regular_file(path, f"{reason_prefix}_CENTERS_NOT_REGULAR")
    values: dict[str, np.ndarray] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) != ["image_id", "x", "y", "z"]:
                raise InvalidInput(f"{reason_prefix}_CENTERS_HEADER")
            for row in reader:
                if len(row) != 4:
                    raise InvalidInput(f"{reason_prefix}_CENTERS_MALFORMED")
                image_id = row[0]
                if image_id in values:
                    raise InvalidInput(f"{reason_prefix}_CENTERS_DUPLICATE_ID")
                try:
                    xyz = np.asarray([float(v) for v in row[1:]], dtype=np.float64)
                except ValueError as exc:
                    raise InvalidInput(f"{reason_prefix}_CENTERS_NONFINITE") from exc
                if not np.isfinite(xyz).all():
                    raise InvalidInput(f"{reason_prefix}_CENTERS_NONFINITE")
                values[image_id] = xyz
    except UnicodeDecodeError as exc:
        raise InvalidInput(f"{reason_prefix}_CENTERS_MALFORMED") from exc
    missing = sorted(set(expected_ids) - set(values))
    extra = sorted(set(values) - set(expected_ids))
    if missing:
        raise InvalidInput(f"{reason_prefix}_CENTERS_MISSING_ID", missing[0])
    if extra:
        raise InvalidInput(f"{reason_prefix}_CENTERS_EXTRA_ID", extra[0])
    return np.stack([values[image_id] for image_id in sorted(expected_ids)])


def compute_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit source-to-target proper Sim(3) from paired camera centers."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise InvalidInput("CAMERA_CENTERS_MALFORMED")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise InvalidInput("CAMERA_CENTERS_NONFINITE")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    rank = int(np.linalg.matrix_rank(source_centered))
    if rank < 2:
        raise InvalidInput("CAMERA_CENTERS_RANK_DEFICIENT")
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    raw_det = float(np.linalg.det(u @ vt))
    if rank == 3 and raw_det < 0:
        raise InvalidInput("REFLECTION_ALIGNMENT")
    sign = np.ones(3, dtype=np.float64)
    if raw_det < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    if np.linalg.det(rotation) <= 0:
        raise InvalidInput("REFLECTION_ALIGNMENT")
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.dot(singular, sign) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise InvalidInput("CAMERA_CENTERS_SCALE_INVALID")
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    aligned = source @ matrix[:3, :3].T + matrix[:3, 3]
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    if not np.isfinite(matrix).all() or not math.isfinite(rmse):
        raise InvalidInput("CAMERA_ALIGNMENT_NONFINITE")
    return matrix, rmse


def _load_submission_json(path: Path) -> tuple[dict[str, Any], str]:
    _regular_file(path, "SUBMISSION_JSON_NOT_REGULAR")
    raw = path.read_bytes()
    value = _strict_json_loads(raw, "SUBMISSION_JSON_MALFORMED")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_submission_contract(
    contract: dict[str, Any], protocol: dict[str, Any], track: str, scene: str
) -> None:
    unknown = sorted(set(contract) - ALLOWED_SUBMISSION_KEYS)
    if unknown:
        raise InvalidInput("UNKNOWN_SUBMISSION_FIELD", unknown[0])
    product = contract.get("product")
    if product in protocol["unsupported_products"]:
        raise InvalidInput("UNSUPPORTED_PRODUCT", str(product))
    expected = {
        "schema_version": protocol["submission_schema_version"],
        "protocol_version": protocol["protocol_version"],
        "track": track,
        "scene_id": scene,
        "product": "pointcloud",
        "geometry": "pointcloud.las",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise InvalidInput(f"{key.upper()}_MISMATCH")
    configuration_id = contract.get("configuration_id")
    if not isinstance(configuration_id, str) or not CONFIGURATION_ID.fullmatch(
        configuration_id
    ):
        raise InvalidInput("CONFIGURATION_ID_INVALID")
    if track == "rgb-local" and contract.get("camera_centers") != "camera_centers.csv":
        raise InvalidInput("CAMERA_CENTERS_REQUIRED")
    if track == "rgb-oriented" and "camera_centers" in contract:
        raise InvalidInput("CAMERA_CENTERS_NOT_ALLOWED")


def _snapshot_submission(
    submission: Path,
    snapshot: Path,
    protocol: dict[str, Any],
    track: str,
    scene: str,
) -> tuple[dict[str, Any], str]:
    if submission.is_symlink() or not submission.is_dir():
        raise InvalidInput("SUBMISSION_DIR_INVALID")
    submission = submission.resolve()
    contract_path = _snapshot_regular(
        submission / "submission.json",
        snapshot / "submission.json",
        int(protocol["limits"]["max_submission_contract_bytes"]),
        "SUBMISSION_JSON_NOT_REGULAR",
        "SUBMISSION_JSON_TOO_LARGE",
    )
    contract, contract_sha = _load_submission_json(contract_path)
    _validate_submission_contract(contract, protocol, track, scene)
    _snapshot_regular(
        submission / "pointcloud.las",
        snapshot / "pointcloud.las",
        int(protocol["limits"]["max_submission_bytes"]),
        "GEOMETRY_NOT_REGULAR",
        "GEOMETRY_TOO_LARGE",
    )
    if track == "rgb-local":
        _snapshot_regular(
            submission / "camera_centers.csv",
            snapshot / "camera_centers.csv",
            int(protocol["limits"]["max_camera_centers_bytes"]),
            "CAMERA_CENTERS_NOT_REGULAR",
            "CAMERA_CENTERS_TOO_LARGE",
        )
    return contract, contract_sha


def _validate_submission_snapshot(
    bundle: Path,
    track: str,
    scene: str,
    snapshot: Path,
    protocol: dict[str, Any],
    bundle_manifest: dict[str, Any],
    contract: dict[str, Any],
    contract_sha: str,
) -> dict[str, Any]:
    """Validate and bind only private snapshot bytes."""
    geometry = _regular_file(snapshot / "pointcloud.las", "GEOMETRY_NOT_REGULAR")
    point_count = _validate_las(geometry, protocol)
    input_manifest = bundle / "inputs" / track / scene / "input_manifest.json"
    reference_root = bundle / "evaluator-only" / scene
    reference_manifest = reference_root / "reference_manifest.json"
    reference_value = _strict_json_loads(
        reference_manifest.read_bytes(), "REFERENCE_MANIFEST_INVALID"
    )
    expected_ids = sorted(
        item["image_id"]
        for item in _strict_json_loads(
            input_manifest.read_bytes(), "INPUT_MANIFEST_INVALID"
        )["images"]
    )
    camera_centers_sha: str | None = None
    if track == "rgb-local":
        camera_path = _regular_file(
            snapshot / "camera_centers.csv", "CAMERA_CENTERS_NOT_REGULAR"
        )
        submitted_centers = _read_camera_centers(camera_path, expected_ids)
        camera_centers_sha = sha256_file(camera_path)
        publisher_centers = _read_camera_centers(
            reference_root / reference_value["files"]["publisher_camera_centers"]["path"],
            expected_ids,
            "REFERENCE_CAMERA",
        )
        matrix, camera_rmse = compute_similarity(submitted_centers, publisher_centers)
        alignment = {
            "kind": "camera-center Sim(3)",
            "matrix_4x4": matrix.tolist(),
            "camera_center_rmse_m": camera_rmse,
        }
    else:
        alignment = {"kind": "identity", "matrix_4x4": np.eye(4).tolist()}
    scene_record = bundle_manifest["scenes"][scene]
    input_hash = sha256_file(input_manifest)
    reference_hash = sha256_file(reference_manifest)
    if (
        scene_record["input_manifest_sha256"][track] != input_hash
        or scene_record["reference_manifest_sha256"] != reference_hash
    ):
        raise InvalidInput("BUNDLE_SCENE_BINDING_CHANGED", scene)
    return {
        "protocol": protocol,
        "contract": contract,
        "configuration_id": contract["configuration_id"],
        "contract_sha256": contract_sha,
        "camera_centers_sha256": camera_centers_sha,
        "geometry": geometry,
        "geometry_sha256": sha256_file(geometry),
        "point_count": point_count,
        "input_manifest_sha256": input_hash,
        "reference_manifest_sha256": reference_hash,
        "reference_manifest": reference_value,
        "reference_root": reference_root,
        "alignment": alignment,
    }


def validate_submission(
    bundle: Path, bundle_lock: Path, track: str, scene: str, submission: Path
) -> dict[str, Any]:
    """Snapshot and validate a submission against organizer-pinned bundle bytes."""
    protocol = load_protocol()
    if track not in protocol["tracks"]:
        raise InvalidInput("UNKNOWN_TRACK")
    if scene not in protocol["scenes"]:
        raise InvalidInput("UNKNOWN_SCENE")
    manifest = verify_bundle(bundle, bundle_lock)
    scratch_parent = bundle.resolve().parent
    with tempfile.TemporaryDirectory(prefix=".usegeo-validate-", dir=scratch_parent) as temporary:
        snapshot = Path(temporary) / "submission"
        contract, contract_sha = _snapshot_submission(
            submission, snapshot, protocol, track, scene
        )
        validated = _validate_submission_snapshot(
            bundle.resolve(),
            track,
            scene,
            snapshot,
            protocol,
            manifest,
            contract,
            contract_sha,
        )
        return {
            key: value
            for key, value in validated.items()
            if key not in {"geometry", "protocol", "reference_manifest", "reference_root"}
        }


def metric_arrays(
    submitted: np.ndarray,
    full_lidar: np.ndarray,
    refined_lidar: np.ndarray,
    trim_fraction: float = 0.9,
    completeness_threshold: float = 0.2,
) -> dict[str, float | int]:
    """Small-array oracle with the same exact formulas as production scoring."""
    if len(submitted) < 1 or len(full_lidar) < 1 or len(refined_lidar) < 1:
        raise InvalidInput("EMPTY_GEOMETRY")
    if not all(np.isfinite(points).all() for points in (submitted, full_lidar, refined_lidar)):
        raise InvalidInput("NONFINITE_GEOMETRY")
    accuracy_distances, _ = KDTree(full_lidar, leafsize=20).query(submitted, workers=1)
    k = int(math.ceil(trim_fraction * len(submitted)))
    best = np.partition(accuracy_distances, k - 1)[:k]
    completeness_distances, _ = KDTree(submitted, leafsize=20).query(
        refined_lidar, workers=1
    )
    return {
        "submitted_point_count": int(len(submitted)),
        "refined_reference_point_count": int(len(refined_lidar)),
        "accuracy_l1_m": float(np.mean(best)),
        "accuracy_rmse_m": float(np.sqrt(np.mean(best * best))),
        "completeness_0_20": float(
            np.count_nonzero(completeness_distances <= completeness_threshold)
            / len(refined_lidar)
        ),
    }


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(transformed).all():
        raise InvalidInput("NONFINITE_ALIGNED_GEOMETRY")
    return transformed


class _PeakRssMonitor:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _rss() -> int:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.peak = max(self.peak, self._rss())

    def __enter__(self) -> "_PeakRssMonitor":
        self.peak = self._rss()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())


def _production_metrics(validated: dict[str, Any], scratch: Path) -> dict[str, Any]:
    protocol = validated["protocol"]
    limits = protocol["limits"]
    chunk_points = int(limits["las_query_chunk_points"])
    workers = int(limits["query_workers"])
    matrix = np.asarray(validated["alignment"]["matrix_4x4"], dtype=np.float64)
    reference_files = validated["reference_manifest"]["files"]
    full_lidar_path = validated["reference_root"] / reference_files["full_lidar"]["path"]
    refined_lidar_path = (
        validated["reference_root"] / reference_files["refined_lidar"]["path"]
    )

    full_lidar = _load_las_xyz(full_lidar_path, chunk_points=chunk_points)
    full_tree = KDTree(full_lidar, leafsize=20, copy_data=False)
    distance_path = scratch / "accuracy-distances.float64"
    distances = np.memmap(
        distance_path, mode="w+", dtype=np.float64, shape=(validated["point_count"],)
    )
    offset = 0
    for points in _iter_las_xyz(validated["geometry"], chunk_points):
        points = _apply_transform(points, matrix)
        chunk_distances, _ = full_tree.query(points, workers=workers)
        distances[offset : offset + len(points)] = chunk_distances
        offset += len(points)
    if offset != validated["point_count"]:
        raise EvaluatorFailure("submission LAS point count changed during scoring")
    del full_tree, full_lidar
    gc.collect()

    k = int(math.ceil(float(protocol["trim_fraction"]) * len(distances)))
    distances.partition(k - 1)
    best = distances[:k]
    accuracy_l1 = float(np.mean(best))
    accuracy_rmse = float(np.sqrt(np.mean(best * best)))
    del best, distances
    distance_path.unlink()
    gc.collect()

    submitted = _load_las_xyz(validated["geometry"], chunk_points=chunk_points)
    submitted = _apply_transform(submitted, matrix)
    submission_tree = KDTree(submitted, leafsize=20, copy_data=False)
    complete = 0
    refined_count = 0
    for points in _iter_las_xyz(refined_lidar_path, chunk_points):
        refined_count += len(points)
        chunk_distances, _ = submission_tree.query(points, workers=workers)
        complete += int(
            np.count_nonzero(
                chunk_distances <= float(protocol["completeness_threshold_m"])
            )
        )
    del submission_tree, submitted
    gc.collect()
    if refined_count < 1:
        raise EvaluatorFailure("refined reference is empty")
    metrics = {
        "submitted_point_count": int(validated["point_count"]),
        "refined_reference_point_count": int(refined_count),
        "accuracy_l1_m": accuracy_l1,
        "accuracy_rmse_m": accuracy_rmse,
        "completeness_0_20": float(complete / refined_count),
    }
    if not all(
        math.isfinite(value)
        for key, value in metrics.items()
        if key not in {"submitted_point_count", "refined_reference_point_count"}
    ):
        raise InvalidInput("NONFINITE_METRIC")
    return metrics


def _resource_limits() -> dict[str, list[int | str]]:
    names = ["RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_NOFILE", "RLIMIT_STACK"]
    result: dict[str, list[int | str]] = {}
    for name in names:
        if not hasattr(resource, name):
            continue
        values = resource.getrlimit(getattr(resource, name))
        result[name] = ["infinity" if value == resource.RLIM_INFINITY else value for value in values]
    return result


def _stable_score_identity(score: dict[str, Any]) -> dict[str, Any]:
    ignored = {"started_at", "finished_at"}
    return {key: value for key, value in score.items() if key not in ignored}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_result_bundle(
    result_dir: Path,
    protocol: dict[str, Any],
    *,
    bundle_manifest: dict[str, Any] | None = None,
    bundle_manifest_sha256: str | None = None,
    bundle_lock_sha256: str | None = None,
    require_valid: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise InvalidInput("RESULT_BUNDLE_INVALID")
    result_dir = result_dir.resolve()
    members = {path.name for path in result_dir.iterdir()}
    if members != {"score.json", "run_manifest.json"}:
        raise InvalidInput("RESULT_BUNDLE_FILE_SET_MISMATCH")
    score_path = _regular_file(result_dir / "score.json", "RESULT_SCORE_INVALID")
    run_path = _regular_file(result_dir / "run_manifest.json", "RESULT_RUN_MANIFEST_INVALID")
    score = _strict_json_loads(score_path.read_bytes(), "RESULT_SCORE_INVALID")
    run = _strict_json_loads(run_path.read_bytes(), "RESULT_RUN_MANIFEST_INVALID")
    common_score_keys = {
        "schema_version",
        "protocol_version",
        "evaluator_name",
        "status",
        "invalid_reasons",
        "track",
        "scene_id",
        "product",
        "configuration_id",
        "alignment",
        "input_manifest_sha256",
        "reference_manifest_sha256",
        "submission_geometry_sha256",
        "submission_contract_sha256",
        "submission_camera_centers_sha256",
        "bundle_manifest_sha256",
        "bundle_lock_sha256",
        "upstream_commit",
        "corrections",
        "started_at",
        "finished_at",
    }
    valid_score_keys = common_score_keys | {"metrics"}
    invalid_score_keys = common_score_keys - {"alignment"}
    expected_score_keys = valid_score_keys if score.get("status") == "valid" else invalid_score_keys
    _expect_exact_keys(score, expected_score_keys, "RESULT_SCORE_SCHEMA_MISMATCH")
    if (
        score["schema_version"] != SCHEMA_VERSION
        or score["protocol_version"] != protocol["protocol_version"]
        or score["evaluator_name"] != protocol["evaluator_name"]
        or score["track"] not in protocol["tracks"]
        or score["scene_id"] not in protocol["scenes"]
        or score["upstream_commit"] != protocol["upstream"]["commit"]
        or score["corrections"] != protocol["corrections"]
        or score["status"] not in {"valid", "invalid"}
        or not isinstance(score["invalid_reasons"], list)
    ):
        raise InvalidInput("RESULT_SCORE_RELATION_MISMATCH")
    if require_valid and score["status"] != "valid":
        raise InvalidInput("AGGREGATE_REQUIRES_VALID_RESULTS")
    if score["status"] == "valid":
        if (
            score["product"] != "pointcloud"
            or score["invalid_reasons"] != []
            or not isinstance(score["configuration_id"], str)
            or not CONFIGURATION_ID.fullmatch(score["configuration_id"])
        ):
            raise InvalidInput("RESULT_SCORE_RELATION_MISMATCH")
        metrics = score["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != {
            "submitted_point_count",
            "refined_reference_point_count",
            "accuracy_l1_m",
            "accuracy_rmse_m",
            "completeness_0_20",
        }:
            raise InvalidInput("RESULT_METRICS_SCHEMA_MISMATCH")
        if (
            not isinstance(metrics["submitted_point_count"], int)
            or isinstance(metrics["submitted_point_count"], bool)
            or metrics["submitted_point_count"] < 1
            or not isinstance(metrics["refined_reference_point_count"], int)
            or isinstance(metrics["refined_reference_point_count"], bool)
            or metrics["refined_reference_point_count"] < 1
            or not _is_number(metrics["accuracy_l1_m"])
            or metrics["accuracy_l1_m"] < 0
            or not _is_number(metrics["accuracy_rmse_m"])
            or metrics["accuracy_rmse_m"] < 0
            or not _is_number(metrics["completeness_0_20"])
            or not 0 <= metrics["completeness_0_20"] <= 1
        ):
            raise InvalidInput("RESULT_METRICS_INVALID")
        alignment = score["alignment"]
        expected_alignment_keys = {"kind", "matrix_4x4"} | (
            {"camera_center_rmse_m"} if score["track"] == "rgb-local" else set()
        )
        if not isinstance(alignment, dict) or set(alignment) != expected_alignment_keys:
            raise InvalidInput("RESULT_ALIGNMENT_SCHEMA_MISMATCH")
        try:
            matrix = np.asarray(alignment["matrix_4x4"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidInput("RESULT_ALIGNMENT_INVALID") from exc
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise InvalidInput("RESULT_ALIGNMENT_INVALID")
        if score["track"] == "rgb-local":
            if alignment["kind"] != "camera-center Sim(3)" or not _is_number(
                alignment["camera_center_rmse_m"]
            ):
                raise InvalidInput("RESULT_ALIGNMENT_INVALID")
        elif alignment["kind"] != "identity" or not np.array_equal(matrix, np.eye(4)):
            raise InvalidInput("RESULT_ALIGNMENT_INVALID")
        for key in (
            "input_manifest_sha256",
            "reference_manifest_sha256",
            "submission_geometry_sha256",
            "submission_contract_sha256",
            "bundle_manifest_sha256",
            "bundle_lock_sha256",
        ):
            if not isinstance(score[key], str) or not HEX_SHA256.fullmatch(score[key]):
                raise InvalidInput("RESULT_HASH_INVALID", key)
        if score["submission_camera_centers_sha256"] is not None and (
            not isinstance(score["submission_camera_centers_sha256"], str)
            or not HEX_SHA256.fullmatch(score["submission_camera_centers_sha256"])
        ):
            raise InvalidInput("RESULT_HASH_INVALID", "submission_camera_centers_sha256")
    run_keys = {
        "schema_version",
        "protocol_version",
        "protocol_sha256",
        "upstream_files_sha256",
        "status",
        "started_at",
        "finished_at",
        "cli_arguments",
        "python",
        "packages",
        "platform",
        "host_resource_limits",
        "elapsed_seconds",
        "peak_rss_bytes",
        "configuration_id",
        "input_manifest_sha256",
        "reference_manifest_sha256",
        "submission_geometry_sha256",
        "submission_contract_sha256",
        "submission_camera_centers_sha256",
        "bundle_manifest_sha256",
        "bundle_lock_sha256",
        "method_metadata",
        "output_sha256",
    }
    _expect_exact_keys(run, run_keys, "RESULT_RUN_MANIFEST_SCHEMA_MISMATCH")
    if (
        run["schema_version"] != RUN_SCHEMA_VERSION
        or run["protocol_version"] != protocol["protocol_version"]
        or run["protocol_sha256"] != sha256_file(PROTOCOL_PATH)
        or run["upstream_files_sha256"] != protocol["upstream"]["files_sha256"]
        or run["status"] != score["status"]
        or run["started_at"] != score["started_at"]
        or run["finished_at"] != score["finished_at"]
        or not isinstance(run["cli_arguments"], list)
        or not all(isinstance(value, str) for value in run["cli_arguments"])
        or not isinstance(run["python"], str)
        or not isinstance(run["packages"], dict)
        or set(run["packages"]) != {"numpy", "scipy", "laspy", "Pillow"}
        or not all(isinstance(value, str) for value in run["packages"].values())
        or not isinstance(run["platform"], str)
        or not isinstance(run["host_resource_limits"], dict)
        or not _is_number(run["elapsed_seconds"])
        or run["elapsed_seconds"] < 0
        or not isinstance(run["peak_rss_bytes"], int)
        or isinstance(run["peak_rss_bytes"], bool)
        or run["peak_rss_bytes"] < 0
        or run["output_sha256"] != {"score.json": sha256_file(score_path)}
    ):
        raise InvalidInput("RESULT_RUN_MANIFEST_RELATION_MISMATCH")
    for key in (
        "configuration_id",
        "input_manifest_sha256",
        "reference_manifest_sha256",
        "submission_geometry_sha256",
        "submission_contract_sha256",
        "submission_camera_centers_sha256",
        "bundle_manifest_sha256",
        "bundle_lock_sha256",
    ):
        if run[key] != score[key]:
            raise InvalidInput("RESULT_RUN_MANIFEST_BINDING_MISMATCH", key)
    if bundle_manifest is not None:
        scene_record = bundle_manifest["scenes"][score["scene_id"]]
        if (
            score["input_manifest_sha256"]
            != scene_record["input_manifest_sha256"][score["track"]]
            or score["reference_manifest_sha256"]
            != scene_record["reference_manifest_sha256"]
            or score["bundle_manifest_sha256"] != bundle_manifest_sha256
            or score["bundle_lock_sha256"] != bundle_lock_sha256
        ):
            raise InvalidInput("RESULT_BUNDLE_AUTHORITY_MISMATCH")
    return score, run


def _publish_score(
    output: Path,
    score: dict[str, Any],
    run_manifest: dict[str, Any],
    protocol: dict[str, Any],
    bundle_manifest: dict[str, Any] | None,
    bundle_manifest_sha256: str | None,
    bundle_lock_sha256: str | None,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise EvaluatorFailure(f"result path is not a directory: {output}")
        existing, _ = _validate_result_bundle(
            output,
            protocol,
            bundle_manifest=bundle_manifest if score["status"] == "valid" else None,
            bundle_manifest_sha256=bundle_manifest_sha256,
            bundle_lock_sha256=bundle_lock_sha256,
        )
        if _stable_score_identity(existing) == _stable_score_identity(score):
            return existing
        raise EvaluatorFailure(f"existing result directory differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _write_json(temporary / "score.json", score)
        run_manifest["output_sha256"] = {
            "score.json": sha256_file(temporary / "score.json")
        }
        _write_json(temporary / "run_manifest.json", run_manifest)
        _validate_result_bundle(
            temporary,
            protocol,
            bundle_manifest=bundle_manifest if score["status"] == "valid" else None,
            bundle_manifest_sha256=bundle_manifest_sha256,
            bundle_lock_sha256=bundle_lock_sha256,
        )
        os.replace(temporary, output)
        return score
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def score_submission(
    bundle: Path,
    bundle_lock: Path,
    track: str,
    scene: str,
    submission: Path,
    output: Path,
    cli_arguments: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Score one scene, atomically publishing valid or invalid strict JSON."""
    started_clock = time.monotonic()
    started_at = _utc_now()
    protocol = load_protocol()
    validated: dict[str, Any] | None = None
    bundle_manifest: dict[str, Any] | None = None
    bundle_manifest_sha: str | None = None
    bundle_lock_sha: str | None = None
    partial_contract: dict[str, Any] | None = None
    partial_contract_sha: str | None = None
    partial_geometry_sha: str | None = None
    if not submission.is_symlink() and submission.is_dir():
        try:
            partial_contract, partial_contract_sha = _load_submission_json(
                submission / "submission.json"
            )
        except InvalidInput:
            pass
    try:
        bundle_manifest_sha = sha256_file(
            _regular_file(bundle / "manifest.json", "BUNDLE_INVALID")
        )
    except InvalidInput:
        pass
    try:
        bundle_lock_sha = sha256_file(
            _regular_file(bundle_lock, "BUNDLE_LOCK_NOT_REGULAR")
        )
    except InvalidInput:
        pass
    invalid: InvalidInput | None = None
    metrics: dict[str, Any] | None = None
    with _PeakRssMonitor() as monitor:
        try:
            if track not in protocol["tracks"]:
                raise InvalidInput("UNKNOWN_TRACK")
            if scene not in protocol["scenes"]:
                raise InvalidInput("UNKNOWN_SCENE")
            bundle_manifest = verify_bundle(bundle, bundle_lock)
            bundle_manifest_sha = sha256_file(bundle / "manifest.json")
            bundle_lock_sha = sha256_file(bundle_lock)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".usegeo-score-", dir=output.parent) as scratch:
                scratch_path = Path(scratch)
                snapshot = scratch_path / "submission"
                partial_contract, partial_contract_sha = _snapshot_submission(
                    submission, snapshot, protocol, track, scene
                )
                validated = _validate_submission_snapshot(
                    bundle.resolve(),
                    track,
                    scene,
                    snapshot,
                    protocol,
                    bundle_manifest,
                    partial_contract,
                    partial_contract_sha,
                )
                metrics = _production_metrics(validated, Path(scratch))
                if sha256_file(validated["geometry"]) != validated["geometry_sha256"]:
                    raise EvaluatorFailure("private submission snapshot changed during scoring")
        except InvalidInput as exc:
            invalid = exc
            geometry = submission / "pointcloud.las"
            try:
                if (
                    not geometry.is_symlink()
                    and geometry.is_file()
                    and geometry.stat().st_size
                    <= int(protocol["limits"]["max_submission_bytes"])
                ):
                    partial_geometry_sha = sha256_file(geometry)
            except OSError:
                pass
    elapsed = time.monotonic() - started_clock
    if monitor.peak > int(protocol["limits"]["max_peak_rss_bytes"]):
        raise EvaluatorFailure(
            f"peak RSS {monitor.peak} exceeds {protocol['limits']['max_peak_rss_bytes']}"
        )
    finished_at = _utc_now()
    base = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "evaluator_name": protocol["evaluator_name"],
        "track": track,
        "scene_id": scene,
        "product": (
            partial_contract.get("product", "pointcloud")
            if partial_contract is not None
            else "pointcloud"
        ),
        "configuration_id": (
            partial_contract.get("configuration_id")
            if partial_contract is not None
            and isinstance(partial_contract.get("configuration_id"), str)
            else None
        ),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "bundle_lock_sha256": bundle_lock_sha,
        "upstream_commit": protocol["upstream"]["commit"],
        "corrections": protocol["corrections"],
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if invalid is None and validated is not None and metrics is not None:
        score = {
            **base,
            "status": "valid",
            "invalid_reasons": [],
            "metrics": metrics,
            "alignment": validated["alignment"],
            "input_manifest_sha256": validated["input_manifest_sha256"],
            "reference_manifest_sha256": validated["reference_manifest_sha256"],
            "submission_geometry_sha256": validated["geometry_sha256"],
            "submission_contract_sha256": validated["contract_sha256"],
            "submission_camera_centers_sha256": validated["camera_centers_sha256"],
        }
        status_code = 0
    else:
        score = {
            **base,
            "status": "invalid",
            "invalid_reasons": [
                {"code": invalid.reason, "detail": invalid.detail}  # type: ignore[union-attr]
            ],
            "input_manifest_sha256": None,
            "reference_manifest_sha256": None,
            "submission_geometry_sha256": partial_geometry_sha,
            "submission_contract_sha256": partial_contract_sha,
            "submission_camera_centers_sha256": None,
        }
        status_code = 2
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "upstream_files_sha256": protocol["upstream"]["files_sha256"],
        "status": score["status"],
        "started_at": started_at,
        "finished_at": finished_at,
        "cli_arguments": list(cli_arguments or []),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "laspy": laspy.__version__,
            "Pillow": pillow_version,
        },
        "platform": platform.platform(),
        "host_resource_limits": _resource_limits(),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": monitor.peak,
        "configuration_id": score["configuration_id"],
        "input_manifest_sha256": score["input_manifest_sha256"],
        "reference_manifest_sha256": score["reference_manifest_sha256"],
        "submission_geometry_sha256": score["submission_geometry_sha256"],
        "submission_contract_sha256": score["submission_contract_sha256"],
        "submission_camera_centers_sha256": score["submission_camera_centers_sha256"],
        "bundle_manifest_sha256": score["bundle_manifest_sha256"],
        "bundle_lock_sha256": score["bundle_lock_sha256"],
        "method_metadata": (
            validated["contract"].get("method_metadata")
            if validated is not None
            else partial_contract.get("method_metadata")
            if partial_contract is not None
            else None
        ),
    }
    published_score = _publish_score(
        output,
        score,
        run_manifest,
        protocol,
        bundle_manifest,
        bundle_manifest_sha,
        bundle_lock_sha,
    )
    return status_code, published_score


def aggregate_scores(
    result_dirs: list[Path],
    bundle: Path,
    bundle_lock: Path,
    track: str,
    configuration_id: str,
) -> dict[str, Any]:
    """Verify and macro-average one declared configuration across all scenes."""
    protocol = load_protocol()
    if not CONFIGURATION_ID.fullmatch(configuration_id):
        raise InvalidInput("CONFIGURATION_ID_INVALID")
    manifest = verify_bundle(bundle, bundle_lock)
    manifest_sha = sha256_file(bundle / "manifest.json")
    lock_sha = sha256_file(bundle_lock)
    result_bundles = [
        _validate_result_bundle(
            path,
            protocol,
            bundle_manifest=manifest,
            bundle_manifest_sha256=manifest_sha,
            bundle_lock_sha256=lock_sha,
            require_valid=True,
        )
        for path in result_dirs
    ]
    scores = [score for score, _ in result_bundles]
    if len(scores) != 3 or {score["scene_id"] for score in scores} != set(protocol["scenes"]):
        raise InvalidInput("AGGREGATE_SCENE_SET_MISMATCH")
    if any(score["track"] != track for score in scores):
        raise InvalidInput("AGGREGATE_REQUIRES_VALID_SINGLE_TRACK")
    if any(score["configuration_id"] != configuration_id for score in scores):
        raise InvalidInput("AGGREGATE_CONFIGURATION_MISMATCH")
    names = ["accuracy_l1_m", "accuracy_rmse_m", "completeness_0_20"]
    by_scene = {score["scene_id"]: (score, run, path) for (score, run), path in zip(result_bundles, result_dirs)}
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "evaluator_name": protocol["evaluator_name"],
        "track": track,
        "configuration_id": configuration_id,
        "status": "valid",
        "scene_ids": sorted(protocol["scenes"]),
        "bundle_manifest_sha256": manifest_sha,
        "bundle_lock_sha256": lock_sha,
        "macro_mean": {
            name: float(np.mean([score["metrics"][name] for score in scores]))
            for name in names
        },
        "result_sha256": {
            scene: {
                "score.json": sha256_file(path / "score.json"),
                "run_manifest.json": sha256_file(path / "run_manifest.json"),
            }
            for scene, (_, _, path) in sorted(by_scene.items())
        },
        "submission_contract_sha256": {
            scene: score["submission_contract_sha256"]
            for scene, (score, _, _) in sorted(by_scene.items())
        },
        "submission_geometry_sha256": {
            scene: score["submission_geometry_sha256"]
            for scene, (score, _, _) in sorted(by_scene.items())
        },
        "input_manifest_sha256": {
            scene: score["input_manifest_sha256"]
            for scene, (score, _, _) in sorted(by_scene.items())
        },
        "reference_manifest_sha256": {
            scene: score["reference_manifest_sha256"]
            for scene, (score, _, _) in sorted(by_scene.items())
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and score the method-agnostic UseGeo point-cloud benchmark v1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="prepare all three scenes")
    prepare_parser.add_argument("--archives", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-bundle", help="verify a prepared bundle")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--bundle-lock", type=Path, required=True)
    for command in ("validate-submission", "score"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--bundle", type=Path, required=True)
        command_parser.add_argument("--bundle-lock", type=Path, required=True)
        command_parser.add_argument("--track", choices=["rgb-local", "rgb-oriented"], required=True)
        command_parser.add_argument("--scene", choices=["Dataset-1", "Dataset-2", "Dataset-3"], required=True)
        command_parser.add_argument("--submission", type=Path, required=True)
        if command == "score":
            command_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate", help="macro-average three valid scene scores")
    aggregate_parser.add_argument("--bundle", type=Path, required=True)
    aggregate_parser.add_argument("--bundle-lock", type=Path, required=True)
    aggregate_parser.add_argument("--track", choices=["rgb-local", "rgb-oriented"], required=True)
    aggregate_parser.add_argument("--configuration-id", required=True)
    aggregate_parser.add_argument("--results", nargs=3, type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = prepare(arguments.archives, arguments.output)
        elif arguments.command == "verify-bundle":
            manifest = verify_bundle(arguments.bundle, arguments.bundle_lock)
            result = {"status": "valid", "protocol_version": manifest["protocol_version"]}
        elif arguments.command == "validate-submission":
            validated = validate_submission(
                arguments.bundle,
                arguments.bundle_lock,
                arguments.track,
                arguments.scene,
                arguments.submission,
            )
            result = {
                "status": "valid",
                "track": arguments.track,
                "scene_id": arguments.scene,
                "point_count": validated["point_count"],
                "alignment": validated["alignment"],
                "submission_geometry_sha256": validated["geometry_sha256"],
                "submission_contract_sha256": validated["contract_sha256"],
            }
        elif arguments.command == "score":
            code, result = score_submission(
                arguments.bundle,
                arguments.bundle_lock,
                arguments.track,
                arguments.scene,
                arguments.submission,
                arguments.output,
                list(sys.argv if argv is None else argv),
            )
            print(json.dumps(result, allow_nan=False, sort_keys=True))
            return code
        else:
            result = aggregate_scores(
                arguments.results,
                arguments.bundle,
                arguments.bundle_lock,
                arguments.track,
                arguments.configuration_id,
            )
            _write_json(arguments.output, result)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except InvalidInput as exc:
        print(
            json.dumps(
                {"status": "invalid", "invalid_reasons": [{"code": exc.reason, "detail": exc.detail}]},
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 2
    except (EvaluatorFailure, OSError, zipfile.BadZipFile) as exc:
        print(
            json.dumps(
                {"status": "error", "error": type(exc).__name__, "detail": str(exc)},
                allow_nan=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
