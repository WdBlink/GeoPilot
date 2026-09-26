# UseGeo RGB-oriented mesh benchmark v1

This directory is an isolated triangle-mesh evaluator for the three frozen
UseGeo scenes. It reuses the accepted point-cloud v1 organizer bundle and lock,
but does not change them. The only participant input track is `rgb-oriented`,
the only product is `mesh`, and the only geometry file is `mesh.ply`.

There are two deliberately independent products:

- `benchmark score` is the formal local protocol. It reports area-uniform
  surface-to-full-LiDAR accuracy/precision and fixed-refined-LiDAR-to-triangle
  completeness, plus topology diagnostics.
- `official_compat official-compat` is a bounded repair of the published
  triangle-correspondence evaluator. It has its own process, result schema,
  output directory, timeout and failure status. Formal scoring neither imports
  nor invokes this module, and does not depend on its completion.

The compatibility repair uses a separate boolean `computed` array, retains
exactly `ceil(0.9*N)` index-stably ordered triangle distances, and fails on
non-finite values, invalid primitive IDs, no progress, 10,000 passes, or 7,200
seconds. It is compatibility evidence, not an upstream-endorsed score or an
original-paper reproduction.

## Runtime and commands

From the repository root:

```bash
uv venv --python 3.10 out/usegeo_mesh_benchmark/.venv
uv pip install --python out/usegeo_mesh_benchmark/.venv/bin/python \
  -r code/usegeo_mesh_benchmark/requirements.txt

PYTHONPATH=code out/usegeo_mesh_benchmark/.venv/bin/python \
  -m usegeo_mesh_benchmark.benchmark validate-submission \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --scene Dataset-1 --submission /path/to/submission

PYTHONPATH=code out/usegeo_mesh_benchmark/.venv/bin/python \
  -m usegeo_mesh_benchmark.benchmark score \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --scene Dataset-1 --submission /path/to/submission \
  --output /path/to/result

PYTHONPATH=code out/usegeo_mesh_benchmark/.venv/bin/python \
  -m usegeo_mesh_benchmark.official_compat official-compat \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --scene Dataset-1 --submission /path/to/submission \
  --output /path/to/official-result

PYTHONPATH=code out/usegeo_mesh_benchmark/.venv/bin/python \
  -m usegeo_mesh_benchmark.benchmark aggregate \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --configuration-id example-fixed-v1 \
  --results /result/Dataset-1 /result/Dataset-2 /result/Dataset-3 \
  --output /result/aggregate.json
```

`score` runs in a child process. Its parent samples peak RSS every 0.5 seconds
and fails closed after three unavailable samples or above 60 GiB or 14,400
seconds. The parent writes a cancellation flag and the worker checks it and its
deadline between sampling, LAS, KD-tree and triangle-query chunks before a
bounded kill fallback. Scenes and real acceptance repetitions are serial.

## Strict submission

`submission.json` has exactly these fields:

```json
{
  "schema_version": "usegeo-mesh-submission-1.0",
  "protocol_version": "usegeo-mesh-rgb-oriented-v1",
  "track": "rgb-oriented",
  "scene_id": "Dataset-1",
  "product": "mesh",
  "geometry": "mesh.ply",
  "configuration_id": "example-fixed-v1",
  "method_metadata": {"name": "example", "family": "non-Agent"}
}
```

`method_metadata` is provenance only. No validation, metric, transform,
sampling or aggregation branch may depend on it.

The PLY header is fixed, ASCII with LF endings, followed by little-endian
binary payload: float64 XYZ vertices, then records containing `uchar(3)` and
three signed int32 indices. Comments, extra/reordered fields, polygons,
textures, normals, colors, non-finite coordinates, invalid/repeated indices,
payload mismatch, trailing bytes, and triangles with area at most `1e-12 m²`
are invalid. Minimum geometry is three vertices and one face. Limits are 20
GiB, 20 million vertices and 40 million triangles. Inputs must be contained
regular files, not symlinks.

Validation, scoring, aggregation and fixture generation enforce the exact
frozen Python, NumPy, SciPy, laspy and Open3D versions. Every command verifies
the organizer bundle and external lock before opening a submission. The
contract and mesh are copied with no-follow semantics into
private scratch and hashed while copied; only snapshot bytes are evaluated.
Final directories are sibling-temporary builds published by atomic rename.
Existing differing or partial outputs are refused; verified identical results
may be reused. Invalid, evaluator-error and non-completed attempts also publish
immutable status and run-manifest evidence without metric fields.

## Formal metrics

The frozen sampler uses NumPy `PCG64`, seed `20260916`, triangle-order-preserving
area probabilities, square-root barycentric sampling and exactly 1,000,000
samples per scene. A SciPy KDTree over the complete `full_lidar.las` is queried
with one worker.

- `accuracy_l1_m` and `accuracy_rmse_m`: all surface samples.
- `accuracy_best90_l1_m` and `accuracy_best90_rmse_m`: exactly 900,000 samples,
  stably ordered by `(distance, sample_index)`.
- `precision_0_20`: untrimmed sample fraction at distance `<= 0.20 m`.
- `completeness_0_20`: every fixed refined-LiDAR point queried against the
  triangle surface, never against vertices, at distance `<= 0.20 m`.

Completeness uses Float64 exact point-to-triangle refinement. A `cKDTree` over
triangle AABB-sphere centres supplies only conservative candidates: the exact
distance to the nearest-centre triangle is an upper bound, and every sphere
lower bound capable of improving it is retained. Power-of-two radius bins keep
an isolated long triangle from expanding every candidate search, while adaptive
subchunks cap ordinary materialization at 2,000,000 query/triangle pairs. In the
protocol maximum, one pathological query may still inspect all 40 million
triangles; the scene timeout and 60 GiB watchdog remain the hard ceiling. The
refined-LiDAR AABB origin is still recorded as a reference-only frame marker.
It is not fitted alignment: the mesh is never moved relative to the reference,
cropped, repaired, filled, smoothed, scaled or rotated.

Diagnostics report surface area, degenerate triangles, unused vertices,
boundary count/length, nonmanifold edges and shared-edge triangle components.
Open boundaries and multiple components remain valid. There is no composite
score or ranking.
Repeated or overlapping faces are not deduplicated: each contributes its own
area to sampling and to topology diagnostics. Geometry cleanup belongs to the
submitting method, never to this evaluator.

## Full-scene acceptance fixtures

`generate-real-fixtures` makes a bounded, deterministic acceptance fixture from
every publisher-MVS point. Pass one obtains point count and XY bounds. Pass two
computes arithmetic mean XYZ per occupied 1 m cell. Every occupied cell becomes
a vertex; every fully occupied adjacent 2x2 block gets two fixed-winding
triangles on the lower-left-to-upper-right diagonal. There is no LiDAR access,
support crop, height filter, outlier rejection, subsampling or tuning.

The mandatory label is **publisher-mvs-derived approximate 2.5D acceptance
fixture**. It is not an author mesh, OpenMVS reconstruction claim, participant
method, paper result, or ground truth. `fixture_manifest.json` binds source and
output hashes, counts, bounds, algorithm, runtime and no-filter policy. UseGeo
prepared data and generated fixtures remain under CC BY-NC-SA 4.0 and must not
be published from this repository.

The acceptance runner holds a sibling OS file lock across preparation, execution
and terminal publication, and journals completed formal scene results. Rejected
callers never write into an existing output. A failed owned attempt writes
`protected-after.json`, `acceptance-error.json`, and terminal execution state
when storage remains writable. A retry is allowed only for that verified failed shape, archives
its terminal evidence under `attempt-history/`, strictly revalidates fixtures
and completed score directories, and resumes missing results without deletion.
Locks are released by the OS on process death; their sibling files are retained
to avoid inode replacement races. A live or unverifiable recorded PID is never
treated as proof of interruption.
The final `acceptance-summary.json` is published only after terminal protection
checks and execution state succeed; `acceptance-candidate.json` is not a success
marker. Interrupted pre-publication attempts can resume without overwriting
history. Reused scene results must also match the current submission and bundle
bindings, not only their own adjacent hashes.

## Paper-result admission

The separate `release.py` entry validates a portable evidence manifest:

```bash
PYTHONPATH=code out/usegeo_mesh_benchmark/.venv/bin/python \
  -m usegeo_mesh_benchmark.release /path/to/release_manifest.json
```

Schema `usegeo-mesh-release-1.0` binds source snapshots, protocol, criteria,
data lock, exact runtime, commands/logs, fixed acceptance checks, campaign
outputs and an independently identified review. References use relative
ordinary-file paths below the manifest directory plus SHA256; missing,
changed, escaping or symlink evidence is rejected. The validator checks
identity and completeness, not the scientific truth of a report.

The two scopes are not interchangeable: `evaluator_release` requires A01–A14,
six independently recomputed full-fixture scores and three actual official
attempts. `paper_results` additionally requires A15–A18, a bound READY
evaluator release and provenance-complete, same-configuration real method
results for all three scenes. The `readiness` field must equal the value
derived from these gates and the independent review; it is not an override.
Exit zero means READY; incomplete or invalid evidence exits nonzero. An
official full-scene failure remains separately reported and is not itself a
formal surface-score failure.

Each real method campaign binds an `experiment` reference using schema
`usegeo-mesh-experiment-1.0`: method/version/configuration/seed, input and
pre-registration hashes, scene/run/status, evaluator-release identity,
mesh/score/run evidence, and separate method versus evaluator costs.
Deterministic evaluator repeats are not independent method trials. Acceptance
fixtures never become method results by changing their label.

The upstream evaluator is
`UseGeoEvaluation/DepthEstimationAnd3DReconstruction` commit
`aa6897530e1827d2527d09f32147edb1e44c8542` (MIT). The repaired comparator
addresses termination, status and exact-count defects. The formal surface
metrics and the 2.5D fixtures are local protocol additions. Nothing here claims
official endorsement, leaderboard comparability, new reconstruction quality,
or reproduction of the original paper's mesh scores.
