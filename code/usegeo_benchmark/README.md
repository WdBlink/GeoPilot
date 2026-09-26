# UseGeo Point-Cloud Benchmark v1

This directory contains the formal, method-agnostic evaluator for dense point
clouds reconstructed from the three UseGeo UAV image collections. It measures
the geometric quality of the submitted data product. It does **not** score
Agent behavior, tool use, recovery, cost, process quality, or method family.
It is also not a reproduction of the full published leaderboard.

Version 1 intentionally supports one product: an uncompressed LAS point cloud.
`depth` and `mesh` submissions return `UNSUPPORTED_PRODUCT`. They are not
silently converted and are not described as fixed. A formal depth track needs
authoritative image-plane calibration conventions that are not available in
the inspected release. The reproduced upstream mesh all-tie quantile defect is
also outside this point-cloud release.

## Runtime

Use CPython 3.11 and the exact packages in `requirements.txt`:

```bash
python3.11 -m venv out/usegeo_benchmark/.venv
out/usegeo_benchmark/.venv/bin/pip install -r code/usegeo_benchmark/requirements.txt
```

Open3D is not part of the production evaluator. The vendored scripts document
the source algorithm and remain byte-identical to upstream; `benchmark.py`
implements the bounded-memory derived evaluator with NumPy, SciPy, and laspy.

## Frozen tracks

Both tracks contain the same 828 matched original JPEG bytes at 7952 x 5304.
Dataset-1 has 224 images, Dataset-2 has 327, and Dataset-3 has 277. The unmatched
Dataset-2 image `2021-04-23_12-02-40_S2223314_DxO.jpg` is excluded from both.
Every selected image is checked for an absent EXIF `GPSInfo` tag and is copied
without re-encoding.

- `rgb-local` exposes only RGB. A submission supplies a point cloud and one
  camera center for every frozen image ID. The evaluator fits one global,
  proper 3D Umeyama Sim(3) from those centers to withheld publisher centers.
  This removes only the global coordinate-frame ambiguity. No LiDAR/MVS
  residual, RANSAC, ICP, best-camera subset, or per-axis scale affects it.
- `rgb-oriented` exposes the same RGB plus the byte-identical publisher
  orientation table. A submission must already be in the dataset-native frame;
  scoring uses the identity transform. Absolute frame error is retained.

The `method_metadata` value may describe an Agent, a non-Agent system, or
anything else. It is provenance only. Validation, alignment, nearest-neighbor
queries, metrics, and aggregation never read or branch on it.

## Coordinates and unresolved calibration

Publisher MVS LAS files declare EPSG:25832, and the orientation X/Y coordinates
occupy the same numeric frame. XYZ distances and thresholds are in metres. The
LiDAR LAS contains no parsed CRS and the inspected sources do not establish the
vertical datum, so Z is called only `dataset-native Z in metres`; it is not
claimed to be orthometric or ellipsoidal height.

The orientation table exposes `omega/phi/kappa`, `c`, `x0`, `y0`, and distortion
terms, but the release does not normalize the rotation order/sign conventions
or the principal-point/image-axis convention into a formal camera model. V1
copies this table only for `rgb-oriented`; it does not synthesize OpenCV
intrinsics or rotations. These limitations do not change point-cloud XYZ
scoring, but they block a formal depth evaluator without new authoritative
calibration evidence.

## Data preparation

The three source archives are read-only. Their local provenance SHA-256 values
are frozen in `protocol_v1.json`; these identify the locally inspected bytes
and are not a publisher authenticity claim. Preparation verifies all three
hashes before creating any published bundle, checks at least 220 GiB free, then
builds all scenes sequentially in a sibling temporary directory:

```bash
out/usegeo_benchmark/.venv/bin/python code/usegeo_benchmark/benchmark.py prepare \
  --archives /path/to/UseGeo \
  --output out/usegeo_benchmark/prepared

out/usegeo_benchmark/.venv/bin/python code/usegeo_benchmark/benchmark.py verify-bundle \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json
```

`prepare` accepts only contained regular ZIP members, rejects traversal,
duplicates, symlinks and selected-member CRC failures, and extracts only the
allowlisted image/orientation/reference files. It never invokes Metashape.
Publication is one atomic rename. A byte-consistent existing bundle is verified
and reused; a different `v1` directory is never deleted or overwritten.

The resulting method-input roots are:

```text
prepared/v1/inputs/rgb-local/Dataset-N/
prepared/v1/inputs/rgb-oriented/Dataset-N/
```

The evaluator-only roots are deliberately separate:

```text
prepared/v1/evaluator-only/Dataset-N/
  full_lidar.las
  publisher_mvs.las
  publisher_camera_centers.csv
  refined_lidar.las
  reference_manifest.json
```

Never give an evaluator-only directory to a method. `Depth_resized`, LiDAR,
publisher MVS, observation files, trajectories, ODM files, and
reference-derived masks/transforms are forbidden from method-input roots.
`prepare` also publishes `v1.lock.json` beside the bundle. This external
organizer lock pins the bundle-manifest digest, source-archive hashes, vendored
source hashes, and all scene-manifest relations. Keep it outside the bundle and
distribute it through the organizer-controlled evaluation procedure;
participant-supplied or bundle-internal locks are not an authority.
`verify-bundle` requires this lock, checks every manifest schema/count/relation,
re-hashes every declared file, and rejects missing, extra, changed, symlinked,
or leaked content.

## Fixed references and metrics

Accuracy queries every aligned submitted point against the full publisher
LiDAR in 3D. With `n` submitted points, exactly
`ceil(0.90 * n)` smallest Euclidean distances are retained. Their mean and
root-mean-square are `accuracy_l1_m` and `accuracy_rmse_m`. Exact-count
partition is the deliberate v1 correction to upstream's iterative threshold
search: it is deterministic for equal distances and returns finite zero for a
perfect all-tie cloud.

Completeness queries every fixed refined-LiDAR point against the aligned
submission. `completeness_0_20` is the fraction at distance less than or equal
to 0.20 m.

The refined reference is prepared once, before any submission. It retains a
full-LiDAR point exactly when its XY distance to the nearest publisher-MVS
point is at most 5 m, preserving the upstream predicate and direction. This is
named **publisher-MVS-derived 5 m XY support**. It is a fixed organizer choice,
not a physically exact visibility volume and not completeness over all LiDAR.
Publisher MVS is neither a method input nor the geometric scoring target.

There is no spatial crop, point sampling, voxel downsampling, hidden point-count
normalization, per-submission support, ICP, or fit to LiDAR. Per-scene output
reports point counts and all three metrics; no composite scalar is emitted.
Macro means are permitted only for the three valid scenes of one track.

## Submission contract

Each submission directory contains `submission.json` and `pointcloud.las`.
`rgb-local` also contains `camera_centers.csv`. LAS must be uncompressed,
parse completely, contain 1 through 300,000,000 finite decoded XYZ points, and
be no larger than 20 GiB. Other LAS dimensions are ignored.

An oriented submission example:

```json
{
  "schema_version": "usegeo-submission-1.1",
  "protocol_version": "usegeo-pointcloud-v1",
  "track": "rgb-oriented",
  "scene_id": "Dataset-1",
  "product": "pointcloud",
  "geometry": "pointcloud.las",
  "configuration_id": "example-fixed-v1",
  "method_metadata": {"name": "example", "family": "non-Agent"}
}
```

For `rgb-local`, add exactly:

```json
"camera_centers": "camera_centers.csv"
```

The CSV header is exactly `image_id,x,y,z`. It contains every frozen image ID
once, no missing or extra IDs, and finite values in the same frame as the
submitted point cloud. Lexical image-ID order defines the fit. Centered source
rank must be at least two. Rank-two camera layouts receive a proper-rotation
solution for the mathematically unconstrained normal axis; a rank-three
reflection relation is rejected.

`configuration_id` is a required non-scoring identifier (1-128 ASCII letters,
digits, `.`, `_`, or `-`). A formal macro comparison must predeclare one fixed
configuration ID and use it unchanged for all three scenes. The evaluator binds
the identifier and all three contract hashes; organizer procedure remains
responsible for preventing a dishonest participant from relabeling a changed
configuration. Unknown top-level fields are rejected so methods cannot
influence the scorer through undeclared knobs. Geometry and camera filenames
are literal, contained regular files; symlinks, traversal, malformed
JSON/CSV/LAS, empty geometry, non-finite values, and size/count excess are
invalid.

## Validation and scoring

```bash
out/usegeo_benchmark/.venv/bin/python code/usegeo_benchmark/benchmark.py validate-submission \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --track rgb-oriented --scene Dataset-1 --submission SUBMISSION_DIR

out/usegeo_benchmark/.venv/bin/python code/usegeo_benchmark/benchmark.py score \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --track rgb-oriented --scene Dataset-1 --submission SUBMISSION_DIR \
  --output RESULT_DIR
```

Exit 0 means a valid completed score, exit 2 means an invalid bundle/submission,
and exit 1 means evaluator/runtime failure. Invalid scores contain stable reason
codes and no `metrics` object—never zero, NaN, Infinity, or another numeric
sentinel that could masquerade as a valid result. JSON uses sorted keys and
`allow_nan=false`.

Before validation, scoring copies the contract, geometry, and local-track camera
centers to a private bounded scratch snapshot. Hashing, alignment, and both
metric phases use only those snapshot bytes. Scoring writes `score.json` and
`run_manifest.json` through one atomic directory rename. A semantically
identical existing result is reused only after both files pass their exact
schemas, bindings, and output-hash check; a different or partial result
directory is refused. The score binds protocol, organizer bundle lock, input
manifest, reference manifest, submission contract, camera-center file, and
geometry hashes. The run manifest records exact CLI arguments, package/Python
versions, host limits, elapsed time, peak RSS, and output hashes. Timestamps and
absolute paths may vary; status, alignment, counts, metrics, reason codes, and
bound content hashes are expected to reproduce.

After obtaining all three scene results for one track:

```bash
out/usegeo_benchmark/.venv/bin/python code/usegeo_benchmark/benchmark.py aggregate \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --track rgb-oriented \
  --configuration-id example-fixed-v1 \
  --results RESULT_1 RESULT_2 RESULT_3 \
  --output rgb-oriented-macro.json
```

Aggregation consumes the three complete result directories and validates their
exact score/run-manifest schemas, output hashes, finite metrics, current
protocol/evaluator identity, organizer-bundle bindings, scene set, track, and
configuration ID. It rejects missing/invalid scenes, partial or forged results,
mixed configurations, mixed tracks, and cross-track averages.

## Resource behavior

Scenes are prepared and scored sequentially. LAS queries use 1,000,000-point
chunks and at most eight SciPy workers. Accuracy builds the full-LiDAR tree and
stores submission distances in an on-disk float64 memmap. It releases that tree
before completeness builds the submitted-point tree. No two-scene trees are
held together. Peak RSS is sampled every 0.5 seconds and must stay below 60 GiB;
exceeding the cap is evaluator failure, not a method score.

## Real-data acceptance

The formal acceptance run must prepare and verify all three complete archives,
then submit each scene's `publisher_mvs.las` as a separately copied,
`rgb-oriented`, clearly labeled **publisher fixture**. All three results must be
valid and finite and their hashes retained. This checks the evaluator pipeline;
it is not a new reconstruction, a ground-truth self-comparison, or reproduction
of the paper leaderboard. A 10 m crop or other smoke fixture is never a formal
score.

Run deterministic controls before the real path:

```bash
out/usegeo_benchmark/.venv/bin/python -m pytest -q \
  code/usegeo_benchmark/tests/test_benchmark.py
```

## Provenance and licensing

The evaluator derives from
`UseGeoEvaluation/DepthEstimationAnd3DReconstruction` commit
`aa6897530e1827d2527d09f32147edb1e44c8542`. Byte-identical relevant upstream
files, their hashes, repository URL, retrieval date, and MIT license are under
`vendor/usegeo-aa689753/`. The retained directions, thresholds, and deliberate
v1 deviations are also frozen in `protocol_v1.json` and bound into every score.

Vendored evaluator code remains MIT licensed. Prepared UseGeo data remains
under the dataset's CC BY-NC-SA 4.0 terms. Preparation does not publish or
relicense it; operators must keep the bundle local and follow the dataset terms.
