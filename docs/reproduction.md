# Reproduction and source map

This release exposes the GeoPilot implementation, numerical adapters, experiment
drivers, evaluators and their tests. It is a source-and-evidence release, not a
one-command reproduction of every historical reconstruction. Dataset archives,
large meshes, frozen local environments, credentials and the RSI-Harness source
are not distributed here. Synthetic checks below do not establish reconstruction
quality or a learning benefit.

## 1. Fetch the pinned Agent runtime

Use Git, Node.js **22.19 or newer**, npm and Python 3.10 or newer. From this
repository's root:

```sh
python3 scripts/setup_upstream.py
python3 scripts/setup_upstream.py --verify-only
node code/geopilot_rsih/check.mjs
node --test --test-skip-pattern='GeoPilot entry uses actual upstream CLI' \
  code/geopilot_rsih/test_program.mjs \
  experiments/geopilot_rsih_learning/test_native_proposal.mjs
```

The setup script fetches RSI-Harness commit
`33c4f8dfac4359987f2e814e187de67c332498de`, verifies all 132 files in
`code/geopilot_rsih/upstream-lock.json`, installs its locked npm dependencies,
and builds it. Existing installations are never overwritten. This dependency
had no declared license in the pinned snapshot; our license does not apply to
it. Consult its upstream terms before reuse or redistribution.

`check.mjs` exercises a real upstream session with a **mock model**, without an
API call. The Node tests check program semantics, evidence bindings and native
proposal import using explicit synthetic fixtures. One historical integration
test is excluded by its exact name because its `controlled_fixture` launcher
also binds the real Dataset-1 manifest and OpenMVS executables. It is not a
data-free test; no fake dataset is installed to make it pass.

## 2. Check the evaluators without downloading a dataset

Use **Python 3.10.20** to satisfy the frozen evaluator runtime. Keep this
environment separate from the reconstruction environment in Section 3.

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=code python -m unittest discover -s code/usegeo_mesh_benchmark/tests
PYTHONPATH=code python -m pytest -q code/usegeo_benchmark/tests/test_benchmark.py
PYTHONPATH=code python -m unittest discover \
  -s experiments/geopilot_rsih_learning -p test_sampling_diagnostic.py
PYTHONPATH=code python -m unittest discover \
  -s experiments/geopilot_rsih_learning -p test_fixed_prefix_control.py
python -m unittest discover -s code/geopilot_rsih -p test_supervise.py
python -m unittest discover -s experiments/geopilot_v1 -p test_provider.py
```

These suites construct small geometry and result fixtures themselves. Run the
two evaluator suites in separate processes: both retain a historical test module
name `test_benchmark`. The full historical evaluator dependency snapshot is
retained at `code/usegeo_mesh_benchmark/requirements.txt`; the root requirements
contain the direct dependencies needed for the checks above.

Do not run every test by filename indiscriminately. `test_real_boundary.mjs`,
`test_native_launch.mjs`, `test_learning.py` and `test_continuation.py` also consume
the local frozen evidence tree and/or sandbox runtime. Their source is published
for inspection and later replay when those inputs are available. They were not
converted into toy experiments for release. `evaluator/probe_real.py` and
`code/usegeo_mesh_benchmark/tests/real_acceptance.py` likewise require real data.

## 3. Reconstruction and historical replay prerequisites

The reconstruction environment used Python 3.10.20, NumPy 1.26.4, SciPy 1.11.1
and pycolmap 3.12.6. For a separate numerical environment:

```sh
python3.10 -m venv .venv-reconstruction
.venv-reconstruction/bin/python -m pip install \
  numpy==1.26.4 scipy==1.11.1 pycolmap==3.12.6
.venv-reconstruction/bin/python code/usegeo_mesh_baseline/test_runner.py
```

Full reconstruction additionally needs:

- OpenMVS **2.4.0** executables `InterfaceCOLMAP`, `DensifyPointCloud`,
  `ReconstructMesh` and `RefineMesh` at the path recorded by the launchers.
- The original UseGeo archives and their matching SHA-256 identities. Acquire
  them through the [dataset publisher](https://github.com/UseGeoEvaluation/DepthEstimationAnd3DReconstruction)
  under its terms; use the preparation CLI in
  [`code/usegeo_benchmark/README.md`](../code/usegeo_benchmark/README.md).
- The prepared input bundle, bundle lock, frozen release manifest and runtime
  bindings used by the relevant experiment. They are not reconstructed from
  aggregate numbers in this repository.
- For strict historical launch: the macOS sandbox facility and expected local
  filesystem layout. The current launchers preserve those historical boundaries;
  this release does not claim an equivalent Linux sandbox implementation.
- For replay of prior proposals: the actual parent program/genome, launch/run
  records, original response and receipt, score and score manifest, and every
  cited evidence file. `knowledge.json` retains its historical source paths and
  hashes. Curated public knowledge cards are not replacements for those bytes.

See [`code/geopilot_rsih/README.md`](../code/geopilot_rsih/README.md) for the
historical entry and [`experiments/geopilot_rsih_learning/README.md`](../experiments/geopilot_rsih_learning/README.md)
for the evidence-bound proposal flow. These files describe their registered
versions; changing a path, tool capability or runtime requires a newly recorded
configuration, not relabeling an old result as reproduced.

The offline legacy API client reads only `MINIMAX_API_KEY`, `MINIMAX_BASE_URL`
and `MINIMAX_MODEL` from the process environment. Do not commit their values.
Its recorded experiment freezes `MiniMax-M3`; it is not the native Codex adapter.
Calling it sends the supplied context to that configured provider and incurs
provider costs. None of the checks in Sections 1–2 call it. The public snapshot
removes the historical `.zshrc` fallback, with an offline configuration test;
numerical programs and frozen score definitions are unchanged.

## 4. Implementation map

| Location | Purpose |
| --- | --- |
| `code/geopilot_rsih/` | Program registry/validation, upstream Genome patch interface, isolated launcher, numerical tools and export |
| `experiments/geopilot_rsih_learning/learning.py`, `continuation.py` | Offline feedback/context construction and historical proposal workflows |
| `experiments/geopilot_rsih_learning/native_proposal.mjs`, `native_launch.mjs` | Native Agent receipt import and controlled execution boundaries |
| `experiments/geopilot_rsih_learning/*diagnostic*.py`, `publisher_raster_*.py` | Camera, raster and mesh diagnostic/control drivers; most use historical input paths |
| `experiments/geopilot_rsi/run.py` | Reused numerical adapter, including native mesh I/O and camera/world transformation |
| `experiments/geopilot_v1/provider.py` | Legacy stateless provider client, environment configuration only |
| `code/usegeo_mesh_baseline/` | Camera preparation and fixed sparse/mesh baseline |
| `code/usegeo_mesh_benchmark/` | Original float64 mesh evaluator, compatibility checks and acceptance manifest validation |
| `experiments/geopilot_rsih_learning/evaluator/` | Later separately identified adaptive-batch evaluator; do not mix its scores with older runtime aggregates |
| `code/usegeo_benchmark/` | Dataset preparation, point-cloud protocol and licensed upstream reference files |

The outer research Agent is a host-driven workflow, supported by these scripts
and curated knowledge. This release does not contain a newly invented autonomous
orchestration framework, an automatic all-round memory distiller, or a claim that
every proposed change improved quality.

## 5. Release verification scope

The 2026-09-26 release was checked in a fresh source copy. RSI-Harness was fetched
and built there rather than copied from the research checkout. Python checks
used the pre-existing frozen evaluator/numerical environments; this is a clean
source test, not a claim of testing every operating system or a fresh Python
dependency installation. No new reconstruction or model request was run for
packaging. The excluded large data and historical runtime bindings remain
prerequisites for full result replay.

## 6. Current outer-host checks

The default README quick start now includes a standard-library-only dispatcher/receipt/registry suite. Run only the named modules/class there: numerical integration tests require additional frozen inputs and are not represented as public data-free checks. The new model client takes an explicit model identity (MiniMax-M3.1-Flash-Preview for prospective comparisons); the legacy M3 client above remains historical. See [usage and IO](usage.md). No new end-to-end reconstruction is claimed by this update.
