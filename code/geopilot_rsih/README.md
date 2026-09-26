# GeoPilot reconstruction benchmark entry

`upstream/` is the locked RSI-Harness source at `33c4f8dfac4359987f2e814e187de67c332498de`. The registered P0 executes `prepare → densify → mesh → stop`. `p0.json` freezes OpenMVS resolution level 1 and mesh decimation 0.25; `tools.py` freezes COLMAP undistortion at 2400 px. Do not change these after observing a score under this configuration ID.

From the repository root, check the runtime and sandbox without starting reconstruction:

```bash
node code/geopilot_rsih/check.mjs
node --test code/geopilot_rsih/test_program.mjs code/geopilot_rsih/test_real_boundary.mjs
npm --prefix code/geopilot_rsih/upstream run check
node code/geopilot_rsih/launch.mjs --program code/geopilot_rsih/p0.json \
  --output out/geopilot-rsih-preflight-<new-id> \
  --python out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime/bin/python \
  --inputs out/usegeo_benchmark/prepared/v1/inputs/rgb-oriented/Dataset-1 \
  --scope development --preflight-only
```

A successful preflight records `isolation.json` and `run.json` with `status=preflight_passed` and `benchmark_eligible=false`. It checks the pinned Dataset-1 bundle lock and manifest, exact 224 input files, the registered P0 byte hash, installed Node/Python dependency bytes, and the frozen release evaluator/runtime bytes. It also probes denied reference, offline evaluator and personal config reads, denied network access in Node and Python, and actual pycolmap/OpenMVS loader help inside the same macOS sandbox. The worker inherits only the fresh run's configuration and agent directory; release evaluator files stay outside the online read profile.

Run the full 224-image reconstruction in a new output directory by omitting `--preflight-only` from the command above. The external observer enforces four hours and 48 GiB total process-group RSS. `supervision.json`, runtime logs, `events.jsonl`, node artifacts and `run.json` preserve actual results and costs. A successful real mesh is copied after the sandboxed process exits into a new `submission/`, checked against source and artifact hashes, and passed to the frozen `validate-submission` CLI. Only then is `benchmark_eligible=true`. The dense coverage manifest records each input image's SfM, matching, registration, undistortion and depth-map status plus omissions; the submission reports achieved counts. This verifies use of all permitted inputs and labels observed coverage. The frozen scorer determines geometric completeness. A fixture run uses `--scope controlled_fixture` and remains ineligible.

Independent acceptance scores the exact submission using the frozen release source and clean evaluator runtime:

```bash
PYTHONPATH=out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code \
  out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python \
  -m usegeo_mesh_benchmark.benchmark score \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --scene Dataset-1 --submission out/geopilot-rsih-p0-dataset1-<new-id>/submission \
  --output out/geopilot-rsih-p0-score-<new-id>
```

The score is external to this online P0 run. A fixture or failed run cannot stand in for a Dataset-1 result. The frozen protocol, evaluator, references, source bundle and old experiments are not modified by this entry.

An offline model candidate may use the same launcher with `--program <proposal>/p1.json --candidate <proposal>/candidate.json`. The parent checks the complete evidence hashes,
the actual response program and the upstream applied patch before entry. Only program
content is copied into the online Genome; offline requests, scores and proposals are
denied by the worker sandbox. This first candidate route is bound to the r3 P0 parent.
The exporter reuses the frozen validator and records a distinct candidate configuration.
`decision` events bind the observed diagnostic, rule, successor and program identity.
Online model calls remain disabled; `online_tokens=0` is a compatibility constant,
with `model_usage.measured=false`, not provider telemetry. Real offline usage belongs
to the saved provider response.

The new offline bridge and separately versioned equivalent scorer live in
`experiments/geopilot_rsih_learning/`. Their development results do not overwrite
release-v1 results or constitute an accepted evaluator release or program release.
