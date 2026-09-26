# Evidence-bound GeoPilot learning and bounded continuation

This version keeps the sealed r3 fixed P0 and its failed original score. The new
[evaluator](evaluator/README.md) changes computation scheduling, not geometry
semantics. Its results are versioned development evidence, not release-v1 scores.
No automatic publication is performed: the nine-run real-geometry calibration
and full development set required by the system spec remain separate gates.

Run from the repository root. Use a fresh output path for every command.

```sh
PYTHONDONTWRITEBYTECODE=1 out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python -B \
  experiments/geopilot_rsih_learning/evaluator/benchmark.py score \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json --scene Dataset-1 \
  --submission out/geopilot-rsih-p0-mu9geojt-r3/submission \
  --output out/NEW-r3-score

python3 -B experiments/geopilot_rsih_learning/learning.py \
  --run out/geopilot-rsih-p0-mu9geojt-r3 \
  --score out/NEW-r3-score/score.json --manifest out/NEW-r3-score/run_manifest.json \
  --output out/NEW-proposal

node code/geopilot_rsih/launch.mjs --scope development \
  --inputs out/usegeo_benchmark/prepared/v1/inputs/rgb-oriented/Dataset-1 \
  --python out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime/bin/python \
  --program out/NEW-proposal/p1.json --candidate out/NEW-proposal/candidate.json \
  --output out/NEW-p1-preflight --preflight-only
```

Only after a successful preflight, remove `--preflight-only` and use another fresh
output to reconstruct P1. Score its `submission` with the same evaluator command
and version. Compare by passing `--p1-run`, `--p1-score`, `--p1-manifest` alongside
the P0 arguments to `learning.py`, with a fresh output JSON file.

The offline entry reuses `experiments/geopilot_v1/provider.py` (MiniMax-M3) and the
actual upstream `applyHarnessPatch` via `bridge.mjs`. It saves the real request,
response and measured usage, domain source identities, structured patch, exact
program and parent. There is one candidate, with no handwritten repair or best-of
selection. Invalid proposals retain their response and rejection. A missing valid
score cannot trigger a provider call.

The candidate changes dense/mesh/refine decisions only. Prepare is fixed. Current
numeric adapter limitations prohibit a second densify on any path because its
depth-map workspace is shared. Conditional execution must distinguish tool
behavior, and records current observation, rule, branch, successor and actual
calls. Neither a synthetic check nor a successful call proves better geometry.
Online execution is deterministic; offline evidence is denied to its sandbox.

```sh
python3 -B -m unittest discover -s experiments/geopilot_rsih_learning -p test_learning.py
node --test code/geopilot_rsih/test_program.mjs code/geopilot_rsih/test_real_boundary.mjs
```

These fixtures are implementation checks, not actual model/reconstruction results.
The active attempt and original hashes are registered in
`out/geopilot-learning-20260921/experiment.json`; never overwrite its outputs.

## History-guided single-parameter exploration

`continuation.py` compiles a separate stable P0 and verified historical attempts;
it reuses the score importer, original provider, upstream patch and numerical
entry. Version-2 candidates may change exactly one mesh parameter without a
condition. Version-1 candidates still require a meaningful condition. Neither
mode permits repeated densify or changes to prepare, tools, evaluation or export.

```sh
python3 -B -m unittest discover -s experiments/geopilot_rsih_learning -p 'test_*.py'
python3 -B experiments/geopilot_rsih_learning/continuation.py \
  --plan out/geopilot-learning-20260923/plan.json --output out/NEW-history-proposal
```

Use `program.json` and `candidate.json` from the new proposal with the same
preflight/launch commands above. The frozen plan binds knowledge and historical
request/response/run/score identities. Failed candidates remain experiences,
not stable parents. A new knowledge file does not rewrite the original
`knowledge.json`; retain each source file for replay. This bounded entry keeps
registered P0 stable; it does not implement automatic multi-scene promotion.

The `history_update` and `test_prediction` fields document the model's stated
use of history and prospective observations. They are not causal proof that
history improved the proposal. Actual geometry results and a history-ablated
control are distinct evidence. Publication still requires the existing
calibration/full-development checks; compare remains an exploratory unresolved
record, not a completed release controller. Historical internal contract labels: REQ-039, G50–G52, T033. The local project context store is not part of this public source snapshot.
