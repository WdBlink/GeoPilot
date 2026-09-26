# Equivalent scorer candidate v3

Independent candidate copied from release-v1 `usegeo_mesh_benchmark/benchmark.py`.
This is not the accepted release-v1 runtime and is not an evaluator release acceptance.
The only protocol value changed is `runtime.implementation_id`:
`usegeo-mesh-geopilot-equal-v3-adaptive-batch`. Consequently protocol SHA differs;
results cannot be mixed into old-runtime aggregates. Submission protocol/version is unchanged.

Changes: keep the last successful reverse-query batch size (still count/check each batch),
emit stage/reference progress into sibling `OUTPUT.progress.jsonl`, retain known bindings
and observed RSS on failures, execute this exact new file as child. Parent/worker limits,
Float64 distance routine, stable tie order, all reference points, 1M surface samples,
seed, <=0.20 threshold, numerical environment remain unchanged. One safety difference:
when even one query exceeds the 2M candidate-pair cap, fail explicitly instead of the
old size==1 exception permitting an over-cap allocation. No reference point is skipped.

The immutable result directory still contains exactly `score.json` and `run_manifest.json`.
Progress sidecars use exclusive creation; choose a fresh output for every invocation.
Source SHA is in `score_start`; source/protocol identity and diff are in evaluator-tests.
Unknown peak RSS remains null on an early failure, never invented as zero.

From repository root, existing frozen validator (do not replace it):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python -B -m usegeo_mesh_benchmark.benchmark validate-submission --bundle out/usegeo_benchmark/prepared/v1 --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json --scene Dataset-1 --submission out/geopilot-rsih-p0-mu9geojt-r3/submission
```

Candidate scoring command (requires independent review before full execution):

```sh
PYTHONDONTWRITEBYTECODE=1 out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python -B experiments/geopilot_rsih_learning/evaluator/benchmark.py score --bundle out/usegeo_benchmark/prepared/v1 --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json --scene Dataset-1 --submission out/geopilot-rsih-p0-mu9geojt-r3/submission --output out/geopilot-learning-20260921/evaluator-r3-score
```

Checks: `test_equivalence.py` (synthetic exact/brute-force equivalence and failure limits),
`probe_real.py` (bounded 100k real reference windows; NOT a benchmark score). Execute both
with the frozen clean-runtime Python. Local speed ratios do not establish full-scene
completion within 4h. No reconstruction needs repeating for this candidate score.
