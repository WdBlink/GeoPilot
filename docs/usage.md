# Using GeoPilot

GeoPilot currently provides a host research workflow, a frozen reconstruction executor and reusable outer-host components. This page separates commands you can run now from the inputs required to adapt a new scene.

## Historical reconstruction

**Supported identity:** the restricted launcher accepts the prepared UseGeo Dataset-1 RGB-oriented bundle (224 matching images), its original lock/manifests, the registered P0 or an evidence-bound candidate, and the original runtime layout. It explicitly rejects arbitrary directories. Changing a folder name does not convert another scene into Dataset-1.

Before the commands below, follow [reproduction prerequisites](reproduction.md): macOS sandbox support; Node.js 22.19+; pinned RSI-Harness; the separate Python 3.10.20 numerical/evaluation environments; COLMAP/pycolmap and OpenMVS 2.4.0; acquired source data; frozen input and evaluator manifests. These are prerequisites, not files produced by `git clone`.

From the repository root, use a fresh output name:

```bash
node code/geopilot_rsih/launch.mjs \
  --program code/geopilot_rsih/p0.json \
  --inputs out/usegeo_benchmark/prepared/v1/inputs/rgb-oriented/Dataset-1 \
  --python out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime/bin/python \
  --scope development --preflight-only \
  --output out/my-dataset1-preflight
```

`preflight_passed` means the input/runtime/isolation checks passed; it produces no scored mesh. After checking that result, remove `--preflight-only` and use a new output such as `out/my-dataset1-run` to reconstruct. The launcher applies per-process resource protections; those are not a global research Token or API-call budget.

Expected input structure (existing prepared bundle, not a substitute manifest):

```text
Dataset-1/
  input_manifest.json
  Image_orientations_dataset1.xyz
  images/                         # the exact 224 hash-bound image files
```

Expected successful output objects:

| File / artifact | Meaning |
|---|---|
| `run.json` | Status, scope, input/program/runtime bindings and supervision outcome |
| `events.jsonl` | Actual execution events and decisions |
| `isolation.json`, `supervision.json` | Isolation checks and process resource outcomes |
| Stage artifacts and coverage manifest | Actual registrations, undistortion/depth coverage, omitted images and tool results; follow recorded paths |
| `submission/mesh.ply` | Delivered triangle mesh in the registered scene's world coordinates and metre units |
| `submission/submission.json` | Product/scene/configuration identity, geometry filename and achieved image counts |

Use the [registered coordinate contract and validator](../code/usegeo_mesh_benchmark/README.md) when interpreting the PLY. PLY alone is not a complete CRS description. Do not infer an EPSG code, axis convention, vertical datum or accuracy from its extension. Preserve the matching scene contract and camera/world transform with the product. This release does not generate GeoTIFF, DSM/DTM, texture atlases or a generic geospatial sidecar.

An execution error retains available status/logs and may have no `submission/`. Do not rename partial or fixture output into a successful submission. Existing output directories are not overwritten.

### Optional external geometry evaluation

Geometry scores require separately acquired reference data. References never become reconstruction inputs. A development evaluation of an eligible mesh uses a **new score directory**:

```bash
out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python -B \
  experiments/geopilot_rsih_learning/evaluator/benchmark.py score \
  --bundle out/usegeo_benchmark/prepared/v1 \
  --bundle-lock out/usegeo_benchmark/prepared/v1.lock.json \
  --scene Dataset-1 --submission out/my-dataset1-run/submission \
  --output out/my-dataset1-score
```

Check the terminal `score.json` status and its `run_manifest.json`; do not interpret a partial progress log as a score. Later development evaluator results keep their own version and are not relabeled as the original release-v1 benchmark. Without a valid score, deliver geometry with **quality unmeasured**, not zero error.

## Bring your own scene

The intended practical input is aerial imagery plus the available camera and georeferencing information. The current frozen launcher is not a portable adapter for it. Use [this host task template](../examples/host-task.md) to specify the job before implementing an adapter or asking your existing coding/research Agent to work on it.

| Input | What must be explicit | Why it affects reconstruction |
|---|---|---|
| Image files and IDs | Actual pixel versions, sizes, capture groups; whether resized or undistorted | Intrinsics/distortion must match those pixels; filenames are insufficient |
| Calibration, if supplied | Camera model, intrinsic units, distortion, image-to-camera grouping | Sharing is a hypothesis for compatible groups, not a universal default |
| Poses / camera centres, if supplied | Coordinate frame, axis/order, rotation convention, scale and units; acquisition accuracy when known | A world-frame camera centre is not automatically a complete pose |
| Georeferencing, if required | CRS and vertical datum or documented local-to-world transform and its source | Unknown absolute scale/location must remain unknown |
| Objective | Region and coverage, desired geometry, intended use and permitted resources | Avoid optimizing a metric while silently dropping difficult areas |
| Optional experience | Frozen source-linked entries with applicability, exceptions and evidence status | Past outcomes are guidance, not new-scene answers |
| Optional evaluation references | Separate evaluator location, access policy and compatible metric definition | No hidden reference access by the reconstruction Agent |

For a new adapter, require an output mesh plus a metadata record containing coordinate frame/units, actual cameras and image coverage, effective tool parameters, execution status and diagnostic/decision provenance. These are **adapter acceptance requirements**, not extra fields already emitted by every historical entry. If the scene has no georeferencing, explicitly deliver in a local frame. Retain excluded-image reasons and failed attempts; do not turn them into geometry scores.

A host task may diagnose and propose a configuration, but a recommendation is not a reconstructed product. A production-ready arbitrary-scene adapter still needs real input/coordinate validation, dependency discovery and an end-to-end run on that input class. [Research and portability roadmap](research-roadmap.md)

## Outer host API

The experimental modules `outer_agent_dispatch.py`, `outer_agent_receipt.py`, `outer_config.py` and `decision_state.py` can be imported using `PYTHONPATH=experiments/geopilot_rsih_learning`. They do not install a daemon or discover private credentials. `dispatch(...)` receives an explicit task, configuration registry, reconstruction/scoring boundaries, model client, experience and rules. The host validates actions and projects feedback; the model does not directly execute shell commands.

Use the [offline tests](../experiments/geopilot_rsih_learning/test_outer_agent_dispatch.py) as executable interface examples. They inject fake numerical/model responses and must not be used as geometry evidence. The current registry exposes three configurations: baseline, mesh decimation 0.25 and densification resolution level 2. The registry does not itself execute camera grouping or photo selection. A new named diagnostic hook accepts trusted host callbacks; domain-specific readers and numerical adapters are supplied by the integrator, not automatically discovered.

For **new** model comparisons, explicitly request `MiniMax-M3.1-Flash-Preview` through `HttpModelClient`, using a configured HTTPS endpoint and a process-supplied key. The client verifies the returned model identity and records actual usage. Keep the old `geopilot_v1/provider.py` and its historical `MiniMax-M3` receipts as the frozen legacy path; do not rewrite old results as M3.1.

`Policy()` has no implicit episode step/call/run/time limit; `HttpModelClient` omits `max_tokens` unless explicitly supplied. A specific preregistered mechanism comparison can supply equal explicit limits. Transport timeout/response-size protection and numerical process guards remain separate. There is no hidden global research budget. The public component checks run **no live model request** or numerical experiment. They do not establish the performance of the new state mechanism on real reconstruction tasks.

## Structured decision state

This is an **opt-in host API for development**, validated with offline fixtures. The portable checks need only Python and this checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/geopilot_rsih_learning \
  python3 -B -m unittest test_decision_state \
  test_outer_agent_dispatch.DispatcherTest.test_structured_state_tool_has_equal_capabilities_and_real_persistence
```

The test creates synthetic observations and temporary state records, then cleans them up. It needs no dataset, credentials or model service. It is an executable API example, not a reconstruction benchmark.

For an actual integration, `dispatch(...)` retains its explicit task, registry, model client, rule table, frozen experience and injected numerical/scoring boundaries. It adds:

| Argument | Required input and behavior |
|---|---|
| `diagnostics` | Optional mapping from a fixed name to `description`, `source_ids` and a zero-argument `run()` callback returning a permitted-facts dictionary. Every source ID must be bound in `task.information_sources` with a file path and SHA-256. The host checks those sources before and after the callback. |
| `read_only=True` | Allows inspection and stopping, plus state updates if enabled. It refuses numerical execution. Default is `False`. |
| `structured_state=True` | Exposes the same `update_decision_state` tool and state input to all four arms. G0/GE must record a prospective claim before numerical action and update stale rows before the next action or final stop; L/R may use the tool optionally. Default is `False`. |

Callbacks are trusted host code, not a sandbox for arbitrary model code. They must return only allowed input/tool facts. Diagnostic callbacks must not expose evaluation reference geometry or raw scores; the interface rejects prohibited score/reference fields, and geometric feedback remains behind the separate scoring boundary. The model selects an exact menu name and a reason, never a file path or executable code. Missing or changed evidence produces an explicit failure.

The state tool accepts `claims` and `experience` arrays with the row fields advertised in its tool schema:

- A claim records `claim_id`, `observation_refs`, `alternatives`, `proposed_action`, `predicted_observation`, `observed_result_refs` and `status` (`unresolved`, `supported`, `contradicted`, or `not_tested`). Before a numerical action, `proposed_action` names its exact registered configuration and result references remain empty.
- An experience row records a frozen `entry_id`, `applicability_refs` and `status` (`eligible`, `inapplicable`, or `unresolved`). The host retains the supplied entry text, provenance, conditions and counterevidence; the model cannot rewrite it or promote a new lesson through this tool.
- Observation references use actual `step-N` feedback IDs or source IDs from successful diagnostics. After execution, a supported/contradicted claim must cite that run's feedback and preserve its original prediction and action. An unresolved interpretation may remain unresolved.

New actual observations mark existing rows `needs_update`; state updates and host refusals do not create new observations. These checks establish provenance and consistency, **not scientific correctness**. A supported label is the Agent's interpretation and still needs independent evaluation. No geometry benefit follows merely from maintaining the table.

For each host-specified fresh `output` directory, the dispatcher stores:

| Artifact | Contents |
|---|---|
| `receipt.json` | Actual model/tool calls, usage when reported, failed attempts, source identities and the enabled state treatment; the top-level receipt schema remains version 1 |
| `raw/step-*.response.body` | Observed provider responses, when any arrived; private run outputs are not automatically published |
| `step-*/diagnostic/result.json` | Diagnostic source hashes, permitted facts or explicit failure, and elapsed time |
| `decision-state/state-*.json` | With the feature enabled: state snapshots, actual claim-to-action bindings, frozen experience and update trigger; paths and hashes are bound through the receipt's instruction record |

Each new model request includes the current state and the complete allowed feedback history. `previous_feedback` is null initially, then a `history_index` pointing to `history[index].feedback`; the latest full result is not serialized twice. This removes duplication without trimming facts, changing evidence permissions or imposing an episode budget. Byte reduction alone is not a measured speedup.

If bound dispatcher/state source changes during execution, the episode stops with `code_source_drift`, retaining expected/observed hashes and an invalidated table; it does not hot-load modified code. A numerical failure retains its failure state and pending updates, following the existing termination policy. No new automatic retry policy or numerical adapter is added here.

## Troubleshooting

- **Missing data/runtime or binding mismatch:** install the documented prerequisite and use the matching scene/version. Do not delete a hash check to accept arbitrary data.
- **`pycolmap` / `laspy` missing:** numerical and evaluator environments differ; use the command's intended interpreter. No need to install all numerical packages for the data-free host tests.
- **Partial scoring or resource termination:** preserve the mesh and logs; inspect host resources and the declared process guard. A rerun must use a new output and correctly bound version.
- **Unsupported intervention:** register and verify the capability before use, then give it to all relevant controls. Changing an unused JSON field is not a numerical intervention.
- **No observed benefit:** retain the attempt and its scope. Experience citations alone are not decision improvement.
