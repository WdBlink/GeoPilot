# Research roadmap: decisions that can change a reconstruction

Updated 2026-09-30. The primary research goal remains **incremental value from the Agent and frozen experience**. The current manuscript is a working draft. This roadmap is not a completed experiment, performance promise or final preregistration.

## Why broaden the decision domain?

If a few configurations can be exhaustively evaluated, a strong enumerating baseline already selects the best observed candidate. An Agent may reduce the work needed to reach an acceptable result; a bigger Cartesian parameter grid alone does not prove intelligence or novelty. Tasks should come from actual reconstruction problems, permit different useful actions, and let diagnostics change the decision.

## Candidate task families

| Real need | Decision to study | Implementation status |
|---|---|---|
| Image/calibration mismatch or heterogeneous camera groups | Check pixel-camera bindings; choose authorized matching inputs; test fixed, per-image or group-shared intrinsics | Historical diagnostic scripts exist; named host callbacks are exposed, but generic input/calibration adapters are not released |
| Similar aggregate error with competing explanations | Choose effective-parameter, projection, coverage or coordinate-metadata diagnostics before changing a parameter | Named read-only diagnostic interface is available; each domain reader and downstream intervention still needs its own verification |
| Dense/mesh/refine interactions and limited resources | Change effective stage settings or resume a valid dependent stage | Registered inner parameters and a three-configuration outer registry exist; richer versioned surface pending |
| Relevant versus conflicting experience | Select sourced guidance, check applicability and decide whether to act or gather evidence | Curated knowledge and optional source-bound applicability state exist; general automatic recall/distillation remains pending |

Image selection must preserve the task's fixed target region and report omissions; removing difficult areas must not create an apparent gain. All baselines receive the same legal inputs and tools. Do not hide observations from a baseline or design tasks after seeing which condition wins.

## Execution order

1. **Development diagnosis and wiring.** Start with real input/camera-binding and effective-parameter/next-diagnostic cases. Verify that each candidate action reaches the intended tool, changes relevant artifacts and has a defined failure/unknown path. Keep a source-linked task inclusion ledger. Cases where every reasonable strategy takes the same action remain useful controls, not proof of Agent value.
2. **Freeze and compare.** Strong fixed/adaptive rules, ordinary LLM tool use, GeoPilot without historical experience, and GeoPilot with a frozen experience library share capabilities. New model arms use MiniMax-M3.1-Flash-Preview; historical M3 observations keep their original identity. Where feasible, include search/enumeration controls. The ordinary LLM also receives its own run feedback, so this design does not isolate feedback alone.
3. **Independent validation.** Separate new scenes early, and evaluate only after code, tools, experience, selection policy and evaluation choices are frozen. A scene used for tuning becomes development data. Repeat reconstruction separately from repeated sampling of one mesh; neither measures systematic reference error by itself.

Before the corresponding measurements, declare actual task membership, primary metric and meaningful effect, guardrails for the other metrics, selection/stopping rules, repeat structure and resource accounting. These numbers are not invented here. If the task is world-coordinate delivery, say so; reference-assisted alignment is a separate diagnostic and cannot silently become a deployed repair.

Measure **executed intervention, final geometry and valid delivery, failed attempts, diagnostics, full time and resource breakdown**. Record whether selected experience entered the proposal and changed the action. A citation or more convincing explanation alone is insufficient.

Mechanism comparisons may match a specific resource such as candidate evaluations. Cross-system comparisons report quality and resources transparently rather than treating Tokens, network waits and numerical work as one compute unit. No global Token/model-call cap is introduced. Individual process protections are recorded separately.

Jev remains an optional local-decision component: compare it only where it replaces a real LLM decision and measure downstream effects. Main research does not wait for Jev to win. Camera component controls do not substitute for Agent/experience controls.

## Practical tool roadmap

The next portable scene adapter should accept actual image/camera/georeferencing bundles, validate their compatibility, expose effective tool settings, and produce a mesh with coordinate/coverage/provenance metadata. General images-only SfM, calibrated-camera ingestion, textured/GIS products and cross-platform execution must be individually tested before being called supported. The current [usage guide](usage.md) identifies what runs today.

## Component status: optional structured decision state

`structured_state=True` exposes the same update tool to all four arms. G0/GE must register a prospective observation and revisit their claim/applicability rows after actual feedback before another numerical action or final stop. L and R can use the same tool optionally. The host checks structure, observed IDs and executed-action bindings; it does not decide whether a scientific claim is true. Frozen experience text stays intact.

This mechanism has passed offline tests with synthetic boundaries. It has not established an Agent/experience benefit in a real-model reconstruction comparison. The current three registered numerical configurations are unchanged. New ETH adapters and ongoing Dataset-2 development experiments are outside this component update and are not advertised as portable services or new results. The next research comparison must freeze the mechanism and provide the same legal inputs and numerical tools to its controls.
