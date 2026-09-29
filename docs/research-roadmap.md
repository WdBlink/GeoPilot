# Research roadmap: decisions that can change a reconstruction

Updated 2026-09-29. The primary research goal remains **incremental value from the Agent and frozen experience**. The current manuscript is a working draft. This roadmap is not a completed experiment, performance promise or final preregistration.

## Why broaden the decision domain?

If a few configurations can be exhaustively evaluated, a strong enumerating baseline already selects the best observed candidate. An Agent may reduce the work needed to reach an acceptable result; a bigger Cartesian parameter grid alone does not prove intelligence or novelty. Tasks should come from actual reconstruction problems, permit different useful actions, and let diagnostics change the decision.

## Candidate task families

| Real need | Decision to study | Implementation status |
|---|---|---|
| Image/calibration mismatch or heterogeneous camera groups | Check pixel-camera bindings; choose authorized matching inputs; test fixed, per-image or group-shared intrinsics | Historical diagnostic scripts exist; not exposed by the outer dispatcher |
| Similar aggregate error with competing explanations | Choose effective-parameter, projection, coverage or coordinate-metadata diagnostics before changing a parameter | Script-level evidence exists; unified diagnostic-action interface pending |
| Dense/mesh/refine interactions and limited resources | Change effective stage settings or resume a valid dependent stage | Registered inner parameters and a three-configuration outer registry exist; richer versioned surface pending |
| Relevant versus conflicting experience | Select sourced guidance, check applicability and decide whether to act or gather evidence | Curated knowledge exists; general automatic recall/distillation pending |

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
