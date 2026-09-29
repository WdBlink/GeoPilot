# Host task template: one aerial reconstruction job

This is a request template for a host Agent/developer, **not** a config accepted by the frozen launcher. Replace the placeholders with real local inputs. Do not put credentials or private data in public issues.

```text
Goal: reconstruct [region/site] for [intended use].
Inputs: [image directory], [image IDs/capture groups], [actual raster version].
Cameras: [calibration/poses/centres and formats, or explicitly unavailable].
Coordinates: [frame/CRS, units, axes and vertical datum, or local/unknown].
Permitted tools and changes: [installed tool versions and allowed interventions].
Coverage: [required area/images and acceptable omissions].
Experience: [source-linked frozen snapshot, or none].
Output directory: [fresh empty path].
Resource protection: [actual machine memory/disk and individual process limits].
Evaluation: [separate evaluator and reference access policy, or unmeasured].

First verify image-camera-coordinate compatibility and available capabilities.
Separate facts, hypotheses and missing information. Choose a diagnostic whose
result can change the next action. Record the proposed intervention and its
alternative explanation before execution. Preserve all attempts and exact
inputs/tool settings. Return the actual mesh and its coordinate metadata,
coverage and execution/decision records; if blocked or failed, return that
status rather than claiming a mesh or score exists. Do not use evaluation
reference geometry to fit reconstruction unless this is a separately labeled
reference-assisted diagnostic. Do not publish input data or submit a paper.
```

Follow [input/output requirements](../docs/usage.md#bring-your-own-scene). A host must implement a new data adapter before claiming the existing benchmark launcher supports this scene. This template establishes the job; it does not supply an autonomous system or evidence of Agent superiority.
