<div align="center">

# GeoPilot
### Feedback-guided agents for aerial 3D reconstruction

[Getting started](#getting-started) · [Inputs & outputs](docs/usage.md) · [System](docs/system.md) · [Experience](knowledge/README.md) · [Research roadmap](docs/research-roadmap.md)

**Research preview · source, executable components and curated experience**

</div>

GeoPilot helps researchers turn reconstruction feedback and domain experience into **executable interventions**: inspect imagery and cameras, diagnose an unsatisfactory result, change a reconstruction procedure, and measure the delivered mesh. It combines an outer research Agent, a constrained numerical executor and an external geometric evaluator.

**Current availability:** data-free checks and host-dispatch components run from this repository. Full historical reconstruction requires the original prepared data, numerical tools and frozen runtime. The outer workflow uses a host Agent; arbitrary photo-folder reconstruction is **not yet a portable one-command service**. Start with the route that matches your inputs below.

![GeoPilot workflow](assets/geopilot-method.png)

## Getting started

### 1. Inspect the release — no dataset, credentials or model calls

Prerequisite: Git and Python 3.10 or newer.

```bash
git clone https://github.com/WdBlink/GeoPilot.git
cd GeoPilot
python3 scripts/verify_artifacts.py
```

Expected: seven published metric records, eight separately labeled alignment-diagnostic records, shared-intrinsics deltas and experience records pass the consistency check. This validates saved values; it does not reconstruct a scene or recalculate geometry.

### 2. Check the outer Agent components offline

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/geopilot_rsih_learning \
  python3 -B -m unittest test_outer_agent_dispatch test_outer_agent_receipt \
  test_outer_config.RegisteredSetTest
```

The tests exercise rule / ordinary LLM / GeoPilot without history / GeoPilot with history, using **synthetic model and numerical responses**. They check capability parity, evidence isolation, error recording and optional limits. No mesh or scientific result is produced. These components are an experimental host API, not a general reconstruction CLI.

### 3. Choose a reconstruction route

| What you have | Entry | What you get today |
|---|---|---|
| Published values and experience | Commands above | Local implementation and consistency checks |
| Original prepared Dataset-1, frozen runtime and macOS sandbox | [Historical reconstruction guide](docs/usage.md#historical-reconstruction) | Checked program execution, a world-frame mesh submission and execution records |
| Your own aerial imagery and camera/georeferencing information | [Bring your own scene](docs/usage.md#bring-your-own-scene) | An explicit intake checklist and [host task template](examples/host-task.md); data adaptation and execution require a host/developer |
| A numerical backend to integrate | [Outer host API](docs/usage.md#outer-host-api) | Tested dispatcher/receipt components; you provide verified reconstruction and scoring boundaries |

[Full dependencies and reproduction commands](docs/reproduction.md) distinguish portable checks from historical replay. No PyPI package, Linux sandbox equivalent, or standalone autonomous memory service is advertised.

## Inputs and outputs

A real reconstruction task needs more than an image directory:

- **Imagery:** original pixel files, image IDs, capture groups and their calibration relationship. Resized, distorted and undistorted images are different inputs.
- **Cameras and location:** known intrinsics/poses when available, their conventions, coordinate frame and units. Missing calibration must be estimated, not fabricated. Without georeferencing, a local-frame result must remain labeled local.
- **Task requirements:** region/coverage, intended output, allowed tools, relevant prior experience and practical resource constraints.

The implemented historical path delivers `submission/mesh.ply`, `submission/submission.json`, `run.json`, `events.jsonl`, coverage and intermediate artifacts. Independent evaluation writes a separate score and source manifest. A failed/preflight-only run is not a delivered mesh, and an unscored mesh has no claimed geometric accuracy.

PLY geometry is the supported research product; automated textured assets, orthomosaics, DSM/DTM, GIS exports and arbitrary-scene packaging remain future integrations. See [exact formats, output states and practical constraints](docs/usage.md).

## How the system works

1. **Diagnose:** distinguish observed facts, hypotheses, tool semantics and uncertainty.
2. **Use experience:** supply source-linked guidance with applicability and counterexamples.
3. **Intervene:** propose a checked program or a separately versioned configuration change.
4. **Execute and evaluate:** record actual tool calls and deliver a mesh; measure geometry outside the reconstruction process.
5. **Preserve evidence:** retain outcomes and conditional lessons for later tasks.

The original restricted executor runs without online LLM calls. The outer host is responsible for research decisions and does not automatically inherit the executor's macOS sandbox. Reference LiDAR is evaluation-only. [System boundary](docs/system.md)

The public experience is a curated snapshot, not private transcripts or an automatic distillation of every run. Observations and interpretations remain separate. [Knowledge library](knowledge/README.md)

## Research evidence

In a recorded Dataset-1 **M-input** development episode, the Agent proposed shared intrinsics before seeing the resulting improvement. A matched configuration control reduced surface L1 from 1.989684 m to 1.849701 m; precision and completeness at 0.20 m increased by 1.4462 and 0.704880 percentage points. All six original measures are available in the [measurement snapshot](evidence/measurements.json).

This supports a feedback-to-intervention case, not an established Agent or experience advantage over strong rules. There was one mesh per configuration. The ranking reversed under reference-assisted global-alignment diagnostics, so the original world-frame score should not be presented as isolated local-shape quality. [Case and diagnostic evidence](knowledge/shared-intrinsics.md)

The next study retains **Agent / experience incremental contribution** as its goal and investigates realistic diagnostic choices, camera/input compatibility and stage-specific interventions. Broader capability and stronger evidence remain research work, not released performance claims. [Roadmap and comparison design](docs/research-roadmap.md)

## News & timeline

| Date | Milestone | Status |
|---|---|---|
| **2026-09-29** | Usage and input/output guide; experimental outer dispatcher and regression checks; explicit optional experiment limits; alignment-sensitive experience update | Main-branch development update; no new geometry result |
| **2026-09-26** | Initial system source, curated experience, software citation and data-free checks | [v0.1.0 research prerelease](https://github.com/WdBlink/GeoPilot/releases/tag/v0.1.0) |
| 2026-09-25 | Feedback-led shared-intrinsics comparison completed | Single-scene development case |
| 2026-09-23 | P2 acquired valid development scores | Advantage unresolved |
| Planned | Portable scene adapters, richer diagnostic actions, core controls and independent validation | Not completed; no release date committed |
| Planned | Manuscript submission and paper metadata | Manuscript in preparation |

GitHub updates are software milestones, not paper publication or scientific acceptance. Tagged historical releases remain available.

## Repository map

| Path | Purpose |
|---|---|
| `code/geopilot_rsih/` | Restricted program representation, validation, numerical tools and execution |
| `experiments/geopilot_rsih_learning/` | Proposal/history workflows, camera diagnostics and experimental outer host components |
| `code/usegeo_mesh_baseline/` | Camera preparation and baseline integration |
| `code/usegeo_mesh_benchmark/`, `code/usegeo_benchmark/` | Evaluation, data preparation and contracts |
| `examples/` | Host task template for an explicitly specified scene |
| `knowledge/`, `evidence/` | Curated experience and source-identified scalar observations |
| `docs/` | Usage, environment, research plan, release scope and data access |
| `scripts/` | Artifact checks and separate pinned upstream setup |

## Data, citation and license

Obtain imagery and references from their [original providers](docs/data.md). Raw data, meshes, private logs, credentials, model weights and machine environments are not redistributed. RSI-Harness is fetched separately at a pinned commit; its snapshot has no declared license and is not relicensed here. [Third-party notices](THIRD_PARTY.md)

Use [CITATION.cff](CITATION.cff) for the tagged software artifact, and record the commit for development-branch use. Paper authors, DOI and publication metadata are not final. Cite the original numerical tools and data when used.

Original GeoPilot code, documentation and curated knowledge use the [MIT License](LICENSE). [Contributing](CONTRIBUTING.md) · [Release scope](docs/release-scope.md)
