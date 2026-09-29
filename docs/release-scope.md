# Public release scope

This is a public source and evidence snapshot accompanying ongoing GeoPilot research. It includes project-specific implementation and curated experience, not the entire private development workspace or a complete numerical dataset archive.

The manuscript is in preparation. No preprint, journal submission, acceptance or paper publication is asserted. Paper authorship is still incomplete; the software citation identifies the GitHub maintainer account, not a finalized paper author list.

## Included and excluded

Included: original system/experiment source, original diagrams, selected scalar measurements with source identities, curated domain knowledge/experience, installation and reproduction boundaries, and synthetic/data-free checks.

Excluded: upstream RSI-Harness checkout and npm dependencies; private sessions and personal configuration; credentials; images/LiDAR/depth/meshes/checkpoints; publisher articles and official manuscript templates; the unfinished manuscript PDF. Included third-party UseGeo helper files retain their MIT notice.

The root MIT grant covers original GeoPilot files only. The release does not supply a blanket license for separately acquired tools, hosts, services or data. Native host orchestration is documented as a workflow; a portable automated memory engine is not claimed.

README organization was informed by the public VGGT and DUSt3R research repository patterns (overview, setup, research updates, data and citation), using original project-specific wording. See https://github.com/facebookresearch/vggt and https://github.com/naver/dust3r .

## 2026-09-29 main-branch update

Adds usage/IO documentation, a host task template, the experimental outer dispatcher/receipt/configuration modules and offline tests. Their numerical and scoring interfaces are injected; the new components do not make historical scene adapters portable. Original numerical tools, scoring definitions, legacy M3 client and measurements retain their identities. The current experiment-limit defaults are explicit `None`; historical runs are not reinterpreted. No raw model responses, private sessions, data or manuscript PDF are added.

README information order also draws on the public [COLMAP](https://github.com/colmap/colmap) and [VoltAgent](https://github.com/VoltAgent/voltagent) READMEs: concrete setup and entry points, then deeper architecture. No external artwork or wording is copied.

## 2026-09-30 component update

Adds original GeoPilot decision-state bookkeeping and tests, and updates the public dispatcher with named read-only diagnostic callbacks, accurate failed-attempt accounting, transport failure metadata and a latest-feedback reference instead of duplicate context. Existing receipt/configuration modules and the three registered numerical configurations retain their source identity. The public procedure extract and historical measurement records are unchanged.

The new `decision_state.py` uses only the Python standard library and existing GeoPilot interfaces. These files and tests contain original project implementation; no upstream Agent framework, model response, diagnostic dataset, private session or third-party source is copied. Test fixtures are synthetic. The state feature is opt-in and has only offline validation; it is not a scientific truth checker, autonomous knowledge-promotion service or proof of Agent/experience advantage.

No current ETH adapter, private host/numerical runner, unfinished Dataset-2 experiment, manuscript or new geometric result is included. Source hashes written by future host runs describe that run; they do not replace or relabel the recorded identities of earlier research.
