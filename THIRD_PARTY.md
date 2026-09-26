# Third-party components and scope

The root MIT license covers original GeoPilot work only. It does not override external terms.

| Component | Treatment |
|---|---|
| [RSI-Harness](https://github.com/CosmosMind-ai/RSI-Harness) | Commit `33c4f8dfac4359987f2e814e187de67c332498de`. No declared license in the pinned snapshot. Source and dependencies excluded from this release; setup fetches upstream separately. Review upstream terms before use or redistribution. |
| Vendored UseGeo evaluator helpers | Original MIT license and source identity retained under `code/usegeo_benchmark/vendor/usegeo-aa689753/`. Their original copyright applies. |
| [COLMAP](https://github.com/colmap/colmap), pycolmap | External numerical dependency. Not bundled or relicensed. |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | External numerical dependency. Binaries and source not bundled or relicensed. |
| UseGeo imagery, cameras and LiDAR | Not included. Obtain from providers under the original data terms. |
| Codex/native Agent host and model services | External dependencies; no host executable, credentials or proprietary model weights included. |

Research scripts and knowledge may refer to commands and measurements made with these tools. Such references do not transfer their rights to GeoPilot. Tool-help captures and publisher PDF/template files are not redistributed in this release.
