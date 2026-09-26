# E01 — Shared intrinsics as a testable development intervention

**Status:** favorable measured configuration difference on one development scene. Not a feedback-policy ablation or transfer result.

**Observed problem:** historical reconstruction produced meter-scale camera-center fitting residuals, while each image had independently optimized intrinsics despite a homogeneous model/focal-length input record. This justified a sharing hypothesis; metadata alone did not prove one physical camera or constant calibration.

**Recorded chronology (25 September 2026, UTC):** camera-residual diagnosis 04:07; explicit sharing hypothesis 04:37; matched-arm specification 04:43; BA execution 04:52; first favorable forward mesh measurements 06:38; complete six-metric report 10:33. The local audit uses timestamped host messages and actual execution artifacts; private sessions are not redistributed.

**Intervention:** same saved sparse model, same initial intrinsics, poses, points and observations; A retains separate per-image intrinsics, B uses one shared intrinsic model. Both use the same BA settings and fresh undistortion/dense/mesh stages. No evaluation LiDAR is used for fitting. The principal point stays fixed.

**Result:** all six measured directions favor B, including L1 −0.139982802281 m, precision +1.4462 percentage points, completeness +0.7048799171 percentage points. Exact source values are in records A/B of `../evidence/measurements.json`.

**Derived guidance:** for a structurally homogeneous input group, test intrinsic sharing under matched initialization and solver conditions before assuming per-image freedom is helpful. Verify camera grouping and regenerate downstream products after changing cameras.

**Limits:** one reconstruction per arm; low absolute threshold coverage; no frozen unseen-scene replay. Always-shared selection could explain the chosen configuration. The historical restricted `prepare` action could not make this edit; the outer research Agent implemented it in experiment code.

**Source identities:** diagnosis SHA256 `cf9fdc5e556e8448ec5527384025974ab2564cb493fbcc192c39792e37822371`; pre-execution plan SHA256 `b0cfebf08565238155814508588306b3805a2641401b2bd4efd39765ec0ddba1`. These identify original non-redistributed local records. The matched implementation is `../experiments/geopilot_rsih_learning/shared_camera_ba_diagnostic.py`.
