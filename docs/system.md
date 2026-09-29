# System and experience boundaries

GeoPilot has two observed levels. The restricted program compiler/executor works within a versioned tool registry. The outer native-host research workflow inspects development feedback and diagnostics and can write separate configuration or adapter experiments. This broader system description was made explicit retrospectively; historical restricted programs are not relabeled as having performed actions unavailable to them.

A practical outer research episode follows:

1. Read permitted input metadata, tool semantics, completed outcomes and execution failures.
2. Separate observed behavior from a causal hypothesis and identify an alternative explanation.
3. Save a concrete intervention and a matched control before observing their comparison result.
4. Execute in fresh output locations; retain the actual tool versions, failures and resource records.
5. Evaluate numerical products externally, then record conditional guidance with its source.

This describes the demonstrated host workflow, not an additional software daemon. No autonomous general memory-distillation/retrieval service is released. The executable offline prompt and history construction are in `experiments/geopilot_rsih_learning/learning.py` and `continuation.py`.

The restricted history compiler consumes selected validly scored attempts and registered knowledge. Unscored crashes can inform the outer workflow but are not converted into valid geometric scores. Knowledge changes are external context changes, not model-weight training.

## Optional host decision state

The experimental outer dispatcher now accepts named, source-bound diagnostic callbacks. The host supplies their code and returns permitted facts; the model can select only an advertised name. With `structured_state=True`, all four arms also receive `update_decision_state`. G0/GE must maintain explicit claims, alternative explanations, predictions, observed-result references and frozen-experience applicability. The general LLM and rule arms may use the same tool without that enforcement. Default execution leaves this feature disabled.

Actual tool feedback marks existing rows as needing an update. State updates are not new evidence, and a model cannot create observations by writing a prediction. Before an executed claim can be marked supported or contradicted, it must cite the corresponding run feedback. This verifies an evidence reference, not the scientific validity of the interpretation. Host-owned state snapshots and action bindings accompany the receipt; source-code drift stops the episode. This mechanism has been checked offline only. See [the API and output contract](usage.md#structured-decision-state).
