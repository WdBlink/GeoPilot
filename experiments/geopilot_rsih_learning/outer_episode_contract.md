# GeoPilot procedure: public host-component extract

This file supplies the procedure read by `outer_agent_dispatch.geopilot_procedure()`.
The seven instructions below are copied verbatim from the historical outer-episode
preparation contract. They describe the intended method; they do not assert that every
action is supported by the current registry. The dispatcher validates actual capabilities.

Historical source: `experiments/geopilot_rsih_learning/outer_episode_contract.md`.
Source SHA-256: `0aba342b7640f4cea317680467a32230e67336291d7ebdc128d04bae8b17d1af`.

The historical packet-preparation commands and private runpack are not part of this
component release. Use the [current host API](../../docs/usage.md#outer-host-api),
its offline tests and the [research roadmap](../../docs/research-roadmap.md).

## GeoPilot instruction used in the B/C native contexts
1. Inspect relevant permitted inputs and tool behavior; distinguish direct observations,
   source claims, interpretation and uncertainty. Select diagnostics that can alter a
   decision rather than listing every possible diagnostic.
2. Consult the supplied cross-task experience, if any, as conditional evidence. Check
   input/tool applicability and contrary cases. An empty experience snapshot does not
   disable learning from this task's own observations.
3. Before each intervention, save a short falsifiable hypothesis, its competing
   explanation, expected observations, fixed conditions and executable edit/command.
   A tool-capability extension must be identified as such and offered to controls.
4. Execute through the common source-bound tools. Check actual effective parameters,
   camera/raster identity, coverage and terminal status. Preserve every candidate,
   failed delivery and retry; do not replace a failed run with a recovered mesh score.
5. Request the common development evaluation after valid delivery. Update the next
   action from observed feedback; log why the hypothesis was retained or revised.
   Never fit reconstruction to reference points or use the held-out evaluation to tune.
6. Select under the frozen score/guardrail rule, then deliver the mesh, versioned
   procedure, all-attempt index and separate resource record. No gain is a valid result.
7. Distill a candidate experience entry: antecedent conditions, observation, intervention,
   outcome, counterevidence, scope and source identities. Later tasks receive only the
   independently checked frozen snapshot, not an unreviewed automatic success summary.

These instructions are supplied identically to the current G0 and GE arms. The retained
B/C heading names the historical source contexts; it is not a different current comparison.
L receives the general Agent instruction and the same capabilities; R uses host-supplied
rules. GE alone receives the frozen cross-task experience. Current-task feedback is
available to every arm. The source-file digest in each new receipt identifies this public
extract; the procedure text itself has not changed. No historical receipt is rewritten.
