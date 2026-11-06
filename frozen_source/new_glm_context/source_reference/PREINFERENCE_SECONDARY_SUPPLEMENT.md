# Append-only secondary audit supplement

Timing: after source acquisition and CPU input selection/preflight, before any new task-model judgment. The original main schedule, 128 targets per task, prompts, four primary contrasts, interval allocation and positive-evidence criterion are unchanged. The additions below were independently reviewed before inference. They are secondary and all settings/results must be retained.

## A. Transport of observed singleton agreement

For each target, four reference calls are the two singleton repeats and the two random-context full-first repeats. A target passes the reference probe exactly when all four calls are valid and return the same target label. Qualification uses predictions and validity, never gold. It is observed agreement on these calls, not a statistical certificate, an online optimizer or an implementation of NousDB.

Report the number of targets with all four reference calls valid, the number passing, and failures/invalids among the two near-context full-first calls. Do not restrict eligibility to the all14 common-complete set; invalid near responses are not silently removed as successful transport.

On passing targets with both near calls valid, report (i) any flip: at least one near target label differs from the common reference label; (ii) stable flip: both near target labels equal the opposite reference label. Stable flip is a subset of any flip, not an independent count to add. Publish denominators, observed reference-label gold errors, and random-to-near improvements/worsenings. Empty denominators produce NA, never zero evidence. Also show overall four-call passes that cannot be evaluated because of invalid near calls. No new threshold is fitted from outcomes.

The qualification uses four calls per target; two further calls diagnose transport, six calls total. Do not call all six calls an online probing cost, or imply this procedure saves work. These records are already in the main schedule, so A adds no inference.

## B. Serial-request sensitivity on a fixed subset

For each task, use selected ordinals 0 through 31, fixed before inference. Repeat exactly their fourteen prepared main-experiment requests, including both repeats, with one prompt per LLM.generate invocation and a blocking wait for completion before submitting the next. No other request is submitted by this runner concurrently. Keep the loaded model, maximum-sequence configuration, prompt bytes, token IDs, grammar, temperature, caps, no-prefix-cache setting and parser unchanged.

Create a separate schedule with seed 2026110205 before inference and use it identically for both models. The serial sensitivity follows the complete mixed-request main run of the same model. No outcome-based restart, replacement, prompt adjustment or subset change is allowed. Keep failures and submission records. At 32 targets per task, B adds 896 requests per model, 1792 total; main plus sensitivity totals 8960 requests across both models, excluding controls.

Compare serial and mixed execution only on this identical 32-target subset, rather than comparing a subset to all 128. Recompute excess, replay, validity, gold quality and both scope contrasts for each task/model. The primary four-family positive-evidence criterion still applies only to the original main run. All serial and paired serial-minus-mixed intervals are secondary unadjusted 95% target-bootstrap descriptions. A bootstrap comparison uses the same target indices in both configurations and states its common-complete denominator.

COUNT/ratio diagnostics for this sensitivity use the fixed 32-target subset relation, with no background labels and no available-case denominator. Any missing target causes the corresponding subset aggregate to abstain. Also report stable within-target near/random changes with both replays individually stable.

Record that serial follows mixed; kernel shape, execution order and runtime history can differ. Persistence of a stable change in serial execution shows that other simultaneously submitted requests are not necessary for that observation. Disappearance does not establish that dynamic scheduling caused the original difference. This is a sensitivity check, not a pure scheduler causal experiment or an online latency comparison.

## Frozen implementation and verification

Copy this supplement into the experiment directory and hash it together with the derived serial schedule and new analysis code before the first task judgment. Preserve the original protocol and preselection freezes. Record main and serial request counts separately. Independently verify the serial prompt/grammar/token identities, serial submission sizes, all statistics and all reference-probe denominators from saved untrimmed responses.
