# Post-hoc ratio-instability floor for unequal pool sizes

This finite-sample diagnostic is designed after the new ESCI results and both
models' outputs have been observed. It reuses the same 250 sampled pools, raw
keyed original and stable-ID reversed responses, and their existing event
records. It does not run inference, create new labels, or establish a position
effect. The two outputs must be complete, aligned to the same candidate
occurrences, and compared to a single fixed binary reference per pool.

For pool q with n_q candidates and positive counts a_q and b_q in the two
outputs, the triangle inequality gives a lower bound on the *larger of the two
query-macro MAEs* of sum_q(|a_q-b_q|/n_q)/(2Q). For unequal sizes the bound may
not be attained because reference counts are integers. Report an explicitly
constructed unrestricted reference-count witness. Construct it by assigning
the |a_q-b_q| elementary steps of weight 1/n_q to the first output until the
sum approaches half the total, choosing the closer side of the crossing. The
resulting worse MAE is an upper bound on the sharp minimum and exceeds the
triangle floor by at most 1/(2 Q min_q n_q). A witness is not the observed
reference and does not prove the triangle floor itself is sharp. Additional
duplicate-label or cross-pool constraints may restrict attainability.

Before calculation, hash this protocol, the calculation and test scripts,
selected input and raw events. Reconstruct original and reversed outputs for
Qwen and SmolLM2 using the existing event replayer, and independently verify
results by a separate strict keyed-output parser. For validation only, compute
the actual E/S-positive MAEs and check the floor. The bound itself must use no
gold labels. Use exact rational arithmetic; cover small mixed-size examples,
integer slack, missing/malformed inputs and witness construction exhaustively.

This is a post-hoc property of the selected finite pools, not a statistical
confidence interval, proof of deployment quality, independent dataset, or
general novelty claim for the triangle inequality.
