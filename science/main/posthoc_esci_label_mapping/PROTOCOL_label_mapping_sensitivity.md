# Post-hoc sensitivity to the ESCI binary relevance mapping

This analysis was designed after the original ESCI and later holdout outcomes
were observed. It reuses the existing 250 new-query pools and their saved model
outputs; it collects no new data and runs no inference. All reported intervals
are unadjusted, descriptive query-cluster bootstrap intervals, not confirmatory
tests, causal effects, or population coverage guarantees.

The original study defines E and S as positive. For a deliberately different
evaluation target, also score the unchanged outputs with E positive and S/C/I
negative. Record grade counts, candidate F1 and balanced accuracy, and
query-macro ratio MAE. Analyze Qwen3-4B and SmolLM2 original keyed, actual
selective-fallback and pointwise outputs, the previously trained supervised
reference, and all-Y/all-N. Use the same 250 complete query pools throughout;
abort if any expected pool or prediction is missing. Pair query resamples for
the selective-minus-keyed MAE under each mapping and their difference of
differences. Use 4,000 resamples and NumPy generator seed 20271031.

The models' original prompts explicitly called functional substitutes positive,
and the supervised model was trained on E/S-positive labels. They are NOT
adapted to the exact-only target. Therefore the E-only numbers measure
sensitivity of the evaluation target applied to fixed outputs, not how models
would perform if requested or trained to solve the exact-only task. WANDS has
different definitions, samples and prompts and is not directly comparable.
This diagnostic cannot determine whether either mapping reflects shoppers'
actual satisfaction or create new annotation evidence.

Before analyzing, record SHA-256 values of this protocol, the analysis script,
the selected inputs and all eight prediction/event inputs. Original inputs,
outputs and pre-existing frozen records remain unchanged. Verify that the
E/S-positive metrics reproduce the archived analysis; independently rebuild
the selective vectors from raw events and validate the new result; use
synthetic checks for mapping, keyed parsing, missing pools and paired arithmetic.
