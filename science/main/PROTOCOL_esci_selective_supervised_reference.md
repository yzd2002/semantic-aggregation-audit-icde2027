# Previously trained supervised reference on the new ESCI pools

This secondary comparator was fixed after selecting the new 250 pools and after
the Qwen run started, but before examining the new pool labels or either model's
outcomes. It does not change the primary selective-minus-keyed comparison in
`PROTOCOL_esci_selective_holdout.md`. It reuses the exact 50,000 training
examples and supervised lexical model from the earlier ESCI study, without
training or threshold selection on the new pools. The new sample excludes all
normalized training queries and all training product IDs at the whole-query
level. These exclusions do not establish independence from model pretraining,
unique product text, unjudged candidates or the two derived samples' unrecorded
product IDs. Training uses 50,000 relevance labels, whereas the LLM plans are
zero-shot; this is a reference under a different information budget.

Reconstruct the original training selection from the pinned source tables and
verify every saved example/query/product/grade in the 50,000-record manifest.
Fit the original `FeatureHasher` and `LogisticRegression` using the fixed
features, library versions, C=1, L2, LBFGS, 1,000 maximum iterations, seed and
0.5 probability threshold. Preserve any convergence warning. Compare the
refitted probabilities and predictions on the original 250-pool test sample
with the previously saved reference; fail closed on a prediction difference or
absolute probability difference above 1e-6. Record the largest difference.

Then score the new 250 pools in their frozen order and save probabilities and
predictions without looking at new test labels during fit or threshold choice.
Only after both model runs finish, report pooled candidate F1, balanced
accuracy and query-macro ratio MAE on their respective common-complete sets and
on all 250 queries. Compare the supervised reference and model plans using the
same fixed query bootstrap as the main new-pool protocol; these comparisons are
secondary, descriptive and uncorrected for multiplicity. Do not equate a
calibrated/supervised reference with unsupervised inference or a production
search ranking system.
