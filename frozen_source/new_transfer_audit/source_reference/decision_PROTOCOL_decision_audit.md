# Actual-relation holdout execution selection

Root approved this exact design for implementation before new development judgments. GPU inference remains gated on the full freeze, independent checks, and root authorization. The implementation details file fixes seeds and lexical tie rules. Both scientific phases are frozen together; held-out inference additionally requires the post-development policy lock and explicit test release.

Status: executable design proposed to root/methods; NOT an inference authorization or a frozen implemented protocol.
Date: 2026-11-02. Author: independent novelty reviewer.
Scope: one additional empirical audit; no new optimizer, routing algorithm, or SOTA performance claim.

1. Question and identifiable increment

On a fixed relation of genuinely unobserved target judgments, does allocating the same development budget to output-key order, rather than the already-established input-order degree of freedom, change a locked physical-plan choice and its singleton fidelity/call cost?
This tests transfer of a finite execution-configuration choice. It does not identify an internal autoregressive mechanism, establish an oracle guarantee, or show one-query end-to-end savings. A null/adverse result remains a result.
Primary contrast: Output selector minus Input selector, separately in all four task/model settings. The canonical-repeat selector is a nested control. Do not replace a failed primary contrast with a more favorable fixed baseline or a test-oracle choice.

2. Data boundary, prompts, and actual execution

Use existing reserve ordinals 0--127 as exposed development and 128--255 as held-out evaluation, separately for DBLP-ACM and BoolQ. The first group has already contributed to design; label it exposed development. Reserve feature contents and public gold labels have existed locally; the intended boundary is previously unobserved LLM judgments, not unknown features/gold or pretraining exclusion.
Read-only inventory independently confirmed that current selected targets are exactly reserve ordinals 0--127 for both tasks and none of the remaining 128 source IDs occurs in the old selected target/context rows. Methods must additionally certify absence from every old prepared scientific prompt/manifest and raw execution ledger, including serial jobs, and check relevant identity/full-text overlap. Do not silently substitute easier targets after inspection.
Each split is one actual 128-row input relation T. Every row is classified exactly once per execution. All records in a batch belong to that T; there are no train-bank filler rows. COUNT and ratio use exactly these 128 occurrences.
Use stable numeric occurrence IDs 1--128 WITHIN each relation, with full identifiers (task, split, source_id, occurrence_id) in the ledger. Preserve occurrence IDs across all batch sizes/groupings/orders and singleton calls. Do not reuse the old local-ID-1 singleton judgments. Mapping both splits to the same numeric ID range avoids an unnecessary 1--128 versus 129--256 digit-length shift.
Freeze a common task instruction, feature rendering, and requested-ID set. Input-order and output-key-order interventions must not alter field content. Requested-ID list is fixed numeric ascending in both order arms; the grammar, not an instruction conflict, constrains output key order. All requested rows are output. The target-only scope is not part of this experiment.
Require tokenizer preflight for every wrapped prompt and maximum output, both models, no truncation. Preserve the already-audited model/runtime settings and raw strict parsing. Synthetic controls remain separately counted.

3. Configuration space

k in {2,4,8}; grouping g in {random, diversity}; input order u in {ascending, reversed}; output key order v in {ascending, reversed}.
Only three combinations of (u,v) are needed: C=(ascending,ascending), I=(reversed,ascending), O=(ascending,reversed). No unneeded fourth factorial arm.
Input ascending/reversed means numeric occurrence-ID order WITHIN each fixed batch; output ascending/reversed means numeric occurrence-ID order of JSON keys for that same batch. Do not renumber positions into new IDs.
The grouping must be a complete disjoint partition of T for each k and g. Freeze the partitions before any new development judgment. The same partition is shared by C/I/O and every replay.
Recommended small label-blind implementation: random uses a seeded shuffle of the 128 IDs then consecutive k-chunks; diversity uses a frozen lexical representation and greedy farthest-first grouping without replacement. For an explicit greedy rule, start each batch with the remaining record of smallest pre-frozen hash rank and repeatedly choose the remaining record minimizing its maximum cosine similarity to the current batch, breaking ties by occurrence ID. Repeat until T is exhausted. Fit TF-IDF only on the already fixed train background bank; transform both split relations without refitting. Publish exact representation/seed/tie handling and batch membership.
This diversity rule is an adaptation inspired by established diverse grouping, NOT the complete BATCHER algorithm. If methods instead uses the official BATCHER cluster-round-robin grouping, pin its feature/clustering options and document differences from its complete demonstration-selection system before inference. No choice between these implementations may use new judgments.

4. Equal-budget selectors and research union

For each task/model, run development C four replays, I two replays, O two replays, and two fresh singleton executions S1,S2.
Input selector sees C replays 1,2 plus I replays 1,2: 12 candidates (6 base k/g configurations times 2 input orders), each with two replays.
Output selector sees C replays 1,2 plus O replays 1,2: 12 candidates (6 base configurations times 2 key orders), each with two replays.
Repeat selector sees C replays 1--4 only: 6 candidates with four replays. It is a secondary nested control, not the primary comparator.
All selectors see the same two singleton references. They may not read either gold labels or evaluation outputs.
Per replay of one C/I/O type, requests = 2*(128/2+128/4+128/8)=224. Each selector's development budget is 4*224+256=1152 requests and 4*6*128+256=3328 requested item judgments. Equal calls/item counts do not by themselves establish equal GPU time or generated tokens; audit actual tokens.
The actually executed development union is 8*224+256=2048 requests per setting, not the sum of three selector budgets.
Held-out evaluation executes C/I/O two replays each plus two fresh singletons, whether or not a plan was selected: 6*224+256=1600 requests per setting. This is a full fixed reference table; only locked policies are evaluated as policies. No test oracle is deployable.
Across four settings: 8192 development + 6400 evaluation = 14592 scientific requests, before separately logged synthetic controls. Requested item judgments: 25600 development + 19456 evaluation = 45056. Check these formulae against the generated manifest; do not treat available research traces as free operational policy calls.

5. Selection rule (fixed before new development inference)

Empirical tolerance tau=0.05. It is a decision threshold, NOT a confidence certificate.
For a two-replay candidate p, d_dev(p) = mean over r=1,2 and all 128 rows of 1[B_p,r(t) differs from S_r(t)]. For the four-replay canonical-repeat selector, reference assignment is S1,S2,S1,S2. This pairs replay r with a fixed common reference and prevents choosing the favorable singleton answer per item.
Any invalid/missing/duplicated output is a failure, never agreement. A batched candidate is eligible only if its entire planned development execution and the common references are valid and d_dev(p)<=tau. Reference failures make the calibration unavailable rather than quietly shrinking the denominator; use the predeclared singleton fallback with a calibration-failure flag.
Among eligible batched candidates choose lowest execution call count (equivalently largest k), then lowest d_dev, then frozen tie order random before diversity and canonical before reversed. If none is eligible, choose singleton. Do not change tau, rank order, or candidate space after development results.
Singleton fallback preserves the intended reference execution path; it does not imply perfect replay consistency or gold accuracy. Report its measured replay disagreement instead of assigning observed fidelity 100% by self-comparison.
Use exact integer disagreement numerators (threshold 0.05 expressed without floating-point boundary ambiguity) and deterministic tie comparisons. There is no rationale for a Clopper-Pearson guarantee from dependent rows of a complete partition, or from counting replays as new independent targets.

6. Lock, then holdout

Freeze this protocol, source/preparation/analysis code, feature-only partitions, model/runtime hashes, both phase schedules, parsing rules, and bootstrap/reporting choices before ANY new scientific development call. Preserve all later amendments and failed runs append-only.
The development runner completes the entire frozen union for both tasks of a model (and ideally both models) and the analyzer mechanically writes all three selected plans for each of the four settings. Freeze/hash this decision file, including the raw-development digests and exact eligibility ledger.
The evaluation runner must require that locked decision file; it must reject missing, stale, or mismatched hashes. Complete the entire fixed evaluation table independent of which plans won. No human adjustment after inspecting development or evaluation judgments. Any code correction after outcomes is a disclosed amendment, with original evidence preserved.
All replays and types have a frozen interleaved schedule, with stable request IDs and block membership. Replays are independent executions, not a vote or an ensemble. Reusing a trace shared by multiple policy definitions is valid for comparison, but role-specific operational cost is separately attributed.

7. Primary quantities and interpretation

For each locked policy and each evaluation replay, report:
(a) fixed-denominator singleton disagreement d_test,r against S_r for batched plans;
(b) strict coverage/invalid count and empirical constraint indicator d_test,r<=0.05;
(c) execution requests N/k (or N for singleton), requested item judgments, actual prompt/output/total tokens.
For singleton policies, use the held-out S1 versus S2 off-diagonal replay disagreement (same value for the two symmetric comparisons), and use the actual S_r labels for gold quality and COUNT. Do not compare S_r with itself. Describe the resulting benchmark as measured agreement with an independent singleton run, not a deterministic oracle.
Primary vector is (mean_r d_Output,r - mean_r d_Input,r, calls_Output-calls_Input), with both per-replay differences and all constraint indicators shown. Output can improve cost while remaining empirically within tolerance, improve fidelity at equal cost, worsen one quantity, or leave the choice unchanged; these are different outcomes.
Use the words "observed dominance" only if output has no larger calls and no larger disagreement in EACH replay, with at least one strict improvement. If fewer calls come with higher disagreement but still below tau, say "lower calls within the empirical tolerance on this pool", not fidelity improvement or a quality guarantee. If a primary policy is unchanged, do not manufacture a gain from another configuration's test score.
The held-out result is a finite-relation audit with only one 128-row relation per task and two execution replays. Do not turn 128*2 labels into IID statistical replications, or claim a population superiority p-value. Report exact points and differences. If intervals are included, the choice must be frozen now; recommended optional sensitivity is resampling connected components of the union of the TWO LOCKED policies' actual batch-membership graphs, treating singleton as isolated vertices and reporting K=1 as NA. It cannot manufacture an independent population guarantee. No interval is needed to state the exact observed census of T. All four settings remain reported; no post-hoc positive-family selection.

8. Gold and aggregate outcomes (never a selector input)

For each locked policy, fixed baseline, and full-table configuration, report Y fraction, confusion counts, accuracy/error, balanced accuracy, and invalid/missing outcome counts. Include zero-call constant-Y/constant-N gold references so an all-Y high-fidelity predicate is visible. Do not use them to choose a plan.
For each replay, report G=#gold positives, predicted COUNT, signed count bias, and ratio absolute error |predicted COUNT-G|/128. An incomplete execution abstains on COUNT rather than dropping rows or inventing labels. Candidate disagreement does not imply a ratio error; opposing errors can cancel. Gold improvements/worsening and singleton fidelity are separate measurements.
Simple fixed references from the already generated table: singleton; random k=8 C; diversity k=8 C; and all k/g C rows in the artifact. These are fixed plans, not fitted/test-selected baselines. The Input selector already offers the established input-order degree of freedom at exactly the same search cardinality and calibration budget as Output.

9. Costs and scope of a useful decision

Report each policy's entire 1152-call development/calibration cost plus one actual 128-row execution; also separate the one-execution cost. One pool does not amortize this development search against 128 singleton calls. Do not call lower execution calls an end-to-end saving.
Pure request-count break-even versus repeated singleton queries is Q*=ceil(1152/(128-128/k)) when k>1: 18 pools for k=2, 12 for k=4, 11 for k=8; singleton has no finite break-even. This is only arithmetic under unchanged future plan cost/fidelity, not measured transfer to Q* new pools or a speedup. Token break-even, if shown, must use actual measured token differences and display NA for a non-positive saving. Mixed-schedule request timing is not an isolated policy latency benchmark.
Prior fixed-target findings motivate the experiment but do not guarantee that a beneficial configuration exists or transfers. A negative result means output-key sensitivity did not yield a useful selection advantage under this budget and split. It cannot justify changing tasks, models, groupings, tau, objective, or test selection.

10. Relation to closest work and the NousDB baseline decision

BATCHER already studies random/similar/diverse batch composition on DBLP-ACM and other ER data, jointly with demonstrations (ICDE 2024, III-A, VI-C, Table IV). BatchPrompt already uses input permutations and voting on BoolQ (ICLR 2024, section 2 and experiments). Sema and NousDB already compare optimized batched paths with singleton references. None should be described as ignoring batch fidelity or composition.
The narrowly tested addition is an equal-budget head-to-head between input-order and output-key-order selection, locked before a new actual relation, with separate reference fidelity, gold COUNT quality, and full calibration cost. This is empirical evidence about a configuration axis, not a novel learning/optimization algorithm.
Do NOT add a mislabelled "NousDB" row by applying its test on exposed development and carrying the chosen k to a new input. Its paper-specified calibration draws probes/fillers from the current operator input and randomizes probe slots. Our sorted/reversed production inputs and cross-relation transfer do not reproduce that sampling distribution.
A faithful paper-level Algorithm 2 comparison would require a separate query-time random-batch calibration and a matching random-slot production plan, explicit eta/delta, CP test, candidate multiplicity, early stopping, and all reference/probe costs; it would not be the official DataFusion system or the equal-budget primary comparison. With no verified public NousDB implementation and no online-SOTA claim, it is not required for THIS narrow experiment and should be omitted rather than approximated under an inaccurate name. If the paper later claims practical superiority over fidelity-certified query-time optimization, that stronger comparison becomes necessary and cannot be supplied by this protocol.

Primary sources:
NousDB v2, 4.2.1 / Algorithm 2 and 5.3: https://arxiv.org/html/2610.28551v2
BATCHER official paper: https://fmh1art.github.io/files/BatchER-ICDE2024.pdf
BATCHER official code: https://github.com/fmh1art/BatchER/tree/b55ba123834fd15163f6f2286d6d03299893ec4d
BatchPrompt, section 2 / Appendix G: https://arxiv.org/html/2309.00384v2
Sema, 5.2 / 7.3: https://arxiv.org/html/2603.11622v1
