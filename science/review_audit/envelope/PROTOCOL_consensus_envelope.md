# Consensus-preserving count envelope: diagnostic protocol

Recorded on 2026-11-02 before computing this diagnostic. This is a post-hoc
analysis of already exposed, saved results, requested during independent review.
It is not a new theorem, algorithm, online routing policy, or prospective test.
No GPU inference, new labels, source replacement, or changes to methods files.

## Frozen scope

Analyze all four previously compared product settings: Qwen and SmolLM2 on the
52 WANDS selective-execution pools and 250 ESCI selective pools, totaling 604
pool-model observations. The same human relevance mappings and candidate pools
as the original runs are retained. Reconstruct original keyed, reversed keyed,
and actual selective vectors from source traces and compare to the existing
`methods/consensus_audit.json` record. WANDS actual selective is distinct from
the cached-full-pointwise variant used in the equal-label-budget ablation.

Missing/incomplete or conflicting records are errors to disclose, never silently
excluded. All settings and every pool-model record will be retained, including
unfavorable or null outcomes. No model/pool selection based on these results.

## Definitions

For each complete, occurrence-aligned pool of size n, let a and b be the two
keyed binary vectors, h the actual selective vector, and s the reference vector.
Define L=sum(a AND b), U=sum(a OR b), G=sum(s), and P=sum(h).
Check 0<=L<=P<=U<=n and verify h retains every agreement of a and b.

Holding agreement fixed but permitting any binary choices on disagreements,
every integer positive count from L through U is attainable. The minimum
attainable ratio error is

    floor = max(L-G, 0, G-U) / n.

This comparison class allows arbitrary occurrence-level decisions; it does not
impose any additional coupled-duplicate consistency constraint. It is broader
than the outputs of the particular recorded pointwise checker.

For j in AND, OR, actual-selective, report

    error_j  = abs(P_j-G)/n
    regret_j = error_j - floor,

where P_AND=L, P_OR=U, and P_actual=P. Regret is a descriptive excess loss over
the hindsight envelope floor, not an online or learned policy regret guarantee.

## Mutually exclusive composition groups

Apply the following rules in this exact order:

1. collapsed_exact: L=U=G;
2. at_or_below_lower: G<=L, excluding rule 1;
3. at_or_above_upper: G>=U, excluding rule 1;
4. strictly_inside: L<G<U.

These four groups partition all records. In particular G=L<U belongs to group
2, L<G=U belongs to group 3, and L=U with G unequal belongs to the appropriate
outside group. Also retain L,U,G,n per row so endpoint and strict-outside counts
can be independently checked.

## Reporting and verification

Report each setting separately: group counts/shares, query-macro mean floor,
mean AND/OR/actual-selective errors and regrets, exact-floor-attaining pool
counts, and group-specific means. All averages weight pools equally; repeated
model executions are not new independent pools. No new confidence intervals,
interaction tests, or population claims are introduced in this diagnostic.

Keep a per-pool JSONL record and source paths/hashes. A separate verification
implementation must independently reparse source records, recompute integer
L/U/G/P, exact rational losses and classification, check every saved row, and
reconcile aggregate AND/OR/selective MAEs with the existing consensus audit.
The verifier must not import the envelope calculation functions or treat the
derived per-pool file as its source of truth.

G is required to locate the reference inside/outside the envelope and to choose
the hindsight-best endpoint. G is unavailable at online execution, so this
analysis does not authorize an AND/OR selector or certify unseen outputs.
If G>=U, OR attains the floor within this comparison class; if G<=L, AND does.
These elementary facts are explanatory diagnostics, not new claimed results.
