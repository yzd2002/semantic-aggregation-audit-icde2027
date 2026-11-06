# Prospective primary-family amendment for the new replication suite

This amendment is fixed before any real model judgment in the new three-family
context or SNLI studies and before any root GPU GO. The earlier private CPU
plans used the historical 98.75% nominal level. They are retained as superseded
preparation records. No historical data, recorded outcome, original statistical
implementation, point estimate, source sample, prompt or schedule is changed.
This is a new, explicitly documented analysis choice; it is not a claim of an
external preregistration for past work.

The coordinated fresh suite has exactly nine primary cases: the Cartesian
product of tasks dblp_acm, boolq, snli and models qwen, smol, glm. In each case,
the primary contrast is full-first near-versus-random target-label change minus
same-condition replay, on the common12 complete targets in mixed execution.
The three-family DBLP/BoolQ study uses its new, three-tokenizer length-matched
random contexts. Exact historical contexts, target-only, full-last, serial
execution, length-pass subsets, correctness changes and COUNT are secondary.
The frozen-policy transfer study is separate and has no new inferential claim
included in this nine-case family.

For the nine fresh primary contrasts, nominal family alpha is .05. Each two-sided
percentile interval uses confidence 1-.05/9 = 179/180 = 99.444444... percent and
quantiles 1/360 and 359/360. Positive-primary evidence requires a strictly positive
lower endpoint. The family remains nine when a case is unavailable; unavailable
cases are not called positive. All nine cases, including null, negative, invalid
and constant-model results, must be reported. No favorable scope or secondary
threshold replaces a failed primary contrast.

There are 4,000 target-bootstrap draws. Each case uses a deterministic seed equal
to the unsigned big-endian integer represented by the first eight SHA-256 bytes
of the UTF-8 string "2026110613|task|model". The source order of common12 targets is
retained. A case may use fewer than its planned 128 targets only under the frozen
common12 completion rule; the denominator and exclusions must then be reported.
The fixed-relation COUNT still requires every one of its 128 targets and abstains
on any missing valid target. An empty primary case is unavailable, never zero.

These are finite-Monte-Carlo percentile intervals describing selected targets.
Their Bonferroni nominal allocation is not a claim of exact population coverage
or exact FWER under dependence, selection bias or bootstrap approximation. About
11.11 of the 4,000 draws lie in each nominal extreme tail. Models share target
samples; they are not additional independent relations. Separate case seeds do
not establish independent data. No isolated effect of model family or size is
claimed.

The original analyze.py and secondary.py are retained byte-identically, with
historical 98.75% fields interpreted only as legacy comparisons. The authoritative
new primary criterion is the explicit global-primary result produced by
primary_family.py. The new manuscript must identify this distinction. Independent
review must verify this amendment, its code and final freeze before root GO.
