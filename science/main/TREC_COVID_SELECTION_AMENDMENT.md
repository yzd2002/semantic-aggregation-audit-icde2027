# Selector correction before sample creation

The fixed protocol includes only the documented TREC-COVID judgments 0, 1 and 2. The initial implementation incorrectly aborted when it encountered a different value. The downloaded NIST file actually contains two rows with grade `-1`; their meaning was not established from the official three-grade description. The initial selector therefore exited before creating any sample directory or running inference. Its original frozen source, manifest and error log are retained.

Before selecting any candidates, `select_trec_covid_transfer_v2.py` was copied from that frozen source and changed to exclude and count grade `-1` explicitly; it still rejects other unexpected grades. The two rows are not treated as negatives and are not candidates for sampling. The protocol's sample size, seeds, eligibility criteria, metrics and model procedure are unchanged. The corrected implementation and this amendment are separately frozen before sampling.
