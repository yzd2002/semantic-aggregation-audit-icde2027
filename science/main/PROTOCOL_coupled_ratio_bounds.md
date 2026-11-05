# Exact conditional bounds with shared duplicate references

This post-hoc mathematical extension concerns a single fixed query and candidate pool. It introduces no new model run, human label, estimated error budget, or accuracy claim. The group partition must be supplied and independently justified: all occurrences in one group are assumed to share one binary reference, and the only constraints between groups are the accepted-label error budget and the group partition. Equal text or different source IDs do not establish or rule out product identity. No assumption is made about a missing label beyond the shared group reference.

For a group g, let a_g, b_g, and u_g be the numbers of accepted Y, accepted N, and missing judgments, and let t_g = a_g + b_g + u_g > 0. Assigning the shared group reference 0 incurs a_g accepted-label errors and contributes 0 positives; assigning reference 1 incurs b_g errors and contributes t_g positives. If G groups contain n > 0 occurrences and m accepted judgments, let integer k lie in 0..m.

`coupled_ratio_bounds.py` maintains, for each processed group and each exact error count e <= k, the minimum and maximum reference-positive count realizable at e errors. It updates these states with the group's two assignments. Taking the minimum and maximum over all final states yields the exact conditional endpoints, divided by n. Every reported endpoint has a witness assembled from group choices. If no state exists, the shared-reference and at-most-k-error assumptions are incompatible; equivalently, sum_g min(a_g,b_g) > k. With G groups, the method takes O(G(k+1)) time and O(k+1) state memory, excluding input storage. It returns exact rational numbers and does not assume continuous feasibility between endpoints: some interior counts can be impossible.

`test_coupled_ratio_bounds.py` independently enumerates all set partitions of pools of sizes 1 through 6, all missing/N/Y occurrence observations, all admissible k, and every possible assignment of group references. It checks endpoint equality, infeasibility, containment in the unrestricted-occurrence bounds, equality to those bounds for singleton groups, and input validation. The saved `COUPLED_RATIO_BOUNDS_TEST.log` records 799,598 partition/observation/budget cases, including 95,370 infeasible cases and 250,462 cases with unattainable interior counts. These are finite synthetic checks supporting, not replacing, the state-induction argument.

The earlier audit found no repeated source ID within any of 1,018 saved raw-source pools across seven groups. The two derived ESCI samples do not retain source IDs, and different IDs may denote the same item or share text. Thus this extension has no demonstrated empirical effect on those recorded raw-source pools, and we do not interpret it as a measured system benefit. Product identity, a defensible error budget and any additional external constraints require independent evidence. The interval is not a statistical confidence interval.

Run from the artifact directory:

```bash
python3 test_coupled_ratio_bounds.py
sha256sum -c COUPLED_RATIO_BOUNDS_SHA256.txt
```
