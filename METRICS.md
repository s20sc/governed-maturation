# E5 metric definitions

The complete fixed protocol and interpretation boundaries are in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Each metric is recorded per
seed, scenario, updater and configuration, and embedded explicitly in E5 JSON.

- **final_confidence**: Final synthetic.skill confidence; T5 proposals affect other capabilities, so this remains at the initial value there.
- **minimum_confidence**: Minimum of the initial synthetic.skill confidence and the affected capability confidence after each applied proposal.
- **final_human_threshold**: Final synthetic.action human-check threshold, with initial/default value 0.5.
- **total_variation**: Sum over applied proposals of absolute changes in the affected capability confidence plus the synthetic.action human-check threshold.
- **applied**: Number of applied evidence-driven proposals, excluding the initial-condition event.
- **authorization_rejected**: Number of proposals rejected by the resolver signature-presence gate before entering governance history.
- **validation_rejected**: Number of unsupported-capability proposals rejected by EMAValidationBackend before entering governance history.
- **floor**: Number of proposals blocked by the selected skill-confidence floor or the independent harness human-threshold ceiling.
- **rate_limit**: Number of proposals clamped by the production per-kind rate predicate after passing the circuit check.
- **circuit_breaker**: Number of proposals clamped by the production circuit predicate before the rate and floor checks.

Also retain attempts, governance_rows and first_circuit_attempt (1-based, null
if absent), evidence hashes and initial state. These are not recovery costs,
time to real threat detection, operational availability or task reward.

Paired contrasts are EMA minus discrete for every grid/profile/scenario and
ablation minus all within each updater at reference thresholds. Each metric's
95% percentile CI resamples **seed differences**, 2,000 replicates, fixed bootstrap
seed 20261021. Percentiles use sorted index floor(p*(B-1)); no independent arm
resampling, no p-value or multiple-testing claim. A zero-width interval often
reflects a deterministic gate, not general certainty. The independent unit is
a synthetic seed trace; these intervals do not generalize to deployments.

