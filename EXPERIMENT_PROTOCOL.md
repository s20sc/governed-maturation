# Synthetic matched-controls experiment protocol (E5)

## Question and scope

Does the numerical update rule change state trajectories when authorization,
floor, rate, circuit, validation and evidence schedules are held equal? This
is a synthetic state-machine benchmark. Confidence trajectories do not establish
task success, safety, useful adaptation, identity equivalence or deployment robustness.

## Fixed parameters

Seeds are 4200–4229 (30 paired seeds). No seed selection after results. A seed
sets initial skill confidence uniformly in [.55,.65) and trace length, with
unobserved capability identifiers also varied for T5. Every configuration and
arm receives the identical immutable per-seed scenario trace; output includes
its SHA-256. One proposal per evidence item, identical timestamps, evidence,
gate order and proposal opportunity schedule; no thinning or batching.

Two numeric update rules: discrete fixed magnitude alpha*.6; EMA magnitude
alpha*abs(target-current). Alpha=.08, except .008 in the flood scenario so the
per-kind rate limit can be seen before the floor. Target is zero for failure
and one for success or human-threshold tightening. The fixed step matches the
EMA decrement at confidence .6. This normalization is a chosen benchmark
convention, not a fitted/optimal rule, nor evidence of equal learning quality.
EMA here means exponential smoothing toward a binary synthetic target; it is
not a Bayesian estimator. Both arms encode their numeric deltas as typed events
and fold actual accepted history through effective_persona. This is a comparison
of update laws at matched opportunities, not continuous-time integration versus
an asynchronous production emitter. A minimum proposal magnitude of 1e-12
avoids zero deltas at numerical saturation. Actual resolver hard [0,1] clamps
remain in every arm and ablation.

Pre-existing confidence is an explicit initial state from one decrease event,
excluded from governance traffic, dated two days before all scenarios. Initial
human threshold is .5. Each scenario starts independently; no state carries
over between scenarios or seeds.

* T1_success_only: 24–31 successes propose unsigned confidence increases. There
  are no failure evidence items in T1. Each proposal is actually passed into
  effective_persona, whose ResolverRejection is counted; rejected proposals do
  not mutate state or enter governance history. no_authorization explicitly
  inserts a HARNESS-AUTHORIZATION-BYPASS presence sentinel; rate limits remain.
* T4_failure_flood: 125–132 failures spaced one minute apart. Every proposed
  failure delta reaches governance, including previously clamped proposals.
  The real rate limit applies after 50 rows; real evaluate_governance returns
  circuit_breaker for attempt 101 onwards after 100 prior rows. This tests
  governance circuit clamping, not notification enqueue/delivery or a persisted
  runtime freeze flag. Clamped rows still count, per production aggregate_count.
* floor_pressure: 72–79 failures spaced one hour apart, allowing the real 24h
  rate limits to slide out so floor effects remain observable independently.
* T5_hallucination: 16–23 proposals for capabilities without any observations.
  The actual EMAValidationBackend.validate receives a failure-rate dictionary
  containing only decoy capability synthetic.skill at 1.0; the absent capability
  defaults to rate zero and proposals are rejected. no_validation bypasses only
  that call; accepted proposals still traverse resolver and governance. No LLM,
  parser, reflection runner, or fabricated real-world failures are involved.
* ceiling_pressure: 64–71 human-check tightening proposals one minute apart,
  using an explicitly labelled presence-only authorization sentinel. This is
  an independent harness stress test of a parameterized human ceiling, not a
  production cryptographic operator workflow. The trace is long enough for both rules to attempt crossing .99; the
  circuit does not activate before the trace ends.

## Factor isolation and production/harness boundary

Profiles: all, no_authorization, no_floor, no_rate, no_circuit, no_validation.
Exactly one protection is disabled per ablation. Numeric EMA versus discrete is
the updater factor, independent of the evidence validation gate despite its
EMAValidationBackend name. Validation is only applicable to T5; do not run
success proposals through this tighten-only gate and mistake its always-reject
relax rule for the authorization factor. Other profiles can therefore have
identical results in a scenario: those null contrasts are retained.

At the production default floor .05, the fully protected arm calls actual
evaluate_governance. Ablations use actual circuit_breaker_active,
kind_class_at_limit and would_violate_floor predicates in the same circuit,
rate, floor order, omitting only the designated predicate. Other floor values
use the same threshold comparison in an independent harness extension without
patching global constants. Blocked proposals get applied=False; accepted state
comes from actual effective_persona over accepted events. This is in-memory
history with no FIFO truncation, no checkpoint and no database.

IMPORTANT: HUMAN_CHECK_THRESHOLD_CEILING=.95 exists in governance, but
would_violate_floor only handles AUTOMATIC_TIGHTEN_KINDS; human tightening is
always operator-signed and is not in that set. The benchmark's ceiling check is
therefore independently implemented, applied equally in both arms, and cannot
be cited as existing production enforcement. no_floor disables both the skill
lower bound and the human upper bound, which affect different scenarios; risk
appetite is not an experimental outcome here.

Default effective_persona enforces operator_signature presence, with a permissive
default cause verifier. Main traces test this real presence check only. A
separate signature_probe records actual missing-signature rejection plus a
forged Ed25519 envelope rejected through make_chain_verifier supplied to
**effective_persona**, and a correctly signed positive control that increases
confidence from .6 to .7. The deterministic test key is generated in memory and
never deployed. That probe verifies envelope authenticity under one trusted test
key; it does not establish a distinct operator/resident authority model,
replay resistance, complete history, anti-rollback, or truthful evidence. No
cryptographic claim is inferred from the harness sentinels.

Parameter sensitivity is the complete 3x3 grid: skill floor .05/.1/.2 and human
ceiling .9/.95/.99, for both fully protected updater arms and all seeds/scenarios.
The reference .05/.95 are reference defaults; alternatives are chosen nearby
stress values, not empirically optimized or theoretically safety-calibrated.
Ablations run at .05/.95 only. Lower floors and higher ceilings simply allow
more numerical movement; the benchmark cannot identify a best value.

## Predefined metrics and inference

All metrics are per seed, scenario, arm and configuration:

* final_confidence: final synthetic.skill confidence (T5 uses other capabilities,
  so this metric intentionally stays at initial state there).
* minimum_confidence: minimum of initial skill confidence and each applied
  proposal's affected capability confidence; not a task safety measure.
* final_human_threshold: final synthetic.action threshold, default .5.
* total_variation: sum of absolute applied changes in the affected capability
  plus human threshold. This exposes T5 movement on unobserved capabilities.
* applied: number of applied proposals, excluding the initial condition.
* authorization_rejected and validation_rejected: proposal counts stopped at
  those gates, without entering governance history.
* floor: number of proposals blocked by the selected skill floor or harness
  human ceiling; rate_limit and circuit_breaker: actual predicate clamp counts.

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

## Reproduction and provenance

From the artifact root, use Python >=3.11 and the environment instructions in
README.md. Run the E5 driver with a fresh output path:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/persona_experiment_e5_matched_controls.py --output /tmp/persona-e5/e5_matched_controls.json
```

Defaults produce 4,200 individual runs (14 configurations x 5 scenarios x
30 seeds x 2 arms) and 120 paired contrast sets, with 2,000 bootstrap replicates.
The imported benchmark also accepts exactly one seed for diagnostic portability
checks, using the same computation; such a run is not a 30-seed experiment.
The bootstrap minimum is 100 replicates. Other seed sets require at least 30
distinct seeds. The CLI retains the full default seed set.

Result metadata contains explicit definitions for all ten metrics and SHA-256
hashes with paths relative to the artifact root. The scope is all Python and
text files under src/, Python files directly under scripts/ and tests/,
reproduce.py, this protocol, METRICS.md, pyproject.toml and requirements-tested.txt.
No parent-directory files or result datasets are inputs to this provenance map.
Hashes describe files present at execution; regenerate results after source changes.
The driver refuses existing output paths unless --overwrite is explicitly supplied.

Not covered: T2/T3, malicious migration, replay/reordering/deletion/rollback,
long-term state reconstruction, checkpoint correctness, disk persistence,
notification delivery, runtime overhead/scalability, actual LLM behavior,
learned policies or task performance/safety. E6 separately exercises a checkpoint
component; it does not extend the evidence measured by E5.

Generated editable-install metadata (*.egg-info), bytecode and test caches are excluded from source provenance. Installation must not change the experiment source commitment. Reference result files are compared separately and are not recursively included in their own source digest.
