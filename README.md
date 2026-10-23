# Governed maturation

Research implementation for governed persona adaptation under a frozen
cryptographic identity. A frozen `PersonaCore` is separated from a mutable
`PersonaAdaptive` overlay. Typed adaptation events are folded by
`effective_persona`, with authorization, governance and provenance mechanisms.
A separate checkpoint store exercises bounded retention and calibration recovery.
Synthetic state and integrity checks do not establish task performance or safety.

## Install and test

Use the tested Python 3.12.13 environment for reproduction. Early Python 3.11 releases lack the tar extraction filter required by the migration driver. `requirements-tested.txt` records the tested Python 3.12.13
dependency versions; `pyproject.toml` declares supported dependency ranges.
From this artifact directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m pip install --no-deps -e .
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

## Reproduce E1–E6

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python reproduce.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/persona_experiment_e5_matched_controls.py --output /tmp/persona-e5/e5_matched_controls.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/persona_experiment_e6_recovery.py --output /tmp/persona-e6/e6_recovery.json
```

Choose fresh E5/E6 output paths on repeat runs. E5 refuses an existing path
unless `--overwrite` is explicitly supplied; E6 always refuses an existing path.
E1–E6 reproduction writes to `out/` and compares against `data/`, excluding
machine-dependent timing and bundle-size fields. `reproduce.py --only e1`
selects an individual experiment; the helper covers E1–E6; use `--only e5` or `--only e6` for the new paths.

| Experiment | Measurement | Reference dataset |
| --- | --- | --- |
| E1 | Three-year simulated maturation, three seeds; identity hash equality | `data/e1_results.json` |
| E2 | Adversarial drift, ten seeds; floor clamps and unsupported-proposal rejection | `data/e2_results.json` |
| E3 | Cross-deployment migration round-trip digests | `data/e3_results.json` |
| E4 | Resolver fold latency and structural scaling checks | `data/e4_results.json` |
| E5 | Matched discrete/EMA synthetic trajectories, ablations and threshold sensitivity | `data/e5_results.json` |
| E6 | Checkpoint calibration recovery with bounded retained tails | `data/e6_results.json` |

All six reference datasets are included in local `data/`. The experiment drivers generate outputs independently; the reproduction helper compares them to the reference results.
E5 defaults to 30 paired seeds, 4,200 runs and 120 contrast sets, with 2,000
bootstrap replicates. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for
fixed parameters, scenarios, production/harness boundaries and provenance scope,
and [METRICS.md](METRICS.md) for the ten metric definitions. One-seed imported
benchmark calls are diagnostic checks, not full experimental reproductions.
E6 uses three seeds and retention values 1/8/64, with 120 events per run.
Latency measurements depend on the host and are not portable performance guarantees.

## Layout and interpretation

- `src/aeros/runtime/`: persona models, resolver, governance, validation, event emission and checkpoint store.
- `src/aeros/audit/`: Ed25519 signing and verification helpers.
- `src/aeros/cli/`: signed migration bundle tools.
- `scripts/`: E1–E6 experiment drivers.
- `tests/`: event, resolver, matched-controls, checkpoint and standalone portability checks.
- `data/`: E1–E6 reference outputs.

Identity-tag strings in drivers are exact inputs to identity hashes and must be
preserved to reproduce those digests. Main E5 traces test signature presence;
the separate cryptographic probe checks valid and forged envelopes. Human
threshold ceilings in E5 are harness checks. E6 detects a stale bundle against a
current anchor but accepts rollback of both state and anchor; it does not deploy
an external monotonic anchor or recover rate/circuit/validation evidence state.

## License

Apache-2.0; see `LICENSE`.
