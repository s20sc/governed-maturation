"""One-command reproduction for the paper's E1-E6 experiments.

Runs every experiment driver against the vendored runtime, writes fresh
outputs under ``out/``, and diffs them against the reference results under
``data/``. Machine-dependent fields (wall-clock, latency samples, tarball
byte sizes) are excluded from the comparison; every other value must match
exactly, including source provenance and SHA-256 identity hashes. E6 append timings and E4 timing-derived verdicts are machine-dependent and excluded.

Usage:
    python reproduce.py            # all of E1-E6 (machine-dependent runtime)
    python reproduce.py --only e1  # one experiment
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"

E1_SEEDS = (42, 43, 44)
T_975_DF2 = 4.303  # two-sided 95% t value, df=2 (3-decimal table, as published)

# Machine-dependent keys excluded from the strict diff.
IGNORE_KEYS = {
    "wall_clock_seconds",
    "resolver_latency",
    "baseline_bundle_size_bytes",
    "bundle_size_bytes",
    "bundle_size_delta_mb",
    "bundle_size_mb",
    # E4 timing values vary run to run; structural fields still compared.
    "p50_us", "p90_us", "p99_us", "max_us", "mean_us", "ns_per_event_p50",
    "p99_at_100k_ms",
    # Verdicts derived from timing are machine-dependent too.
    "within_25ms_budget_p99", "max_window_within_budget",
    "environment",
    "append_including_signing_p50_ms", "append_including_signing_p99_ms",
}

PER_SEED_KEYS = (
    "confidence_trajectory",
    "final_skill_confidence",
    "persona_adaptation_rows",
    "reflections_all_rejected",
    "reflections_with_event",
    "resolver_latency",
    "seed",
    "wall_clock_seconds",
)

TOP_LEVEL_DROP = (
    "seed",
    "persona_adaptation_rows",
    "reflection_attempts",
    "reflections_all_rejected",
    "reflections_with_event",
    "typed_event_counts",
)


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print("  $", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in sorted(value.items()) if k not in IGNORE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in reproduction")
            elif k not in b:
                out.append(f"{path}.{k}: missing in published data")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _aggregate_e1(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-seed E1 runs into the published multi-seed structure."""
    rep = dict(runs[0])  # seed-42 representative
    for key in TOP_LEVEL_DROP:
        rep.pop(key, None)

    aggregate: dict[str, dict[str, float]] = {}
    for capability in sorted(runs[0]["final_skill_confidence"]):
        values = [r["final_skill_confidence"][capability] for r in runs]
        mean = statistics.fmean(values)
        if len(values) > 1 and statistics.stdev(values) > 0:
            half = T_975_DF2 * statistics.stdev(values) / math.sqrt(len(values))
        else:
            half = 0.0
        aggregate[capability] = {
            "ci_95_hi": round(mean + half, 6),
            "ci_95_lo": round(mean - half, 6),
            "max": max(values),
            "mean": round(mean, 6),
            "min": min(values),
        }

    hashes = {r["identity_hash_y3"] for r in runs} | {r["identity_hash_y0"] for r in runs}
    rep.update(
        {
            "seeds": [r["seed"] for r in runs],
            "per_seed": [{k: r[k] for k in PER_SEED_KEYS} for r in runs],
            "final_skill_confidence_aggregate": aggregate,
            "all_seeds_identity_byte_equal": (
                len(hashes) == 1 and all(r["identity_hash_byte_equal"] for r in runs)
            ),
        }
    )
    return rep


def reproduce_e1() -> Path:
    runs = []
    for seed in E1_SEEDS:
        out = OUT / f"e1_seed{seed}.json"
        _run("persona_experiment_e1_3year_maturation.py", "--seed", str(seed), "--output", str(out))
        runs.append(json.loads(out.read_text()))
    merged = _aggregate_e1(runs)
    target = OUT / "e1_results.json"
    target.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return target

T_975_DF9 = 2.262  # two-sided 95% t value, df=9 (3-decimal table, as published)

E2_CI_METRICS = (
    "final_skill_confidence",
    "max_drift_mitigated",
    "max_drift_unmitigated",
    "recovery_cost_clamped_events",
    "time_to_detection_failures",
)


def reproduce_e2() -> Path:
    target = OUT / "e2_results.json"
    _run("persona_experiment_e2_adversarial_drift.py", "--output", str(target))
    # The published aggregation adds 95% t-CIs over the 10-seed sweep on top
    # of the driver's min/mean/max summary; recompute them identically.
    payload = json.loads(target.read_text())
    per_seed = payload["T1_per_seed"]
    for metric in E2_CI_METRICS:
        values = [float(r[metric]) for r in per_seed]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half = T_975_DF9 * std / math.sqrt(len(values)) if std > 0 else 0.0
        payload["T1_aggregate"][metric]["ci_95_lo"] = round(mean - half, 6)
        payload["T1_aggregate"][metric]["ci_95_hi"] = round(mean + half, 6)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def reproduce_e3() -> Path:
    target = OUT / "e3_results.json"
    _run("persona_experiment_e3_cross_deployment.py", "--output", str(target))
    return target


def reproduce_e4() -> Path:
    target = OUT / "e4_results.json"
    _run("persona_experiment_e4_fold_latency.py", "--out", str(target))
    return target


def reproduce_e5() -> Path:
    with tempfile.TemporaryDirectory(prefix="e5-", dir=OUT) as tmp:
        fresh = Path(tmp) / "result.json"
        _run("persona_experiment_e5_matched_controls.py", "--output", str(fresh), "--bootstrap", "2000")
        target = OUT / "e5_results.json"
        shutil.copyfile(fresh, target)
    return target


def reproduce_e6() -> Path:
    with tempfile.TemporaryDirectory(prefix="e6-", dir=OUT) as tmp:
        fresh = Path(tmp) / "result.json"
        _run("persona_experiment_e6_recovery.py", "--output", str(fresh))
        target = OUT / "e6_results.json"
        shutil.copyfile(fresh, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e5", "e6"], default=None)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    jobs = {"e1": reproduce_e1, "e2": reproduce_e2, "e3": reproduce_e3, "e4": reproduce_e4, "e5": reproduce_e5, "e6": reproduce_e6}
    if args.only:
        jobs = {args.only: jobs[args.only]}

    failures = 0
    for name, job in jobs.items():
        print(f"[{name}] running ...")
        produced = job()
        published = DATA / f"{name}_results.json"
        delta = _diff(
            _strip(json.loads(produced.read_text())),
            _strip(json.loads(published.read_text())),
        )
        if delta:
            failures += 1
            print(f"[{name}] FAIL — {len(delta)} field(s) differ from published data:")
            for line in delta[:10]:
                print("   ", line)
        else:
            print(f"[{name}] PASS — matches published data (machine-dependent fields excluded)")

    print("=" * 60)
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} experiment(s) differ)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
