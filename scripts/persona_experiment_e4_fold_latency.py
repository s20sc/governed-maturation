"""E4 — fold-latency scaling sweep for the persona resolver (paper artifact, experiment E4).

Measures how ``effective_persona`` fold latency grows with the size of the
typed-event window, from the ~20-row logs exercised by E1 up to the 100k-row
episodic capacity bound. The sweep compares measured p99 latency with a
25 ms reference budget for the read-time fold; storage and signature costs
are outside this microbenchmark.

Pure in-memory micro-benchmark: no episodic.db involved (the fold consumes an
already-loaded event sequence; storage read time is a separate concern).

Usage:
    PYTHONPATH=src python scripts/persona_experiment_e4_fold_latency.py \
        --out /path/to/e4_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_resolver import effective_persona

logger = logging.getLogger(__name__)

SEED = 42
SIZES = (100, 1_000, 10_000, 100_000)
N_REPS = 100
N_WARMUP = 3
RUNTIME_BUDGET_MS = 25.0

CAPABILITIES = (
    "manipulation.grasp",
    "manipulation.place",
    "navigation.move_to",
    "perception.locate_object",
)

ACTION_KINDS = ("manipulation", "navigation", "perception")

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_BENCH_SIGNATURE = "bench-operator-signature-placeholder"


def _make_event(rng: random.Random, index: int) -> PersonaAdaptiveEvent:
    """Build one valid typed event; ~70% auto-tighten, ~30% signed relax.

    Mirrors the kind mix a long-lived deployment accumulates: tighten events
    dominate (auto-emitted), relax events are rarer and operator-signed.
    """
    ts = _BASE_TS + timedelta(seconds=index)
    cause_ids = (f"ep-{index}",)
    roll = rng.random()

    if roll < 0.40:
        return PersonaAdaptiveEvent(
            kind="skill_confidence_decrease",
            capability=rng.choice(CAPABILITIES),
            delta={"magnitude": 0.01},
            triggering_cause_ids=cause_ids,
            timestamp=ts,
        )
    if roll < 0.55:
        return PersonaAdaptiveEvent(
            kind="drift_event",
            delta={"severity": rng.choice(("low", "medium", "high")), "reason": "bench"},
            triggering_cause_ids=cause_ids,
            timestamp=ts,
        )
    if roll < 0.70:
        return PersonaAdaptiveEvent(
            kind="risk_appetite_calibration_tighten",
            delta={"magnitude": 0.01},
            triggering_cause_ids=cause_ids,
            timestamp=ts,
        )
    if roll < 0.85:
        return PersonaAdaptiveEvent(
            kind="skill_confidence_increase",
            capability=rng.choice(CAPABILITIES),
            delta={"magnitude": 0.01},
            triggering_cause_ids=cause_ids,
            operator_signature=_BENCH_SIGNATURE,
            timestamp=ts,
        )
    if roll < 0.95:
        return PersonaAdaptiveEvent(
            kind="human_check_threshold_relax",
            delta={"action_kind": rng.choice(ACTION_KINDS), "magnitude": 0.01},
            triggering_cause_ids=cause_ids,
            operator_signature=_BENCH_SIGNATURE,
            timestamp=ts,
        )
    return PersonaAdaptiveEvent(
        kind="risk_appetite_calibration_relax",
        delta={"magnitude": 0.01},
        triggering_cause_ids=cause_ids,
        operator_signature=_BENCH_SIGNATURE,
        timestamp=ts,
    )


def _synthesize(size: int) -> list[PersonaAdaptiveEvent]:
    """Deterministically synthesize ``size`` valid events (seeded by size)."""
    rng = random.Random(SEED + size)
    return [_make_event(rng, i) for i in range(size)]


def _percentile(sorted_ns: list[int], pct: float) -> int:
    idx = min(len(sorted_ns) - 1, max(0, round(pct / 100.0 * (len(sorted_ns) - 1))))
    return sorted_ns[idx]


def _bench_size(core: PersonaCore, size: int) -> dict[str, Any]:
    events = _synthesize(size)
    for _ in range(N_WARMUP):
        effective_persona(core, events)

    samples_ns: list[int] = []
    for _ in range(N_REPS):
        t0 = time.perf_counter_ns()
        effective_persona(core, events)
        samples_ns.append(time.perf_counter_ns() - t0)
    samples_ns.sort()

    p50, p90, p99 = (_percentile(samples_ns, p) for p in (50, 90, 99))
    result = {
        "window_size": size,
        "reps": N_REPS,
        "p50_us": p50 / 1e3,
        "p90_us": p90 / 1e3,
        "p99_us": p99 / 1e3,
        "max_us": samples_ns[-1] / 1e3,
        "mean_us": statistics.fmean(samples_ns) / 1e3,
        "ns_per_event_p50": p50 / size,
        "within_25ms_budget_p99": (p99 / 1e6) < RUNTIME_BUDGET_MS,
    }
    logger.info(
        "size=%6d  p50=%9.1f us  p99=%9.1f us  per-event=%6.0f ns  budget(25ms)=%s",
        size, result["p50_us"], result["p99_us"],
        result["ns_per_event_p50"], "OK" if result["within_25ms_budget_p99"] else "EXCEEDED",
    )
    return result


def _environment() -> dict[str, str]:
    cpu = platform.processor() or ""
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip() or cpu
        except OSError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(SEED)

    core = PersonaCore()
    sweep = [_bench_size(core, size) for size in SIZES]

    payload = {
        "experiment": "e4_fold_latency_scaling",
        "seed": SEED,
        "config": {
            "sizes": list(SIZES),
            "reps": N_REPS,
            "warmup": N_WARMUP,
            "runtime_budget_ms": RUNTIME_BUDGET_MS,
            "kind_mix": "40% conf-decrease / 15% drift / 15% risk-tighten / "
                        "15% signed conf-increase / 10% signed hct-relax / 5% signed risk-relax",
        },
        "environment": _environment(),
        "sweep": sweep,
        "headline": {
            "max_window_within_budget": max(
                (r["window_size"] for r in sweep if r["within_25ms_budget_p99"]),
                default=0,
            ),
            "p99_at_100k_ms": next(
                (r["p99_us"] / 1e3 for r in sweep if r["window_size"] == 100_000), None
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
