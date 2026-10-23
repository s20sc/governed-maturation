#!/usr/bin/env python3
"""E1 driver — 3-year maturation deployment trace (paper artifact).

Drives the Active Dreaming runner across an
accelerated ~1100-day deployment trace with a frozen ``IdentityManifest``
to prove Theorem 1 / the state contract invariance over a long horizon: identity
hash byte-equal at year-0 / year-1 / year-3 boundaries despite ~30
reflections, dozens of auto-emitted ``persona_adaptation`` rows, and the
growing episodic chain.

The driver:

* synthesises a per-day clock so the runner's 24h rate-limit (reflection-rate-limit)
  is honoured against simulated time, not wall-clock time;
* injects a deterministic mix of ``execution_result`` rows per day —
  one capability per "month" runs hot (failure rate ≈ 0.5) so the EMA
  validation gate accepts a tighten proposal, the rest stay at success
  baseline so the EMA ``relax`` band rejects any LLM-proposed tighten;
* runs the in-process ``LLMCounterfactualBackend`` stub once per
  reflection day; the stub returns a single tighten proposal on the
  hot capability with confidence 0.7 so a healthy reflection always
  produces an applied event;
* folds the resulting ``persona_adaptation`` chain through
  ``effective_persona`` at 10 cumulative slices to gather resolver
  latency samples for the ``persona_resolver_latency_seconds`` p99
  metric (the state contract budget < 25 ms).

Output JSON lands at ``out/e1_results.json`` with
identity_hash_y0/y1/y3, typed-event counts by kind, per-capability
confidence trajectory at year boundaries, and resolver p99 latency over
the growing event log. Last stdout line is a single-line JSON metrics
object so a CI runner can grep it.
"""

# Identity labels are fixed synthetic inputs to the manifest digest.
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aeros.audit.signer import AuditSigner
from aeros.runtime.active_dreaming import (
    DEFAULT_PROMPT_PATH,
    ENABLED_ENV,
    REFLECTION_SUMMARY_KIND,
    ActiveDreamingRunner,
    LLMCounterfactualBackend,
)
from aeros.runtime.persistent_agent import IdentityManifest, PersistentAgent
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_emit import (
    PERSONA_ADAPTATION_KIND,
    PersonaAdaptationEmitter,
)
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_metrics import reset_for_tests
from aeros.runtime.persona_resolver import effective_persona


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "out" / "e1_results.json"

IDENTITY_TAG = "persona-E1-identity"

# Capabilities the simulated agent ships. "Hot" capability rotates by
# month so the EMA failure-rate gate has something to agree with; the
# other capabilities stay at success baseline.
CAPABILITIES: tuple[str, ...] = (
    "manipulation.grasp",
    "manipulation.place",
    "navigation.move_to",
    "perception.locate_object",
)

# Episodic activity per simulated day. Small enough that 1100 days fits
# inside the agent's bounded ring buffer even without pruning.
EXECUTIONS_PER_DAY = 12
HOT_FAILURE_RATE = 0.5
COLD_FAILURE_RATE = 0.02

# Reflection cadence: one pass every ``DAYS_PER_REFLECTION`` days lands
# 10 reflections / year exactly. Honours the reflection-rate-limit 24h cap (cap=1/day
# is far below 1/36-day cadence) and matches the parent issue's "≈ 10
# reflections/year if cap honored" narrative.
DAYS_PER_REFLECTION = 36

DEFAULT_TOTAL_DAYS = 1095  # 3 × 365
QUICK_TOTAL_DAYS = 1
LATENCY_SAMPLES = 10
DEFAULT_SEED = 42


def _build_agent_and_runner(
    *,
    clock_ref: list[datetime],
    seed: int,
) -> tuple[
    PersistentAgent,
    PersonaAdaptationEmitter,
    ActiveDreamingRunner,
    PersonaCore,
]:
    manifest = IdentityManifest(
        system_prompt="persona-E1-3year-maturation",
        env_policies_hash="env",
        policy_rules_hash="pol",
        ecm_registry_hash="ecm",
        persona_hash="persona",
    )
    # 1100 days × 12 executions = 13.2k rows + ~30 reflections. Bounded
    # ring needs to hold them all so a fold over the full chain is
    # unambiguous; 16k is a safe ceiling for the experiment budget.
    agent = PersistentAgent(manifest, episodic_capacity=16384)
    rng = random.Random(seed)
    priv_obj = Ed25519PrivateKey.from_private_bytes(rng.randbytes(32))
    priv_bytes = priv_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = priv_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signer = AuditSigner(private_key_bytes=priv_bytes, public_key_bytes=pub_bytes)
    persona_core = PersonaCore(name="persona-E1", risk_appetite="balanced")
    emitter = PersonaAdaptationEmitter(
        agent,
        signer,
        failure_threshold=3,
        confidence_decrement=0.05,
        persona_core=persona_core,
    )

    def llm_call(prompt: str, deadline: float) -> tuple[str, int, int]:
        # Deterministic stub: always propose a 0.05-magnitude tighten on
        # the capability the prompt's CURRENT_CONFIDENCES section flags
        # as the highest failure rate. EMA validation gates the proposal
        # against the failure-rate band; a healthy "hot capability"
        # window passes, a cold window is rejected.
        target = _pick_hot_capability(prompt)
        body = json.dumps(
            {
                "reflection_text": (
                    f"Maturation reflection — proposing tighten on "
                    f"{target} based on observed failure cluster."
                ),
                "proposed_deltas": [
                    {
                        "target_kind": "skill_confidence_decrease",
                        "target_capability": target,
                        "proposed_delta": {"magnitude": 0.05},
                        "confidence": 0.7,
                    }
                ],
                "derived_facts": [
                    f"capability={target} drift_window=last_36_days"
                ],
            },
            sort_keys=True,
        )
        return body, len(prompt), len(body)

    backend = LLMCounterfactualBackend(
        llm_call, prompt_template_path=DEFAULT_PROMPT_PATH
    )
    runner = ActiveDreamingRunner(
        agent,
        signer,
        emitter,
        backend,
        episodic_window=200,
        clock=lambda: clock_ref[0],
    )
    return agent, emitter, runner, persona_core


def _pick_hot_capability(prompt: str) -> str:
    """Parse the prompt's CURRENT_CONFIDENCES block for the hottest cap.

    Falls back to ``CAPABILITIES[0]`` when the prompt has no failure
    signal — the EMA gate then rejects the tighten because the cold
    capability's failure rate sits in the relax band, which is exactly
    the "noisy LLM proposes nonsense; EMA catches it" path we want
    the paper to be able to point at.
    """
    best: tuple[float, str] = (-1.0, CAPABILITIES[0])
    for line in prompt.splitlines():
        if "failure_rate=" not in line or ":" not in line:
            continue
        try:
            cap_part, rate_part = line.split(":", 1)
            cap = cap_part.strip().lstrip("- ").strip()
            rate = float(rate_part.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if rate > best[0]:
            best = (rate, cap)
    return best[1]


def _seed_day_episodes(
    agent: PersistentAgent,
    *,
    rng: random.Random,
    hot_capability: str,
) -> None:
    for _ in range(EXECUTIONS_PER_DAY):
        cap = rng.choice(CAPABILITIES)
        if cap == hot_capability:
            outcome = "failure" if rng.random() < HOT_FAILURE_RATE else "success"
        else:
            outcome = "failure" if rng.random() < COLD_FAILURE_RATE else "success"
        agent.record(
            kind="execution_result",
            summary=f"{cap}:{outcome}",
            payload={"capability": cap, "outcome": outcome},
        )


def _typed_event_counts(agent: PersistentAgent) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in agent.episodic.by_kind(PERSONA_ADAPTATION_KIND):
        try:
            ev = PersonaAdaptiveEvent(**row.payload["event"])
        except (TypeError, KeyError, ValueError):
            continue
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
    return counts


def _persona_events(agent: PersistentAgent) -> list[PersonaAdaptiveEvent]:
    out: list[PersonaAdaptiveEvent] = []
    for row in agent.episodic.by_kind(PERSONA_ADAPTATION_KIND):
        try:
            out.append(PersonaAdaptiveEvent(**row.payload["event"]))
        except (TypeError, KeyError, ValueError):
            continue
    return out


def _sample_resolver_latency(
    persona_core: PersonaCore,
    events: list[PersonaAdaptiveEvent],
) -> dict[str, float]:
    """Time the resolver fold across LATENCY_SAMPLES cumulative slices.

    Uses ``time.perf_counter_ns`` rather than the registry histogram so
    the per-slice latency lands in the JSON regardless of registry
    state, while still leaving the registry's ``persona_resolver_latency_seconds``
    quantiles populated for cross-check.
    """
    samples: list[float] = []
    if not events:
        # Fold once over an empty event list so the resolver still runs
        # — keeps the metric non-empty and the JSON shape stable.
        t0 = time.perf_counter_ns()
        effective_persona(persona_core, ())
        elapsed_ns = time.perf_counter_ns() - t0
        samples.append(elapsed_ns / 1e9)
    else:
        step = max(1, len(events) // LATENCY_SAMPLES)
        for i in range(step, len(events) + 1, step):
            slice_events = events[:i]
            t0 = time.perf_counter_ns()
            effective_persona(persona_core, slice_events)
            elapsed_ns = time.perf_counter_ns() - t0
            samples.append(elapsed_ns / 1e9)

    samples.sort()

    def _q(q: float) -> float:
        if not samples:
            return 0.0
        idx = min(len(samples) - 1, int(q * (len(samples) - 1)))
        return samples[idx]

    return {
        "p50_seconds": round(_q(0.50), 6),
        "p90_seconds": round(_q(0.90), 6),
        "p99_seconds": round(_q(0.99), 6),
        "max_seconds": round(samples[-1], 6) if samples else 0.0,
        "samples": [round(s, 6) for s in samples],
    }


def _confidence_trajectory(
    persona_core: PersonaCore,
    events: list[PersonaAdaptiveEvent],
    *,
    snapshots: int = 6,
) -> list[dict[str, Any]]:
    """Per-capability confidence at evenly-spaced event indices.

    Figure data: each entry is ``{event_index, by_capability:
    {cap: confidence}}``. The first entry is index 0 (PersonaCore
    defaults) and the last is the full chain.
    """
    if not events:
        profile = effective_persona(persona_core, ())
        return [
            {
                "event_index": 0,
                "by_capability": dict(profile.adaptive.skill_confidence),
            }
        ]
    step = max(1, len(events) // (snapshots - 1))
    indices = list(range(0, len(events), step))
    if indices[-1] != len(events):
        indices.append(len(events))
    out: list[dict[str, Any]] = []
    for idx in indices:
        profile = effective_persona(persona_core, events[:idx])
        out.append(
            {
                "event_index": idx,
                "by_capability": {
                    cap: round(profile.adaptive.skill_confidence.get(cap, 1.0), 6)
                    for cap in CAPABILITIES
                },
            }
        )
    return out


def _hot_capability_for_day(day_idx: int) -> str:
    """Rotate the failing capability through the registry by month.

    With CAPABILITIES of length 4 and DAYS_PER_REFLECTION = 36, every
    other reflection lands on a freshly-rotated hot capability. This
    keeps each ``skill_confidence_decrease`` event tied to a real EMA
    failure-rate trend so the gate accepts roughly 25-30 of the 30 paper
    reflections — the rest get rejected when the rotation crosses a
    reflection boundary, which is exactly the "EMA caught a stale-window
    proposal" pattern.
    """
    return CAPABILITIES[(day_idx // 30) % len(CAPABILITIES)]


def _run_day_loop(
    *,
    agent: PersistentAgent,
    runner: ActiveDreamingRunner,
    clock_ref: list[datetime],
    rng: random.Random,
    total_days: int,
    year_boundaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    reflection_attempts = 0
    reflections_with_event = 0
    reflections_all_rejected = 0

    for day in range(total_days):
        clock_ref[0] = start + timedelta(days=day, hours=2)  # off-peak (02:00)
        hot = _hot_capability_for_day(day)
        _seed_day_episodes(agent, rng=rng, hot_capability=hot)

        if day % DAYS_PER_REFLECTION == 0 and day > 0:
            reflection_attempts += 1
            event = runner.run_reflection(agent.identity_hash)
            if event is None:
                continue
            if event.validation_result == "all_rejected":
                reflections_all_rejected += 1
            else:
                reflections_with_event += 1

        if day + 1 in year_boundaries:
            year_boundaries[day + 1]["identity_hash"] = agent.identity_hash
            year_boundaries[day + 1]["typed_event_counts"] = _typed_event_counts(
                agent
            )

    return {
        "reflection_attempts": reflection_attempts,
        "reflections_with_event": reflections_with_event,
        "reflections_all_rejected": reflections_all_rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E1 driver: 3-year maturation trace "
            "(Active Dreaming)."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a 1-day smoke pass instead of the full 3-year trace",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="deterministic seed for episodic activity (default 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="path to JSON metrics file (default out/e1_results.json)",
    )
    args = parser.parse_args(argv)

    started_at = time.monotonic()
    os.environ[ENABLED_ENV] = "true"
    reset_for_tests()
    rng = random.Random(args.seed)

    clock_ref: list[datetime] = [datetime(2026, 1, 1, tzinfo=UTC)]
    agent, _emitter, runner, persona_core = _build_agent_and_runner(
        clock_ref=clock_ref, seed=args.seed
    )

    identity_hash_y0 = agent.identity_hash

    total_days = QUICK_TOTAL_DAYS if args.quick else DEFAULT_TOTAL_DAYS
    year_boundaries: dict[int, dict[str, Any]] = {
        365: {"label": "y1"},
        1095: {"label": "y3"},
    }
    if args.quick:
        # In quick mode the 365 / 1095 boundaries never trigger; capture
        # the post-loop state under both labels so the byte-equal
        # assertions still cover the schema-shape contract.
        year_boundaries = {total_days: {"label": "quick"}}

    loop_stats = _run_day_loop(
        agent=agent,
        runner=runner,
        clock_ref=clock_ref,
        rng=rng,
        total_days=total_days,
        year_boundaries=year_boundaries,
    )

    final_hash = agent.identity_hash
    if not args.quick:
        identity_hash_y1 = year_boundaries[365].get("identity_hash", final_hash)
        identity_hash_y3 = year_boundaries[1095].get("identity_hash", final_hash)
    else:
        identity_hash_y1 = identity_hash_y3 = final_hash

    # Theorem-1 invariance: the manifest is frozen, so the hash MUST be
    # identical at every snapshot. Asserting it makes a regression
    # impossible to ship without flipping a test red.
    assert identity_hash_y0 == identity_hash_y1 == identity_hash_y3, (
        "identity_hash MUST be byte-equal across the maturation trace; got "
        f"y0={identity_hash_y0!r} y1={identity_hash_y1!r} y3={identity_hash_y3!r}"
    )

    events = _persona_events(agent)
    typed_event_counts = _typed_event_counts(agent)
    reflection_summary_rows = len(agent.episodic.by_kind(REFLECTION_SUMMARY_KIND))
    confidence_trajectory = _confidence_trajectory(persona_core, events)
    resolver_latency = _sample_resolver_latency(persona_core, events)

    final_profile = effective_persona(persona_core, events)
    final_confidences = {
        cap: round(final_profile.adaptive.skill_confidence.get(cap, 1.0), 6)
        for cap in CAPABILITIES
    }

    metrics = {
        "experiment": "e1_3year_maturation",
        "seed": args.seed,
        "quick_mode": args.quick,
        "total_days": total_days,
        "identity_hash_y0": identity_hash_y0,
        "identity_hash_y1": identity_hash_y1,
        "identity_hash_y3": identity_hash_y3,
        "identity_hash_byte_equal": (
            identity_hash_y0 == identity_hash_y1 == identity_hash_y3
        ),
        "reflection_summary_rows": reflection_summary_rows,
        "reflection_attempts": loop_stats["reflection_attempts"],
        "reflections_with_event": loop_stats["reflections_with_event"],
        "reflections_all_rejected": loop_stats["reflections_all_rejected"],
        "typed_event_counts": typed_event_counts,
        "persona_adaptation_rows": sum(typed_event_counts.values()),
        "confidence_trajectory": confidence_trajectory,
        "final_skill_confidence": final_confidences,
        "resolver_latency": resolver_latency,
        "wall_clock_seconds": round(time.monotonic() - started_at, 4),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
