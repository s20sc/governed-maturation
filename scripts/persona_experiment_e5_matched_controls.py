"""E5 synthetic matched controls. See EXPERIMENT_PROTOCOL.md in the artifact root.

No task outcomes, learned policy, deployment, or security certification is measured.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aeros.runtime.active_dreaming import EMAValidationBackend, ProposedDelta
from aeros.runtime import persona_governance as governance
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_resolver import ResolverRejection, effective_persona

NOW = datetime(2026, 1, 1, tzinfo=UTC)
CAP = "synthetic.skill"
SEEDS = tuple(range(4200, 4230))
PROFILES = ("all", "no_authorization", "no_floor", "no_rate", "no_circuit", "no_validation")
SCENARIOS = ("T1_success_only", "T4_failure_flood", "floor_pressure", "T5_hallucination", "ceiling_pressure")
METRICS = ("final_confidence", "minimum_confidence", "final_human_threshold", "total_variation",
           "applied", "authorization_rejected", "validation_rejected", "floor", "rate_limit", "circuit_breaker")
METRIC_DEFINITIONS = {
    'final_confidence': 'Final synthetic.skill confidence; T5 proposals affect other capabilities, so this remains at the initial value there.',
    'minimum_confidence': 'Minimum of the initial synthetic.skill confidence and the affected capability confidence after each applied proposal.',
    'final_human_threshold': 'Final synthetic.action human-check threshold, with initial/default value 0.5.',
    'total_variation': 'Sum over applied proposals of absolute changes in the affected capability confidence plus the synthetic.action human-check threshold.',
    'applied': 'Number of applied evidence-driven proposals, excluding the initial-condition event.',
    'authorization_rejected': 'Number of proposals rejected by the resolver signature-presence gate before entering governance history.',
    'validation_rejected': 'Number of unsupported-capability proposals rejected by EMAValidationBackend before entering governance history.',
    'floor': 'Number of proposals blocked by the selected skill-confidence floor or the independent harness human-threshold ceiling.',
    'rate_limit': 'Number of proposals clamped by the production per-kind rate predicate after passing the circuit check.',
    'circuit_breaker': 'Number of proposals clamped by the production circuit predicate before the rate and floor checks.',
}
LIMITATIONS = [
    "Synthetic numerical state stress test; confidence/threshold proxies establish no task safety, performance, adaptation quality or deployment robustness.",
    "EMA numeric updater is an independent harness rule; both arms emit one typed proposal per evidence item, not a production continuous-learning implementation.",
    "Default effective_persona verifies signature presence only. Main traces use explicit harness authorization bypass and a presence-only sentinel for authorized human tightening; they do not measure cryptographic authenticity.",
    "Parameterized skill floors and human ceilings are harness checks; production would_violate_floor is additionally called at floor .05. Production human ceiling is declared but not enforced by would_violate_floor.",
    "EMAValidationBackend is real, but its supplied rates are synthetic empirical failure fractions, not temporal EMAs. Numeric EMA update and validation are separate factors.",
    "No emitter, SQLite persistence, circuit notification delivery, bounded-log recovery, checkpoint, migration, replay, rollback, T2 or T3 coverage.",
    "Rejected governance proposals remain in the harness history and count toward limits, matching aggregate_count; resolver-rejected/validation-rejected proposals do not enter history.",
    "One-at-a-time ablations can be masked by remaining protections; results are conditional contrasts, not independent causal contributions or optimized parameters.",
    "Bootstrap intervals quantify variation across this chosen synthetic seed distribution only; degenerate intervals are expected for deterministic gates; no multiplicity adjustment.",
]


@dataclass(frozen=True)
class Evidence:
    timestamp: datetime
    success: bool
    capability: str


def schedule(seed: int, scenario: str) -> tuple[Evidence, ...]:
    """Generate exogenous evidence once, shared unchanged by all arms/configurations."""
    rng = random.Random(seed)
    counts = {"T1_success_only": 24, "T4_failure_flood": 125, "floor_pressure": 72,
              "T5_hallucination": 16, "ceiling_pressure": 64}
    n = counts[scenario] + rng.randrange(8)
    spacing = 3600 if scenario == "floor_pressure" else 60
    return tuple(Evidence(NOW + timedelta(seconds=i * spacing), scenario == "T1_success_only",
                          f"unobserved.{rng.randrange(100000)}" if scenario == "T5_hallucination" else CAP)
                 for i in range(n))


def event(kind: str, magnitude: float, when: datetime, capability: str = CAP) -> PersonaAdaptiveEvent:
    human = kind.startswith("human_check")
    return PersonaAdaptiveEvent(kind=kind, capability=None if human else capability,
        delta={"magnitude": magnitude, **({"action_kind": "synthetic.action"} if human else {})},
        timestamp=when, operator_signature="HARNESS-PRESENCE-ONLY" if human else None)


def decide(ev: PersonaAdaptiveEvent, prior: list[PersonaAdaptiveEvent], adaptive: Any,
           profile: str, floor: float, ceiling: float) -> tuple[PersonaAdaptiveEvent, str]:
    """Real governance default path; composable real predicates for single ablations."""
    if profile == "all" and floor == .05:
        checked, reason = governance.evaluate_governance(ev, prior_events=prior,
            current_adaptive=adaptive, risk_appetite_cap=.7, now=ev.timestamp)
        if reason != "ok":
            return checked, reason
    else:
        if profile != "no_circuit" and governance.circuit_breaker_active(prior, ev.timestamp):
            return ev.model_copy(update={"applied": False}), "circuit_breaker"
        if profile != "no_rate" and governance.kind_class_at_limit(ev.kind, prior, ev.timestamp):
            return ev.model_copy(update={"applied": False}), "rate_limit"
    violates = False
    if profile != "no_floor":
        if ev.kind == "skill_confidence_decrease":
            if floor == .05:
                violates = governance.would_violate_floor(adaptive, ev, .7)
            else:
                violates = adaptive.skill_confidence.get(ev.capability, 1.) - float(ev.delta["magnitude"]) < floor
        elif ev.kind == "human_check_threshold_tighten":
            # Independent harness mechanism, NOT implemented by production floor predicate.
            violates = adaptive.human_check_threshold.get("synthetic.action", .5) + float(ev.delta["magnitude"]) > ceiling
    return (ev.model_copy(update={"applied": False}), "floor") if violates else (ev, "ok")


def run_arm(seed: int, scenario: str, mode: str, profile: str = "all",
            floor: float = .05, ceiling: float = .95) -> dict[str, Any]:
    core = PersonaCore(name="E5 synthetic", risk_appetite="balanced")
    initial = .55 + random.Random(seed + 10000).random() * .1
    initial_event = event("skill_confidence_decrease", 1. - initial, NOW - timedelta(days=2))
    accepted = [initial_event]  # Explicit initial condition, excluded from governance traffic.
    prior: list[PersonaAdaptiveEvent] = []
    adaptive = effective_persona(core, accepted).adaptive
    counts: Counter[str] = Counter()
    minimum = initial
    variation = 0.
    first_circuit = None
    trace = schedule(seed, scenario)
    for i, evidence in enumerate(trace):
        human = scenario == "ceiling_pressure"
        current = (adaptive.human_check_threshold.get("synthetic.action", .5) if human
                   else adaptive.skill_confidence.get(evidence.capability, 1.))
        target = 1. if evidence.success or human else 0.
        alpha = .008 if scenario == "T4_failure_flood" else .08
        magnitude = alpha * .6 if mode == "discrete" else alpha * abs(target - current)
        # Avoid invalid zero-magnitude events after numerical saturation, identically in both arms.
        magnitude = max(1e-12, magnitude)
        kind = ("human_check_threshold_tighten" if human else
                "skill_confidence_increase" if evidence.success else "skill_confidence_decrease")
        ev = event(kind, magnitude, evidence.timestamp, evidence.capability)
        if scenario == "T5_hallucination" and profile != "no_validation":
            proposal = ProposedDelta(target_kind=kind, target_capability=evidence.capability,
                                     proposed_delta={"magnitude": magnitude}, confidence=.9)
            # No observation exists for this capability. Decoy evidence for CAP cannot justify it.
            detail = EMAValidationBackend().validate([proposal], capability_failure_rates={CAP: 1.})[0]
            if detail.rejected:
                counts["validation_rejected"] += 1
                continue
        # Test the REAL resolver gate before counting governance traffic. In T1 the evidence
        # consists exclusively of successes; the pre-existing .55-.65 confidence is not evidence.
        try:
            effective_persona(core, [ev])
        except ResolverRejection:
            if profile != "no_authorization":
                counts["authorization_rejected"] += 1
                continue
            # Explicit experimental bypass of presence check, not a valid operator signature.
            ev = ev.model_copy(update={"operator_signature": "HARNESS-AUTHORIZATION-BYPASS"})
        ev, reason = decide(ev, prior, adaptive, profile, floor, ceiling)
        counts[reason] += 1
        if reason == "circuit_breaker" and first_circuit is None:
            first_circuit = i + 1
        prior.append(ev)
        if ev.applied:
            accepted.append(ev)
            old = adaptive.skill_confidence.get(evidence.capability, 1.)
            old_human = adaptive.human_check_threshold.get("synthetic.action", .5)
            adaptive = effective_persona(core, accepted).adaptive
            new = adaptive.skill_confidence.get(evidence.capability, 1.)
            variation += abs(new - old) + abs(adaptive.human_check_threshold.get("synthetic.action", .5) - old_human)
            counts["applied"] += 1
            minimum = min(minimum, new)
    return {"seed": seed, "scenario": scenario, "mode": mode, "profile": profile,
            "skill_floor": floor, "human_ceiling": ceiling, "initial_confidence": initial,
            "evidence_sha256": hashlib.sha256(json.dumps([(e.timestamp.isoformat(), e.success, e.capability) for e in trace]).encode()).hexdigest(),
            "attempts": len(trace), "governance_rows": len(prior),
            "first_circuit_attempt": first_circuit,
            **{key: counts[key] for key in METRICS if key in ("applied", "authorization_rejected", "validation_rejected", "floor", "rate_limit", "circuit_breaker")},
            "final_confidence": adaptive.skill_confidence.get(CAP, 1.), "minimum_confidence": minimum,
            "final_human_threshold": adaptive.human_check_threshold.get("synthetic.action", .5),
            "total_variation": variation}


def paired_ci(differences: list[float], repetitions: int = 2000) -> dict[str, Any]:
    """Percentile paired bootstrap: resample seed differences, never unpaired arms."""
    if not differences or repetitions < 100:
        raise ValueError("need nonempty differences and at least 100 bootstrap replicates")
    rng = random.Random(20261021)
    n = len(differences)
    means = sorted(statistics.fmean(rng.choices(differences, k=n)) for _ in range(repetitions))
    return {"n_pairs": n, "mean_difference": statistics.fmean(differences),
            "ci95_percentile": [means[int(.025 * (repetitions - 1))], means[int(.975 * (repetitions - 1))]]}


def signature_probe() -> dict[str, bool]:
    """Actual resolver missing-signature and Ed25519 forged-envelope checks."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    from aeros.audit.signer import AuditSigner
    from aeros.audit.chain import SignedPersonaEvent, make_chain_verifier
    raw = hashlib.sha256(b"E5 TEST KEY ONLY NEVER DEPLOY").digest()
    private = Ed25519PrivateKey.from_private_bytes(raw)
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    signer = AuditSigner(raw, public)
    core = PersonaCore()
    initial = event("skill_confidence_decrease", .4, NOW)
    unsigned = event("skill_confidence_increase", .1, NOW)
    missing_rejected = False
    try:
        effective_persona(core, [initial, unsigned])
    except ResolverRejection:
        missing_rejected = True
    signed = unsigned.model_copy(update={"operator_signature": signer.sign(unsigned).signature_hex})
    envelopes = [SignedPersonaEvent(str(i), ev, signer.sign(ev).signature_hex, signer.public_key_hex)
                 for i, ev in enumerate((initial, signed))]
    verifier = make_chain_verifier(envelopes, trusted_pubkeys={signer.public_key_hex})
    valid = effective_persona(core, [initial, signed], verifier=verifier).adaptive.skill_confidence[CAP]
    forged = SignedPersonaEvent("1", signed, "00" * 64, signer.public_key_hex)
    bad_verifier = make_chain_verifier([envelopes[0], forged], trusted_pubkeys={signer.public_key_hex})
    forged_rejected = False
    try:
        effective_persona(core, [initial, signed], verifier=bad_verifier)
    except ResolverRejection:
        forged_rejected = True
    return {"missing_operator_signature_rejected": missing_rejected,
            "forged_envelope_rejected_with_real_verifier": forged_rejected,
            "valid_signed_positive_control_applied": abs(valid - .7) < 1e-12}


def benchmark(seeds: tuple[int, ...] = SEEDS, repetitions: int = 2000) -> dict[str, Any]:
    if (len(seeds) != 1 and len(seeds) < 30) or len(set(seeds)) != len(seeds):
        raise ValueError("use one diagnostic seed or at least 30 distinct paired seeds")
    configs = [(p, .05, .95) for p in PROFILES]
    configs += [("all", f, c) for f in (.05, .1, .2) for c in (.9, .95, .99) if (f, c) != (.05, .95)]
    rows = [run_arm(s, scenario, mode, p, f, c) for p, f, c in configs
            for scenario in SCENARIOS for s in seeds for mode in ("discrete", "ema")]
    index = {(r["profile"], r["skill_floor"], r["human_ceiling"], r["scenario"], r["seed"], r["mode"]): r for r in rows}
    contrasts = []
    for p, f, c in configs:
        for scenario in SCENARIOS:
            contrasts.append({"contrast": "ema_minus_discrete", "profile": p, "floor": f, "ceiling": c,
                "scenario": scenario, "metrics": {m: paired_ci([index[p, f, c, scenario, s, "ema"][m] - index[p, f, c, scenario, s, "discrete"][m] for s in seeds], repetitions) for m in METRICS}})
    for p in PROFILES[1:]:
        for scenario in SCENARIOS:
            for mode in ("discrete", "ema"):
                contrasts.append({"contrast": "ablation_minus_all", "profile": p, "mode": mode,
                    "scenario": scenario, "floor": .05, "ceiling": .95,
                    "metrics": {m: paired_ci([index[p, .05, .95, scenario, s, mode][m] - index["all", .05, .95, scenario, s, mode][m] for s in seeds], repetitions) for m in METRICS}})
    paths = sorted({
        *(p for p in (ROOT / "src").rglob("*") if p.is_file() and p.suffix in (".py", ".txt")
          and not any(part.endswith(".egg-info") or part in ("__pycache__", ".pytest_cache") for part in p.relative_to(ROOT).parts)),
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "tests").glob("*.py"),
        *(ROOT / name for name in ("reproduce.py", "EXPERIMENT_PROTOCOL.md", "METRICS.md",
                                    "pyproject.toml", "requirements-tested.txt")),
    })
    return {"experiment": "e5_matched_controls", "schema_version": 1, "seeds": list(seeds),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": 20261021,
            "metric_definitions": dict(METRIC_DEFINITIONS),
            "signature_probe": signature_probe(), "limitations": LIMITATIONS, "provenance_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "per_seed": rows, "paired_contrasts": contrasts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "persona-e5-out/e5_matched_controls.json")
    parser.add_argument("--overwrite", action="store_true", help="explicitly permit replacing an existing output")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists: {args.output}; use another --output or explicit --overwrite")
    result = benchmark(repetitions=args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "seeds": len(SEEDS), "runs": len(result["per_seed"]), "contrasts": len(result["paired_contrasts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
