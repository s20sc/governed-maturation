"""persona_resolver — fold + signature-policy / chain-verification gating."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aeros.runtime.persona import PersonaProfile
from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_resolver import (
    ResolverRejection,
    effective_persona,
)


_NOW = datetime(2026, 5, 4, 0, 0, 0, tzinfo=UTC)


def _evt(**kwargs: object) -> PersonaAdaptiveEvent:
    base: dict[str, object] = {
        "kind": "skill_confidence_decrease",
        "capability": "manipulation.grasp",
        "delta": {"magnitude": 0.1},
        "triggering_cause_ids": ["c1"],
        "operator_signature": None,
        "timestamp": _NOW,
    }
    base.update(kwargs)
    return PersonaAdaptiveEvent(**base)  # type: ignore[arg-type]


def test_empty_events_yields_core_lift() -> None:
    core = PersonaCore(name="x", risk_appetite="balanced")
    profile = effective_persona(core, ())
    assert profile == PersonaProfile.from_core(core)
    assert profile.adaptive == PersonaAdaptive()


def test_skill_confidence_decrease_folds() -> None:
    core = PersonaCore()
    e = _evt()  # decrease grasp by 0.1
    profile = effective_persona(core, [e])
    assert profile.adaptive.skill_confidence == {"manipulation.grasp": 0.9}


def test_skill_confidence_decrease_chain_clamps_at_zero() -> None:
    core = PersonaCore()
    events = [_evt(delta={"magnitude": 0.5}) for _ in range(5)]
    profile = effective_persona(core, events)
    assert profile.adaptive.skill_confidence["manipulation.grasp"] == 0.0


def test_skill_confidence_increase_requires_signature() -> None:
    core = PersonaCore()
    e = _evt(kind="skill_confidence_increase")
    with pytest.raises(ResolverRejection):
        effective_persona(core, [e])
    e_signed = _evt(kind="skill_confidence_increase", operator_signature="sig:test")
    profile = effective_persona(core, [e_signed])
    assert profile.adaptive.skill_confidence == {"manipulation.grasp": 1.0}


def test_human_check_threshold_relax_requires_signature() -> None:
    core = PersonaCore()
    e_unsigned = _evt(
        kind="human_check_threshold_relax",
        capability=None,
        delta={"action_kind": "grasp", "magnitude": 0.1},
    )
    with pytest.raises(ResolverRejection):
        effective_persona(core, [e_unsigned])


def test_human_check_threshold_tighten_also_requires_signature() -> None:
    core = PersonaCore()
    e_unsigned = _evt(
        kind="human_check_threshold_tighten",
        capability=None,
        delta={"action_kind": "grasp", "magnitude": 0.1},
    )
    with pytest.raises(ResolverRejection):
        effective_persona(core, [e_unsigned])


def test_risk_appetite_tighten_no_signature_required() -> None:
    core = PersonaCore(risk_appetite="balanced")
    e = _evt(
        kind="risk_appetite_calibration_tighten",
        capability=None,
        delta={"magnitude": 0.1},
    )
    profile = effective_persona(core, [e])
    assert profile.adaptive.risk_appetite_actual == pytest.approx(0.4)


def test_risk_appetite_relax_clamps_to_cap() -> None:
    core = PersonaCore(risk_appetite="conservative")  # cap = 0.4
    e = _evt(
        kind="risk_appetite_calibration_relax",
        capability=None,
        delta={"magnitude": 0.5},
        operator_signature="sig:test",
    )
    profile = effective_persona(core, [e])
    assert profile.adaptive.risk_appetite_actual == pytest.approx(0.4)


def test_drift_event_nudges_risk_and_records_history() -> None:
    core = PersonaCore(risk_appetite="balanced")
    e = _evt(
        kind="drift_event",
        capability=None,
        delta={"severity": "medium", "reason": "emergency_stop"},
    )
    profile = effective_persona(core, [e])
    assert profile.adaptive.risk_appetite_actual == pytest.approx(0.4)
    history = profile.adaptive.tendencies["drift_history"]
    assert history == [{"severity": "medium", "reason": "emergency_stop"}]


def test_drift_history_bounded_to_eight() -> None:
    core = PersonaCore()
    events = [
        _evt(
            kind="drift_event",
            capability=None,
            delta={"severity": "low", "reason": f"r{i}"},
        )
        for i in range(12)
    ]
    profile = effective_persona(core, events)
    history = profile.adaptive.tendencies["drift_history"]
    assert len(history) == 8
    assert history[0]["reason"] == "r4"
    assert history[-1]["reason"] == "r11"


def test_applied_false_event_skipped_but_does_not_reject() -> None:
    """Q5: floor-clamped events live in the chain but the fold ignores them."""
    core = PersonaCore()
    e_applied = _evt(delta={"magnitude": 0.1})
    e_skipped = _evt(delta={"magnitude": 0.4}, applied=False)
    profile = effective_persona(core, [e_applied, e_skipped])
    assert profile.adaptive.skill_confidence == {"manipulation.grasp": 0.9}


def test_verifier_can_reject_event_chain() -> None:
    """chain-verification: verifier returning False triggers ResolverRejection."""
    core = PersonaCore()
    e = _evt()

    def reject_all(_e: PersonaAdaptiveEvent) -> bool:
        return False

    with pytest.raises(ResolverRejection):
        effective_persona(core, [e], verifier=reject_all)


def test_resolver_is_pure() -> None:
    """Same input → same output, no shared state across calls."""
    core = PersonaCore()
    e1 = _evt(delta={"magnitude": 0.1})
    e2 = _evt(delta={"magnitude": 0.2})

    a = effective_persona(core, [e1, e2])
    b = effective_persona(core, [e1, e2])
    assert a.adaptive == b.adaptive
    assert a.adaptive is not b.adaptive

    # Running the second fold without e2 must not show drift from the
    # earlier call: no module-level state.
    c = effective_persona(core, [e1])
    assert c.adaptive.skill_confidence["manipulation.grasp"] == pytest.approx(0.9)


def test_persona_profile_to_manifest_payload_excludes_adaptive() -> None:
    """identity-invariance: adaptive fields are NOT in the hash input."""
    core = PersonaCore(name="hr1")
    e = _evt(delta={"magnitude": 0.4})  # would otherwise show in adaptive
    drifted = effective_persona(core, [e])
    fresh = effective_persona(core, ())
    assert drifted.to_manifest_payload() == fresh.to_manifest_payload()
