"""PersonaAdaptiveEvent (the state contract) — model validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aeros.runtime.persona_event import (
    ALWAYS_SIGNED_TIGHTEN_KINDS,
    AUTOMATIC_TIGHTEN_KINDS,
    CURRENT_SCHEMA_VERSION,
    RELAX_KINDS,
    SIGNATURE_REQUIRED_KINDS,
    PersonaAdaptiveEvent,
    is_automatic_tighten_kind,
    is_signature_required,
)


_NOW = datetime(2026, 5, 4, 0, 0, 0, tzinfo=UTC)


def _evt(**kwargs: object) -> PersonaAdaptiveEvent:
    base: dict[str, object] = {
        "kind": "skill_confidence_decrease",
        "capability": "manipulation.grasp",
        "delta": {"magnitude": 0.1},
        "triggering_cause_ids": ["cause-1"],
        "operator_signature": None,
        "timestamp": _NOW,
    }
    base.update(kwargs)
    return PersonaAdaptiveEvent(**base)  # type: ignore[arg-type]


def test_kind_classification_partition() -> None:
    """The three kind sets partition the 7 the state contract kinds with no overlap."""
    all_kinds = {
        "skill_confidence_decrease",
        "drift_event",
        "risk_appetite_calibration_tighten",
        "skill_confidence_increase",
        "human_check_threshold_relax",
        "risk_appetite_calibration_relax",
        "human_check_threshold_tighten",
    }
    union = AUTOMATIC_TIGHTEN_KINDS | RELAX_KINDS | ALWAYS_SIGNED_TIGHTEN_KINDS
    assert union == all_kinds
    assert AUTOMATIC_TIGHTEN_KINDS.isdisjoint(RELAX_KINDS)
    assert AUTOMATIC_TIGHTEN_KINDS.isdisjoint(ALWAYS_SIGNED_TIGHTEN_KINDS)
    assert RELAX_KINDS.isdisjoint(ALWAYS_SIGNED_TIGHTEN_KINDS)
    assert SIGNATURE_REQUIRED_KINDS == RELAX_KINDS | ALWAYS_SIGNED_TIGHTEN_KINDS


@pytest.mark.parametrize("kind", sorted(AUTOMATIC_TIGHTEN_KINDS))
def test_automatic_tighten_kinds_do_not_require_signature(kind: str) -> None:
    assert not is_signature_required(kind)
    assert is_automatic_tighten_kind(kind)


@pytest.mark.parametrize("kind", sorted(SIGNATURE_REQUIRED_KINDS))
def test_relax_and_always_signed_kinds_require_signature(kind: str) -> None:
    assert is_signature_required(kind)
    assert not is_automatic_tighten_kind(kind)


def test_default_schema_version_and_applied() -> None:
    e = _evt()
    assert e.schema_version == CURRENT_SCHEMA_VERSION
    assert e.applied is True


def test_skill_kinds_require_capability() -> None:
    with pytest.raises(ValidationError):
        _evt(capability=None, kind="skill_confidence_decrease")
    with pytest.raises(ValidationError):
        _evt(capability="", kind="skill_confidence_decrease")


def test_skill_kinds_validate_magnitude() -> None:
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": 0.0})
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": -0.1})
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": 0.51})
    # Boundaries: just-positive and exactly the cap are accepted.
    _evt(delta={"magnitude": 0.001})
    _evt(delta={"magnitude": 0.5})


def test_risk_appetite_kinds_validate_magnitude() -> None:
    _evt(
        kind="risk_appetite_calibration_tighten",
        capability=None,
        delta={"magnitude": 0.2},
    )
    with pytest.raises(ValidationError):
        _evt(
            kind="risk_appetite_calibration_tighten",
            capability=None,
            delta={"magnitude": 0.6},
        )
    with pytest.raises(ValidationError):
        _evt(
            kind="risk_appetite_calibration_tighten",
            capability=None,
            delta={},
        )


def test_human_check_threshold_kinds_validate_action_kind_and_magnitude() -> None:
    _evt(
        kind="human_check_threshold_relax",
        capability=None,
        delta={"action_kind": "manipulation.grasp", "magnitude": 0.1},
    )
    with pytest.raises(ValidationError):
        _evt(
            kind="human_check_threshold_relax",
            capability=None,
            delta={"magnitude": 0.1},
        )
    with pytest.raises(ValidationError):
        _evt(
            kind="human_check_threshold_relax",
            capability=None,
            delta={"action_kind": "", "magnitude": 0.1},
        )


def test_drift_event_validates_severity_and_reason() -> None:
    _evt(
        kind="drift_event",
        capability=None,
        delta={"severity": "high", "reason": "emergency_stop"},
    )
    with pytest.raises(ValidationError):
        _evt(
            kind="drift_event",
            capability=None,
            delta={"severity": "catastrophic", "reason": "anything"},
        )
    with pytest.raises(ValidationError):
        _evt(
            kind="drift_event",
            capability=None,
            delta={"severity": "high"},
        )


def test_event_is_frozen_and_extra_forbidden() -> None:
    e = _evt()
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        e.kind = "drift_event"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        PersonaAdaptiveEvent(
            kind="skill_confidence_decrease",
            capability="x",
            delta={"magnitude": 0.1},
            triggering_cause_ids=[],
            operator_signature=None,
            timestamp=_NOW,
            extra_field="injection",  # type: ignore[call-arg]
        )


def test_delta_is_deeply_immutable_after_construction() -> None:
    """Nested ``delta`` dict must reject post-construction mutation so a
    caller cannot bypass per-kind validation by mutating after validation.
    """
    e = _evt(delta={"magnitude": 0.1})
    with pytest.raises(TypeError):
        e.delta["magnitude"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        e.delta["new_key"] = "x"  # type: ignore[index]
    # Shape preserved + read access still works.
    assert e.delta["magnitude"] == pytest.approx(0.1)
    assert dict(e.delta) == {"magnitude": 0.1}


def test_delta_input_dict_aliasing_does_not_leak() -> None:
    """A caller mutating the dict they passed in must not affect the event."""
    src = {"magnitude": 0.1}
    e = _evt(delta=src)
    src["magnitude"] = 999
    src["injected"] = "evil"
    assert e.delta["magnitude"] == pytest.approx(0.1)
    assert "injected" not in e.delta


def test_triggering_cause_ids_is_immutable_tuple() -> None:
    """``triggering_cause_ids`` must be a tuple — list mutation methods absent."""
    e = _evt(triggering_cause_ids=["c1", "c2"])
    assert isinstance(e.triggering_cause_ids, tuple)
    assert e.triggering_cause_ids == ("c1", "c2")
    # Tuples have no append; AttributeError on .append is the contract.
    with pytest.raises(AttributeError):
        e.triggering_cause_ids.append("c3")  # type: ignore[attr-defined]


def test_triggering_cause_ids_input_list_aliasing_does_not_leak() -> None:
    """Mutating the list passed in must not change the event's tuple."""
    src = ["c1", "c2"]
    e = _evt(triggering_cause_ids=src)
    src.append("evil")
    src[0] = "tampered"
    assert e.triggering_cause_ids == ("c1", "c2")


def test_extra_delta_keys_rejected() -> None:
    """Per-kind delta schemas are closed: any key outside the allowed set
    must fail validation, so an attacker can't smuggle audit-shaped fields
    into the canonical signing surface alongside the legitimate ones.
    """
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": 0.1, "extra_unknown": "x"})
    with pytest.raises(ValidationError):
        _evt(
            kind="drift_event",
            capability=None,
            delta={"severity": "low", "reason": "x", "extra": []},
        )
    with pytest.raises(ValidationError):
        _evt(
            kind="risk_appetite_calibration_tighten",
            capability=None,
            delta={"magnitude": 0.1, "audit": []},
        )


def test_delta_rejects_non_json_safe_types() -> None:
    """Delta values must be JSON-safe primitives or nested dict/list of
    the same — sets, bytes, and arbitrary objects are rejected so the
    canonical signing path (json.dumps) can never fail at sign time.
    """
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": {0.1, 0.2}})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _evt(delta={"magnitude": b"\x01"})  # type: ignore[arg-type]


def test_delta_nested_mutation_blocked_for_future_compound_shapes() -> None:
    """Defense-in-depth for kinds that may later allow nested containers:
    the deep-freeze helper must convert lists to tuples and dicts to
    MappingProxyType all the way down, so even if a future kind's schema
    permits a nested container, post-construction mutation is impossible.

    This is a unit test of the helper, since today's per-kind schemas are
    flat (extras-key rejection alone blocks the surface). The helper is
    exercised here so the property is regression-locked when the schemas
    grow.
    """
    from aeros.runtime.persona_event import _deep_freeze

    frozen = _deep_freeze({"a": [1, [2, 3]], "b": {"c": [4]}})
    # Nested list became tuple-of-tuples; nested dict became MappingProxyType.
    assert frozen["a"] == (1, (2, 3))
    assert frozen["b"]["c"] == (4,)
    with pytest.raises(TypeError):
        frozen["a"] = "tampered"  # type: ignore[index]
    with pytest.raises(AttributeError):
        frozen["a"].append(999)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        frozen["b"]["c"].append(999)  # type: ignore[attr-defined]


def test_canonical_payload_byte_equal_across_mutation_attempts() -> None:
    """Audit chain must see the same canonical bytes whether or not a
    caller has tried to mutate the event post-construction.
    """
    pytest.importorskip("aeros.audit.signer")
    from aeros.audit.signer import canonical_event_payload

    e = _evt(delta={"magnitude": 0.1})
    before = canonical_event_payload(e)

    with pytest.raises((TypeError, AttributeError)):
        e.delta["magnitude"] = 999  # type: ignore[index]
    with pytest.raises(AttributeError):
        e.triggering_cause_ids.append("evil")  # type: ignore[attr-defined]

    after = canonical_event_payload(e)
    assert before == after


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        PersonaAdaptiveEvent(
            kind="not_a_real_kind",  # type: ignore[arg-type]
            capability=None,
            delta={},
            triggering_cause_ids=[],
            operator_signature=None,
            timestamp=_NOW,
        )
