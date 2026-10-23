"""PersonaAdaptiveEvent — typed shifts in PersonaAdaptive (the state contract).

The only legal mutation of ``PersonaAdaptive`` is via a typed event
recorded as an ``episodic.db`` row. Each event names a ``kind`` from a
fixed enum, an optional ``capability`` it scopes to, a per-kind ``delta``
shape, the ``triggering_cause_ids`` chain that justifies it, an optional
``operator_signature`` (required for relax-direction kinds per the state contract),
a ``timestamp`` and a ``schema_version``.

The ``applied`` flag is the cap point for safety-floor activations
(event application flag). When an automatic-tighten event
would push past a resolver safety floor (the state contract), the event is still
written to episodic.db (typed-event recording: every adaptation is a typed audit event)
but with ``applied=False``, and the resolver fold skips it. Operators
review ``applied=False`` rows via ``aeros agent review`` and
can manually counter-sign + flip them.

This module ships:
- the model
- the kind classification helpers (signature requirement + automatic
  emit eligibility) shared by the resolver and downstream subs
- per-kind ``delta`` shape validation

Auto-emit, Ed25519 cause-chain signing, CLIs, safety floor + rate limit, and metrics + HTTP all import from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

EventKind = Literal[
    "skill_confidence_decrease",
    "drift_event",
    "risk_appetite_calibration_tighten",
    "skill_confidence_increase",
    "human_check_threshold_relax",
    "risk_appetite_calibration_relax",
    "human_check_threshold_tighten",
]

# the state contract asymmetric trust. Tighten direction: automatic emit is allowed
# (no operator signature needed). Relax direction: rejected unless
# operator-signed. ``human_check_threshold_tighten`` is a tighten in
# direction but is conventionally a deliberate operator action after an
# incident, so it is in the always-signed set.
AUTOMATIC_TIGHTEN_KINDS: frozenset[str] = frozenset(
    {
        "skill_confidence_decrease",
        "drift_event",
        "risk_appetite_calibration_tighten",
    }
)
RELAX_KINDS: frozenset[str] = frozenset(
    {
        "skill_confidence_increase",
        "human_check_threshold_relax",
        "risk_appetite_calibration_relax",
    }
)
ALWAYS_SIGNED_TIGHTEN_KINDS: frozenset[str] = frozenset(
    {"human_check_threshold_tighten"}
)
SIGNATURE_REQUIRED_KINDS: frozenset[str] = RELAX_KINDS | ALWAYS_SIGNED_TIGHTEN_KINDS

CURRENT_SCHEMA_VERSION = 1

# the state contract leaves the per-kind delta shape to be specified at v1
# implementation. Bound everything tightly — adversarial event injection
# (the state contract) is the load-bearing threat model and a too-large magnitude is
# the obvious attack. Per-event magnitudes are deliberately small; large
# shifts must accumulate over many events, where the rate limit (§7.2)
# bounds total drift.
_MAX_MAGNITUDE = 0.5
_DRIFT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})

# Closed per-kind delta key set. Anything outside the allowed keys is
# rejected at construction so an attacker cannot smuggle audit-shaped
# extra fields into the canonical signing surface alongside legitimate
# keys (the signer in ``aeros.audit.signer`` serialises the entire delta).
_ALLOWED_DELTA_KEYS_BY_KIND: dict[str, frozenset[str]] = {
    "skill_confidence_decrease": frozenset({"magnitude"}),
    "skill_confidence_increase": frozenset({"magnitude"}),
    "risk_appetite_calibration_tighten": frozenset({"magnitude"}),
    "risk_appetite_calibration_relax": frozenset({"magnitude"}),
    "human_check_threshold_relax": frozenset({"action_kind", "magnitude"}),
    "human_check_threshold_tighten": frozenset({"action_kind", "magnitude"}),
    "drift_event": frozenset({"severity", "reason"}),
}


def _deep_freeze(value: Any) -> Any:
    """Recursively replace mutable containers with immutable equivalents.

    dict → ``MappingProxyType``; list/tuple → tuple; primitives unchanged.
    Sets, bytes, and arbitrary objects are rejected: the canonical signing
    path is ``json.dumps`` on ``dict(event.delta)`` and only types that
    survive that round-trip belong in the audit surface.

    Recursive: a value that is itself a container is frozen all the way
    down, so post-construction mutation through any nested handle (e.g.
    ``event.delta["audit"].append(...)``) raises rather than silently
    altering the bytes the chain verifier signed.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    raise ValueError(
        f"delta values must be JSON-safe primitives or nested "
        f"dict/list of the same; got {type(value).__name__}"
    )


class PersonaAdaptiveEvent(BaseModel):
    """Typed mutation of ``PersonaAdaptive``. One row in episodic.db.

    Frozen + ``extra="forbid"`` so adversarial extra fields fail
    validation rather than ride along into the audit chain.

    Deep immutability: ``delta`` is wrapped in ``MappingProxyType`` and
    ``triggering_cause_ids`` is a tuple, so a caller cannot mutate either
    after construction to bypass per-kind validation or alter the
    canonical signing surface (the state contract audit-event contract).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EventKind
    capability: str | None = None
    delta: Mapping[str, Any] = Field(default_factory=dict)
    triggering_cause_ids: tuple[str, ...] = Field(default_factory=tuple)
    operator_signature: str | None = None
    timestamp: datetime
    schema_version: int = CURRENT_SCHEMA_VERSION

    # Q5 (orchestrator decision): safety-floor activations
    # write the event with applied=False; resolver fold skips them.
    applied: bool = True

    @field_serializer("delta")
    def _serialize_delta(self, value: Mapping[str, Any]) -> dict[str, Any]:
        # ``MappingProxyType`` isn't natively JSON-serializable; unwrap to
        # a plain dict for ``model_dump`` / ``model_dump_json``.
        return dict(value)

    @model_validator(mode="after")
    def _validate_delta_for_kind(self) -> "PersonaAdaptiveEvent":
        kind = self.kind
        delta = self.delta

        allowed = _ALLOWED_DELTA_KEYS_BY_KIND.get(kind, frozenset())
        extras = set(delta.keys()) - allowed
        if extras:
            raise ValueError(
                f"event kind={kind!r} delta has unexpected keys "
                f"{sorted(extras)}; allowed={sorted(allowed)}"
            )

        if kind in {"skill_confidence_decrease", "skill_confidence_increase"}:
            if not self.capability:
                raise ValueError(
                    f"event kind={kind!r} requires non-empty 'capability'"
                )
            magnitude = delta.get("magnitude")
            if not isinstance(magnitude, (int, float)):
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be a number"
                )
            if not 0.0 < float(magnitude) <= _MAX_MAGNITUDE:
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be in "
                    f"(0, {_MAX_MAGNITUDE}], got {magnitude!r}"
                )

        elif kind in {
            "risk_appetite_calibration_tighten",
            "risk_appetite_calibration_relax",
        }:
            magnitude = delta.get("magnitude")
            if not isinstance(magnitude, (int, float)):
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be a number"
                )
            if not 0.0 < float(magnitude) <= _MAX_MAGNITUDE:
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be in "
                    f"(0, {_MAX_MAGNITUDE}], got {magnitude!r}"
                )

        elif kind in {
            "human_check_threshold_relax",
            "human_check_threshold_tighten",
        }:
            action_kind = delta.get("action_kind")
            if not isinstance(action_kind, str) or not action_kind:
                raise ValueError(
                    f"event kind={kind!r} delta.action_kind must be a non-empty str"
                )
            magnitude = delta.get("magnitude")
            if not isinstance(magnitude, (int, float)):
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be a number"
                )
            if not 0.0 < float(magnitude) <= _MAX_MAGNITUDE:
                raise ValueError(
                    f"event kind={kind!r} delta.magnitude must be in "
                    f"(0, {_MAX_MAGNITUDE}], got {magnitude!r}"
                )

        elif kind == "drift_event":
            severity = delta.get("severity")
            if severity not in _DRIFT_SEVERITIES:
                raise ValueError(
                    f"drift_event delta.severity must be one of "
                    f"{sorted(_DRIFT_SEVERITIES)}, got {severity!r}"
                )
            reason = delta.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    "drift_event delta.reason must be a non-empty str"
                )

        # Snapshot + recursively freeze nested mutables so neither the
        # caller-owned inputs nor any nested handle (e.g.
        # ``event.delta["audit"].append(...)``) can drift the validated
        # state after construction. The model itself is frozen=True;
        # ``object.__setattr__`` is the supported escape hatch for
        # swapping in the immutable proxy.
        object.__setattr__(self, "delta", _deep_freeze(dict(delta)))
        return self


def is_signature_required(kind: str) -> bool:
    """Return True iff a valid event of this kind must carry a signature."""
    return kind in SIGNATURE_REQUIRED_KINDS


def is_automatic_tighten_kind(kind: str) -> bool:
    """Return True iff this kind may be auto-emitted without a signature."""
    return kind in AUTOMATIC_TIGHTEN_KINDS
