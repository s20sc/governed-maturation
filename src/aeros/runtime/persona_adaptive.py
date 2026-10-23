"""PersonaAdaptive — lived-in calibration overlay; not part of the identity hash.

The persona state contract places this layer outside ``M_A^id`` (identity-invariance) so
that any sequence of typed adaptation events (the state contract) can shift state
without flipping ``identity_hash``. The four fields come from the state contract:

- ``risk_appetite_actual`` — operational risk dial within ``risk_appetite_cap``
- ``skill_confidence`` — per-skill belief in ``[0, 1]``
- ``human_check_threshold`` — per-action-kind threshold in ``[0, 1]``
- ``tendencies`` — free-form catch-all bucket for future calibration shifts

The container is built fresh by ``persona_resolver.effective_persona`` on
every read, so callers should treat its dict fields as immutable. The
class is not declared ``frozen=True`` because the dict-typed fields would
clash with the dataclass-derived ``__hash__`` — but the resolver returns
distinct instances per call, never mutating an existing one.

Layer 3 export bundle: the empty payload shape is preserved when no
events have fired (empty-state compatibility null-safe slot). When events exist, migration
uses the bundle to carry the typed-event log so a fresh import
reconstructs the same adaptive state via the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonaAdaptive:
    """Resolver fold output: lived-in calibration over ``PersonaCore``.

    Treated as a value type even though not frozen — every resolver
    invocation returns a fresh instance. Mutating a returned instance is
    not supported and will diverge from the audit chain. Pass the
    instance through ``dataclasses.replace`` (with copies of any dict
    fields) when building the next state in a fold.
    """

    # Operational risk dial. Bounded to ``[0, risk_appetite_cap]`` by the
    # resolver; the cap itself is derived by the resolver from
    # ``PersonaCore.risk_appetite`` (one of the three discrete values).
    # The default 0.5 places a fresh agent at the midpoint of a balanced
    # cap before any adaptive event has fired.
    risk_appetite_actual: float = 0.5

    # Per-skill confidence in ``[0, 1]``. Missing entries default to 1.0
    # at the resolver's read time — a never-touched skill is assumed
    # operationally sound until evidence accumulates.
    skill_confidence: dict[str, float] = field(default_factory=dict)

    # Per-action-kind human-check threshold in ``[0, 1]``. Higher values
    # = require operator attention more often. Missing entries inherit
    # from ``PersonaCore.human_check`` (a discrete preference) at the
    # resolver's read time.
    human_check_threshold: dict[str, float] = field(default_factory=dict)

    # Free-form catch-all. Reserved for adaptive calibrations that don't
    # fit the three numeric fields above (e.g., per-environment
    # preferences, time-of-day defaults). v1 keeps it untyped because
    # the threat model is bounded by the typed-event vocabulary, not by
    # this dict's shape.
    tendencies: dict[str, Any] = field(default_factory=dict)

    def to_bundle_payload(self) -> dict[str, Any]:
        """Layer 3 export bundle slot (``persona_adaptive.json``).

        The payload is null-safe when empty and otherwise carries the
        full state. The typed-event log itself is bundled separately by
        migration round-trip path; this slot is the resolver
        output so a quick import round-trip can sanity-check before
        re-folding the events.
        """
        return {
            "schema_version": "v1",
            "risk_appetite_actual": self.risk_appetite_actual,
            "skill_confidence": dict(self.skill_confidence),
            "human_check_threshold": dict(self.human_check_threshold),
            "tendencies": dict(self.tendencies),
        }

    @classmethod
    def empty(cls) -> "PersonaAdaptive":
        return cls()
