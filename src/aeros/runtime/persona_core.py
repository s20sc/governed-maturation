"""PersonaCore — frozen operator commitments that enter the identity hash.

The persona state contract splits the legacy ``PersonaProfile`` along the
maturation axis: the operator's deployment-time commitments stay frozen
in ``PersonaCore`` (and continue to flow into ``M_A^id``); lived-in
calibration moves to ``PersonaAdaptive`` (and never enters the identity
hash). The split is behaviour-neutral — ``PersonaCore`` carries the same
fields the historical ``PersonaProfile`` did, and its
``to_manifest_payload()`` produces a dict byte-equal to the pre-refactor
``PersonaProfile.to_manifest_payload()`` so ``persona_hash`` is preserved.

Adaptive fields reside in ``PersonaAdaptive`` and are updated through typed
events without changing ``PersonaCore``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RiskAppetite = Literal["conservative", "balanced", "aggressive"]
VerbosityLevel = Literal["terse", "balanced", "verbose"]
HumanCheckPreference = Literal["never", "on_uncertainty", "on_high_risk", "always"]
ValueOrdering = Literal[
    "human_first",
    "task_first",
    "efficiency_first",
]


@dataclass(frozen=True)
class PersonaCore:
    """Frozen operator commitments. Hashed into ``IdentityManifest.persona_hash``.

    Field set and serialisation are kept byte-equal to the historical
    ``PersonaProfile`` so the split refactor preserves
    ``persona_hash`` on every shipped fixture.
    """

    name: str = "default"
    description: str = ""

    risk_appetite: RiskAppetite = "balanced"
    human_check: HumanCheckPreference = "on_high_risk"
    value_ordering: ValueOrdering = "human_first"

    verbosity: VerbosityLevel = "balanced"

    accept_unsoaked_ecm: bool = False
    min_ecm_soak_days: int = 7

    tags: tuple[str, ...] = ()

    def to_manifest_payload(self) -> dict[str, Any]:
        """Stable serialisation for ``hash_persona``. Byte-equal to the
        pre-split ``PersonaProfile.to_manifest_payload()`` shape.
        """
        return {
            "name": self.name,
            "description": self.description,
            "risk_appetite": self.risk_appetite,
            "human_check": self.human_check,
            "value_ordering": self.value_ordering,
            "verbosity": self.verbosity,
            "accept_unsoaked_ecm": self.accept_unsoaked_ecm,
            "min_ecm_soak_days": self.min_ecm_soak_days,
            "tags": list(self.tags),
        }
