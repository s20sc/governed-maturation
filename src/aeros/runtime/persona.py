"""PersonaProfile — runtime read-model for the persona surface.

The bridge already hashes governance state (env_policies + policy_rules +
ecm_registry) into the agent's identity. That tells you what the robot's
*rules* are. What it doesn't tell you is what the robot's *temperament*
is — does it prefer a verbose explanation or a terse one? Will it default
to safer slower options when given a choice? Does it surface uncertainty
to the operator or attempt to resolve it autonomously?

PersonaProfile is the small declarative layer that answers those
questions. Every customer's robot can ship with a different persona
(cautious / ambitious / chatty / silent) without rewriting the bridge
code — same governance underneath, different "how" on top.

the state contract: the underlying state is split into
``PersonaCore`` (frozen, hashed into ``persona_hash``) and
``PersonaAdaptive`` (overlay, not hashed). ``PersonaProfile`` is the
runtime read-model produced by ``persona_resolver.effective_persona`` —
with an empty adaptive overlay, the read-model carries the
same fields the legacy ``PersonaProfile`` exposed, and
``to_manifest_payload()`` delegates to ``PersonaCore`` so the persona
hash is byte-equal pre/post refactor on every shipped fixture.

Loaded by ``bridge/app/main.py:lifespan`` from
``bridge/configs/persona.yaml``. Falls back to a neutral default if the
file is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_core import (
    HumanCheckPreference,
    PersonaCore,
    RiskAppetite,
    ValueOrdering,
    VerbosityLevel,
)

logger = logging.getLogger(__name__)


__all__ = [
    "HumanCheckPreference",
    "PersonaAdaptive",
    "PersonaCore",
    "PersonaProfile",
    "RiskAppetite",
    "ValueOrdering",
    "VerbosityLevel",
    "load_persona",
    "load_persona_core",
]


@dataclass(frozen=True)
class PersonaProfile:
    """Runtime read-model = ``effective_persona(PersonaCore, events)``.

    Keeps the historical flat field shape so every call site (planner
    adapters, evolution validator, MCP tools) sees the legacy surface.
    The ``.core`` accessor exposes the frozen operator commitments that
    ``hash_persona`` consumes; ``.adaptive`` carries the lived-in
    overlay populated by ``persona_resolver.effective_persona`` from
    typed events (the state contract). identity-invariance: only ``.core`` enters the identity hash.
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

    # The adaptive overlay carried into the read-model. ``compare=False``
    # keeps the legacy equality semantics — two profiles with the same
    # operator commitments compare equal regardless of overlay state, so
    # tests pinned to the original profile interface still pass and ``persona_hash``
    # equality is observable independently of adaptive drift.
    adaptive: PersonaAdaptive = field(
        default_factory=PersonaAdaptive,
        compare=False,
        hash=False,
    )

    @classmethod
    def from_core(
        cls,
        core: PersonaCore,
        adaptive: PersonaAdaptive | None = None,
    ) -> "PersonaProfile":
        """Lift a ``PersonaCore`` + adaptive overlay into the read-model.

        ``adaptive`` defaults to a fresh empty overlay so callers that
        only have a core (legacy paths) keep working unchanged.
        """
        return cls(
            name=core.name,
            description=core.description,
            risk_appetite=core.risk_appetite,
            human_check=core.human_check,
            value_ordering=core.value_ordering,
            verbosity=core.verbosity,
            accept_unsoaked_ecm=core.accept_unsoaked_ecm,
            min_ecm_soak_days=core.min_ecm_soak_days,
            tags=core.tags,
            adaptive=adaptive or PersonaAdaptive(),
        )

    @property
    def core(self) -> PersonaCore:
        """Frozen operator commitments — the only input to ``hash_persona``."""
        return PersonaCore(
            name=self.name,
            description=self.description,
            risk_appetite=self.risk_appetite,
            human_check=self.human_check,
            value_ordering=self.value_ordering,
            verbosity=self.verbosity,
            accept_unsoaked_ecm=self.accept_unsoaked_ecm,
            min_ecm_soak_days=self.min_ecm_soak_days,
            tags=self.tags,
        )

    def to_manifest_payload(self) -> dict[str, Any]:
        """Delegates to ``PersonaCore.to_manifest_payload`` — the hash input.

        identity-invariance enforcement point: the adaptive overlay is *never* in this
        payload, so resolver-driven drift cannot bend ``persona_hash``
        and therefore cannot bend ``identity_hash``.
        """
        return self.core.to_manifest_payload()


def _coerce_field(name: str, raw: Any, allowed: tuple[str, ...]) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str) or raw not in allowed:
        raise ValueError(f"persona.yaml field '{name}': got {raw!r}, expected one of {allowed}")
    return raw


def load_persona_core(path: Path | str) -> PersonaCore:
    """Parse ``persona.yaml`` into a ``PersonaCore``.

    Missing file → neutral default. Malformed file → ``ValueError`` with
    a specific field reference; never silently fall back.
    """
    p = Path(path)
    if not p.exists():
        logger.info("persona.yaml not found at %s; using neutral default", p)
        return PersonaCore()

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"persona.yaml at {p} must be a mapping, got {type(raw).__name__}")

    body = raw.get("persona") if "persona" in raw else raw
    if not isinstance(body, dict):
        raise ValueError(f"persona.yaml at {p}: 'persona' key must be a mapping")

    risk = _coerce_field(
        "risk_appetite",
        body.get("risk_appetite", "balanced"),
        ("conservative", "balanced", "aggressive"),
    )
    human_check = _coerce_field(
        "human_check",
        body.get("human_check", "on_high_risk"),
        ("never", "on_uncertainty", "on_high_risk", "always"),
    )
    value_ordering = _coerce_field(
        "value_ordering",
        body.get("value_ordering", "human_first"),
        ("human_first", "task_first", "efficiency_first"),
    )
    verbosity = _coerce_field(
        "verbosity",
        body.get("verbosity", "balanced"),
        ("terse", "balanced", "verbose"),
    )

    tags = tuple(body.get("tags") or ())
    if not all(isinstance(t, str) for t in tags):
        raise ValueError(f"persona.yaml at {p}: 'tags' must be a list of strings")

    core = PersonaCore(
        name=str(body.get("name", "default")),
        description=str(body.get("description", "")),
        risk_appetite=risk,  # type: ignore[arg-type]
        human_check=human_check,  # type: ignore[arg-type]
        value_ordering=value_ordering,  # type: ignore[arg-type]
        verbosity=verbosity,  # type: ignore[arg-type]
        accept_unsoaked_ecm=bool(body.get("accept_unsoaked_ecm", False)),
        min_ecm_soak_days=int(body.get("min_ecm_soak_days", 7)),
        tags=tags,
    )
    logger.info(
        "persona core loaded: name=%s risk=%s human_check=%s",
        core.name,
        core.risk_appetite,
        core.human_check,
    )
    return core


def load_persona(path: Path | str) -> PersonaProfile:
    """Load ``persona.yaml`` into the runtime read-model via the resolver.

    Missing file → resolver fed a neutral ``PersonaCore()``.
    """
    from aeros.runtime.persona_resolver import effective_persona

    core = load_persona_core(path)
    return effective_persona(core, ())
