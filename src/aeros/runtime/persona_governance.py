"""Persona governance — freeze flag, auto-tighten safety floor, rate limit.

the state contract Three orthogonal pieces wrapping the auto-emit
path and the resolver entry points:

* **Freeze** (the state contract) — operator-driven kill-switch that makes the
  resolver behave as ``effective_persona(core, [])`` regardless of what is
  in episodic.db. Stored in ``bridge/configs/runtime_state.json`` so it
  survives bridge restarts; per-identity scoped with a ``"default"`` key
  for single-tenant hosts. CLI: ``aeros agent freeze-persona`` /
  ``aeros agent unfreeze-persona``.
* **Safety floor** (the state contract) — automatic tighten events that would push
  ``PersonaAdaptive`` past a hard floor are written with ``applied=False``
  per the orchestrator Q5 decision: the audit row stays in episodic.db
  (typed-event recording unbroken) but the resolver fold skips it. Operator counter-signs
  via ``aeros agent review`` to flip the row.
* **Rate limit** (the state contract) — per-kind-class ceilings (auto-tighten 50/day,
  operator-relax 10/day, operator-tighten unlimited) plus an aggregate
  100/day circuit breaker. At-limit events are clamped with
  ``applied=False``; tripping the aggregate also enqueues a one-shot
  ``drift_event`` so an operator notices the degraded mode.

The pure-decision API (:func:`evaluate_governance`) takes the prior event
window and the current adaptive state and returns the (possibly clamped)
event plus a human-readable reason. The emitter wraps this; the resolver
relies on the ``applied`` bit alone, so a Layer 3 bundle import
that brings already-clamped rows is folded identically without re-running
governance.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_event import (
    AUTOMATIC_TIGHTEN_KINDS,
    RELAX_KINDS,
    PersonaAdaptiveEvent,
)

logger = logging.getLogger(__name__)

# the state contract — auto-tighten safety floors. Operator-signed events bypass
# (the operator IS the human in the loop, by definition).
SKILL_CONFIDENCE_FLOOR: Final[float] = 0.05
HUMAN_CHECK_THRESHOLD_CEILING: Final[float] = 0.95
RISK_APPETITE_FLOOR_FRACTION: Final[float] = 0.1  # × risk_appetite_cap

# the state contract — rolling-24h ceilings. Per-kind-class ceilings clamp the
# offending event with applied=False; the aggregate ceiling additionally
# trips the circuit breaker (resolver degraded mode + operator drift_event).
AUTOMATIC_TIGHTEN_DAILY_LIMIT: Final[int] = 50
OPERATOR_RELAX_DAILY_LIMIT: Final[int] = 10
AGGREGATE_DAILY_LIMIT: Final[int] = 100
RATE_WINDOW: Final[timedelta] = timedelta(hours=24)

# Default location for the freeze flag. Relative path so a checkout-rooted
# `aeros agent freeze-persona` finds the bridge config without needing an
# explicit flag; the CLI exposes ``--runtime-state`` for tests / non-default
# layouts.
DEFAULT_RUNTIME_STATE_PATH: Final[Path] = Path("bridge/configs/runtime_state.json")

# Default identity key in the freeze map. Single-tenant hosts read/write
# under this key so the file stays human-editable; multi-tenant flows pass
# an explicit identity_hash.
_DEFAULT_IDENTITY_KEY: Final[str] = "default"


# ---------------------------- runtime state file ---------------------------- #


class RuntimeStateFile:
    """JSON-backed flag store; today only owns the persona freeze map.

    Schema (forward-additive — unknown top-level keys are preserved on
    write):

    .. code-block:: json

        {
          "persona_freeze": {
            "default": false,
            "<identity_hash>": true
          }
        }

    Thread-safe at the instance level via a single lock; cross-process
    writes rely on the OS atomic-rename pattern (``.write_text`` + small
    payload), which matches the bridge's ``persona.yaml`` reload pattern.
    """

    def __init__(self, path: Path | str = DEFAULT_RUNTIME_STATE_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text()
        except OSError as exc:
            logger.warning("runtime_state.json unreadable at %s: %s", self._path, exc)
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("runtime_state.json malformed at %s: %s", self._path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    def is_persona_frozen(self, identity_hash: str | None = None) -> bool:
        with self._lock:
            state = self._read()
            freeze = state.get("persona_freeze")
            if not isinstance(freeze, dict):
                return False
            if identity_hash and identity_hash in freeze:
                return bool(freeze[identity_hash])
            return bool(freeze.get(_DEFAULT_IDENTITY_KEY, False))

    def set_persona_freeze(
        self, frozen: bool, identity_hash: str | None = None
    ) -> None:
        with self._lock:
            state = self._read()
            freeze = state.get("persona_freeze")
            if not isinstance(freeze, dict):
                freeze = {}
            key = identity_hash or _DEFAULT_IDENTITY_KEY
            freeze[key] = bool(frozen)
            state["persona_freeze"] = freeze
            self._write(state)


# ---------------------------- pure decision helpers ---------------------------- #


def kind_class(kind: str) -> str:
    """Map an event kind to its rate-limit class.

    Three classes (the state contract): ``automatic_tighten`` (50/day),
    ``operator_relax`` (10/day), ``operator_tighten`` (unlimited).
    """
    if kind in AUTOMATIC_TIGHTEN_KINDS:
        return "automatic_tighten"
    if kind in RELAX_KINDS:
        return "operator_relax"
    return "operator_tighten"


def _events_in_window(
    events: Sequence[PersonaAdaptiveEvent],
    now: datetime,
) -> list[PersonaAdaptiveEvent]:
    cutoff = now - RATE_WINDOW
    return [e for e in events if e.timestamp >= cutoff]


def aggregate_count(
    events: Sequence[PersonaAdaptiveEvent], now: datetime | None = None
) -> int:
    """Count all events in the trailing 24h window (any kind, any applied)."""
    when = now or datetime.now(tz=UTC)
    return len(_events_in_window(events, when))


def kind_class_count(
    cls: str,
    events: Sequence[PersonaAdaptiveEvent],
    now: datetime | None = None,
) -> int:
    when = now or datetime.now(tz=UTC)
    return sum(1 for e in _events_in_window(events, when) if kind_class(e.kind) == cls)


def kind_class_at_limit(
    kind: str,
    events: Sequence[PersonaAdaptiveEvent],
    now: datetime | None = None,
) -> bool:
    cls = kind_class(kind)
    if cls == "operator_tighten":
        return False  # unlimited per the state contract
    limit = (
        AUTOMATIC_TIGHTEN_DAILY_LIMIT
        if cls == "automatic_tighten"
        else OPERATOR_RELAX_DAILY_LIMIT
    )
    return kind_class_count(cls, events, now) >= limit


def circuit_breaker_active(
    events: Sequence[PersonaAdaptiveEvent], now: datetime | None = None
) -> bool:
    """Return True iff aggregate count in the trailing window ≥ 100."""
    return aggregate_count(events, now) >= AGGREGATE_DAILY_LIMIT


# Severity → risk_appetite nudge for drift_event. Mirrors the resolver fold
# (see ``persona_resolver._apply_event`` for kind=drift_event); kept here so
# the floor check uses the same magnitudes the fold will apply.
_DRIFT_NUDGE_BY_SEVERITY: Final[dict[str, float]] = {
    "low": 0.05,
    "medium": 0.10,
    "high": 0.20,
}


def would_violate_floor(
    adaptive: PersonaAdaptive,
    event: PersonaAdaptiveEvent,
    risk_appetite_cap: float,
) -> bool:
    """Return True iff applying ``event`` would push past an auto-tighten floor.

    Only meaningful for the three :data:`AUTOMATIC_TIGHTEN_KINDS`. Relax /
    operator-tighten kinds are operator-signed; the operator IS the human
    in the loop so the floor does not apply (the state contract). The resolver
    enforces the same numeric clamps as a hard ceiling regardless, so
    operator-signed events still cannot escape ``[0, 1]`` or the cap.
    """
    if event.kind not in AUTOMATIC_TIGHTEN_KINDS:
        return False
    delta = event.delta

    if event.kind == "skill_confidence_decrease":
        cap = event.capability or ""
        current = adaptive.skill_confidence.get(cap, 1.0)
        if current <= SKILL_CONFIDENCE_FLOOR:
            # Already at or below floor; any further auto-decrease is queued.
            return True
        magnitude = float(delta.get("magnitude", 0.0))
        return (current - magnitude) < SKILL_CONFIDENCE_FLOOR

    if event.kind == "risk_appetite_calibration_tighten":
        magnitude = float(delta.get("magnitude", 0.0))
        floor_value = RISK_APPETITE_FLOOR_FRACTION * risk_appetite_cap
        if adaptive.risk_appetite_actual <= floor_value:
            return True
        return (adaptive.risk_appetite_actual - magnitude) < floor_value

    # drift_event — folds as a tighten on risk_appetite_actual.
    severity = str(delta.get("severity", "low"))
    nudge = _DRIFT_NUDGE_BY_SEVERITY.get(severity, 0.05)
    floor_value = RISK_APPETITE_FLOOR_FRACTION * risk_appetite_cap
    if adaptive.risk_appetite_actual <= floor_value:
        return True
    return (adaptive.risk_appetite_actual - nudge) < floor_value


# Outcome of governance evaluation. The reason is recorded on the row's
# payload (``governance_reason``) so an operator scrolling the audit chain
# can grep for ``floor`` / ``rate_limit`` / ``circuit_breaker`` rows.
GovernanceReason = str  # Literal["ok", "floor", "rate_limit", "circuit_breaker"]


def evaluate_governance(
    event: PersonaAdaptiveEvent,
    *,
    prior_events: Sequence[PersonaAdaptiveEvent],
    current_adaptive: PersonaAdaptive,
    risk_appetite_cap: float,
    now: datetime | None = None,
) -> tuple[PersonaAdaptiveEvent, GovernanceReason]:
    """Decide whether to clamp ``event`` to ``applied=False`` before persist.

    Order of checks (most-restrictive first):

    1. **Circuit breaker** — aggregate ≥ 100 in the window. Every event
       (including operator-signed) is clamped until the window slides off.
    2. **Per-kind rate limit** — tighten 50/day, relax 10/day. At-limit
       events of that class are clamped.
    3. **Safety floor** — auto-tighten only. Floor is per-skill / per
       action_kind / per-risk-cap; operator-signed events bypass.

    Returns the event unchanged + ``"ok"`` when none fire. Otherwise
    returns ``event.model_copy(update={"applied": False})`` plus a
    short reason tag.
    """
    when = now or datetime.now(tz=UTC)

    if circuit_breaker_active(prior_events, when):
        return event.model_copy(update={"applied": False}), "circuit_breaker"

    cls = kind_class(event.kind)
    if cls != "operator_tighten" and kind_class_at_limit(event.kind, prior_events, when):
        return event.model_copy(update={"applied": False}), "rate_limit"

    if event.kind in AUTOMATIC_TIGHTEN_KINDS and would_violate_floor(
        current_adaptive, event, risk_appetite_cap
    ):
        return event.model_copy(update={"applied": False}), "floor"

    return event, "ok"


def circuit_breaker_just_tripped(
    prior_events: Sequence[PersonaAdaptiveEvent],
    new_event: PersonaAdaptiveEvent,
    now: datetime | None = None,
) -> bool:
    """Return True iff this event is the one that pushes aggregate past 100.

    Used by the emitter to decide whether to enqueue the one-shot operator
    notification. We notify exactly once per window crossing — the
    ``aggregate_count(prior) < limit <= aggregate_count(prior + new)``
    condition fires only on the breaker edge.
    """
    when = now or datetime.now(tz=UTC)
    before = aggregate_count(prior_events, when)
    if before >= AGGREGATE_DAILY_LIMIT:
        return False
    after = before + (1 if new_event.timestamp >= when - RATE_WINDOW else 0)
    return after >= AGGREGATE_DAILY_LIMIT


__all__ = [
    "AGGREGATE_DAILY_LIMIT",
    "AUTOMATIC_TIGHTEN_DAILY_LIMIT",
    "DEFAULT_RUNTIME_STATE_PATH",
    "GovernanceReason",
    "HUMAN_CHECK_THRESHOLD_CEILING",
    "OPERATOR_RELAX_DAILY_LIMIT",
    "RATE_WINDOW",
    "RISK_APPETITE_FLOOR_FRACTION",
    "RuntimeStateFile",
    "SKILL_CONFIDENCE_FLOOR",
    "aggregate_count",
    "circuit_breaker_active",
    "circuit_breaker_just_tripped",
    "evaluate_governance",
    "kind_class",
    "kind_class_at_limit",
    "kind_class_count",
    "would_violate_floor",
]
