"""Auto-emit ``PersonaAdaptiveEvent`` rows from runtime tighten signals.

the state contract (asymmetric trust): tighten-direction events may emit
without an operator signature. This module owns the two automatic emit
paths:

* **failure-threshold path** — N consecutive failures of skill X fold
  into a ``skill_confidence_decrease`` event (capability=X, magnitude
  default 0.1).
* **emergency_stop / runtime_safety advisory path** — each advisory
  emits a ``drift_event`` with severity per-trigger.

Each emission signs the structural surface of the event with the agent's
resident Ed25519 key (``aeros.audit.signer``) and writes a single
``persona_adaptation`` row into the agent's episodic memory. The
signature, signer pubkey, and key fingerprint live in the row's payload
blob so a Layer 3 round-trip carries the audit chain across deployments
unchanged.

This module wires the emit-time governance check (``persona_governance``):
when a ``persona_core`` is supplied at construction, ``_emit`` consults
the prior 24h window + current adaptive state and clamps the new event
to ``applied=False`` if it would cross a safety floor or hit a rate
ceiling. The audit row is still written (typed-event recording: every adaptation is a
typed audit event) — only the resolver fold's effect is suppressed,
which the operator counter-signs via ``aeros agent review``.
Tripping the aggregate circuit breaker also emits a one-shot operator
notification ``drift_event`` (severity=high, reason=rate_limit_circuit_breaker).
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aeros.audit.signer import AuditSigner
from aeros.runtime.persona_event import (
    AUTOMATIC_TIGHTEN_KINDS,
    PersonaAdaptiveEvent,
)

if TYPE_CHECKING:
    from aeros.runtime.persistent_agent import PersistentAgent
    from aeros.runtime.persona_core import PersonaCore

logger = logging.getLogger(__name__)


# The episodic.db row ``kind`` for every PersonaAdaptiveEvent — automatic
# or operator-signed. Resolver, CLIs, and Layer 3 export all filter on
# this single value (see ``EpisodicMemory.by_kind``).
PERSONA_ADAPTATION_KIND = "persona_adaptation"

_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_CONFIDENCE_DECREMENT = 0.1


# Severity calibration for drift_event triggers. The values fold into
# the resolver's per-severity nudge (persona_resolver._apply_event for
# kind=drift_event). Operator override events are surfaced at medium so
# they show up in the drift history but don't dominate the risk dial.
_DRIFT_SEVERITY_BY_TRIGGER: dict[str, str] = {
    "emergency_stop": "high",
    "runtime_safety": "medium",
    "operator_override": "medium",
}


class PersonaAdaptationEmitter:
    """Stateful emitter for automatic ``persona_adaptation`` rows.

    One instance per :class:`PersistentAgent`. Tracks per-capability
    consecutive-failure counters; resets on observed success of the
    same capability or after each emission so the next event requires
    another ``failure_threshold`` consecutive failures (rate-limit
    friendly without sub 5's full circuit-breaker).

    Thread-safe: a single lock guards counter mutations + emission, so
    a burst of concurrent ``record_skill_failure`` calls produces at
    most one event per threshold crossing.
    """

    def __init__(
        self,
        agent: "PersistentAgent",
        signer: AuditSigner,
        *,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        confidence_decrement: float = _DEFAULT_CONFIDENCE_DECREMENT,
        persona_core: "PersonaCore | None" = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1, got {failure_threshold}"
            )
        if not 0.0 < confidence_decrement <= 0.5:
            raise ValueError(
                "confidence_decrement must be in (0, 0.5], got "
                f"{confidence_decrement}"
            )
        self._agent = agent
        self._signer = signer
        self._failure_threshold = failure_threshold
        self._confidence_decrement = confidence_decrement
        self._persona_core = persona_core
        self._consecutive_failures: dict[str, int] = {}
        self._lock = threading.Lock()
        # One-shot guard for the circuit-breaker notification emit. The
        # synthetic drift_event is itself a persona_adaptation row, so we
        # set this flag while it persists to short-circuit the recursion.
        self._notifying_circuit_breaker = False

    @property
    def signer(self) -> AuditSigner:
        return self._signer

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def confidence_decrement(self) -> float:
        return self._confidence_decrement

    def consecutive_failures(self, capability: str) -> int:
        """Read-only view of the current failure counter for tests."""
        with self._lock:
            return self._consecutive_failures.get(capability, 0)

    # -------------------- skill failure / success path -------------------- #

    def record_skill_success(self, capability: str) -> None:
        """Reset the consecutive-failure counter for ``capability``.

        Called by RuntimeGateway on the ``allow`` outcome. No-op for an
        empty capability name (defensive: skill identifiers are
        capability-namespaced so an empty name is never legitimate).
        """
        if not capability:
            return
        with self._lock:
            self._consecutive_failures.pop(capability, None)

    def record_skill_failure(
        self,
        capability: str,
        *,
        cause_ids: list[str] | None = None,
    ) -> PersonaAdaptiveEvent | None:
        """Increment the failure counter and emit on threshold crossing.

        Returns the emitted event when ``failure_threshold`` consecutive
        failures have been observed; otherwise None. Reset semantics
        are emit-and-clear so a sustained burst produces an event every
        ``failure_threshold`` failures (one drop step per crossing).
        """
        if not capability:
            return None
        with self._lock:
            current = self._consecutive_failures.get(capability, 0) + 1
            if current < self._failure_threshold:
                self._consecutive_failures[capability] = current
                return None
            self._consecutive_failures[capability] = 0
        event = PersonaAdaptiveEvent(
            kind="skill_confidence_decrease",
            capability=capability,
            delta={"magnitude": self._confidence_decrement},
            triggering_cause_ids=list(cause_ids or [_synth_cause_id()]),
            operator_signature=None,
            timestamp=_utc_now(),
        )
        self._emit(event, trigger="failure_threshold")
        return event

    # ---------------------- drift event paths ---------------------- #

    def record_drift_event(
        self,
        *,
        reason: str,
        severity: str = "medium",
        cause_ids: list[str] | None = None,
    ) -> PersonaAdaptiveEvent:
        """Emit a single ``drift_event`` row.

        Caller chooses ``severity`` from {"low", "medium", "high"};
        :class:`PersonaAdaptiveEvent` rejects anything else upstream.
        """
        event = PersonaAdaptiveEvent(
            kind="drift_event",
            capability=None,
            delta={"severity": severity, "reason": reason},
            triggering_cause_ids=list(cause_ids or [_synth_cause_id()]),
            operator_signature=None,
            timestamp=_utc_now(),
        )
        self._emit(event, trigger=reason)
        return event

    def record_emergency_stop(
        self,
        *,
        reason: str = "emergency_stop",
        cause_ids: list[str] | None = None,
    ) -> PersonaAdaptiveEvent:
        return self.record_drift_event(
            reason=reason,
            severity=_DRIFT_SEVERITY_BY_TRIGGER["emergency_stop"],
            cause_ids=cause_ids,
        )

    def record_runtime_safety_advisory(
        self,
        *,
        violation: str,
        cause_ids: list[str] | None = None,
    ) -> PersonaAdaptiveEvent:
        """Emit drift_event for a ``RuntimeSafety.enforce`` violation.

        ``violation`` is the typed string from
        ``RuntimeSafetyTrace.violation_type``: "velocity_exceeded",
        "workspace_bounds", "forbidden_zone", "force_limit". The reason
        prefix lets operators grep persona_adaptation rows for runtime-
        safety triggers separately from emergency_stop.
        """
        return self.record_drift_event(
            reason=f"runtime_safety:{violation}",
            severity=_DRIFT_SEVERITY_BY_TRIGGER["runtime_safety"],
            cause_ids=cause_ids,
        )

    def emit_reflection_derived(
        self,
        event: PersonaAdaptiveEvent,
        *,
        trigger: str = "active_dreaming",
    ) -> None:
        """Emit a typed event proposed by an Active Dreaming reflection pass.

        Offline reflection (the state contract) reuses the auto-tighten emit path so the same
        §6.3 asymmetric-trust + §7.2 rate-limit + §8.3 safety-floor gates
        apply to reflection-derived events. The caller (active_dreaming
        runner) supplies a fully-formed event with ``triggering_cause_ids``
        already pointing at the parent ``reflection_summary`` row.
        """
        self._emit(event, trigger=trigger)

    # ----------------------------- internals ----------------------------- #

    def _emit(self, event: PersonaAdaptiveEvent, *, trigger: str) -> None:
        if event.kind not in AUTOMATIC_TIGHTEN_KINDS:
            # Only the three tighten kinds can emit without a signature
            # (signature-policy). Operator-signed events go through aeros agent
            # sign-event, not this emitter.
            raise ValueError(
                f"PersonaAdaptationEmitter cannot auto-emit kind={event.kind!r};"
                " operator-signed kinds must use aeros agent sign-event"
            )
        event, governance_reason, breaker_just_tripped = self._evaluate_governance(event)
        signature = self._signer.sign(event)
        payload: dict[str, Any] = {
            "event": _event_to_payload(event),
            "signature": signature.signature_hex,
            "signer_pubkey": signature.signer_pubkey_hex,
            "fingerprint": signature.fingerprint,
            "trigger": trigger,
            "governance_reason": governance_reason,
        }
        self._agent.record(
            kind=PERSONA_ADAPTATION_KIND,
            intent_id=None,
            summary=_summarise(event, trigger, governance_reason),
            payload=payload,
        )
        logger.info(
            "persona_adaptation emit kind=%s capability=%s trigger=%s "
            "fingerprint=%s applied=%s reason=%s",
            event.kind,
            event.capability,
            trigger,
            signature.fingerprint,
            event.applied,
            governance_reason,
        )
        from aeros.runtime.persona_metrics import get_registry

        registry = get_registry()
        registry.inc_adaptation_event(kind=event.kind, source="automatic")
        if governance_reason != "ok":
            # the state contract names this counter ``persona_floor_clamps_total``
            # but the floor_kind label covers the governance reasons
            # that produce ``applied=False`` rows: floor activations,
            # rate-limit ceilings, and the aggregate circuit breaker.
            registry.inc_floor_clamp(floor_kind=governance_reason)
        if breaker_just_tripped:
            registry.set_circuit_breaker_active(True)
        if breaker_just_tripped and not self._notifying_circuit_breaker:
            self._emit_circuit_breaker_notification()

    def _evaluate_governance(
        self, event: PersonaAdaptiveEvent
    ) -> tuple[PersonaAdaptiveEvent, str, bool]:
        """Return (possibly clamped event, reason tag, breaker_edge_flag).

        Skipped entirely when ``persona_core`` was not supplied at
        construction so the legacy / test paths that don't wire
        governance behave as before. Also skipped while the emitter is
        in the middle of writing the circuit-breaker notification, to
        avoid the synthetic drift_event recursing through the same
        check.
        """
        if self._persona_core is None or self._notifying_circuit_breaker:
            return event, "ok", False

        from aeros.runtime.persona_governance import (
            circuit_breaker_active,
            circuit_breaker_just_tripped,
            evaluate_governance,
        )
        from aeros.runtime.persona_metrics import get_registry
        from aeros.runtime.persona_resolver import _RISK_APPETITE_CAPS, effective_persona

        prior = self._load_prior_events()
        applied_prior = [e for e in prior if e.applied]
        # Reuse the same fold the resolver uses so the floor check and
        # the resolver agree on what the current adaptive state is.
        profile = effective_persona(self._persona_core, applied_prior)
        risk_cap = _RISK_APPETITE_CAPS.get(self._persona_core.risk_appetite, 0.7)
        evaluated, reason = evaluate_governance(
            event,
            prior_events=prior,
            current_adaptive=profile.adaptive,
            risk_appetite_cap=risk_cap,
            now=event.timestamp,
        )
        breaker_edge = circuit_breaker_just_tripped(prior, evaluated, now=event.timestamp)
        # Refresh the gauge on every governance evaluation so the
        # window sliding off (breaker re-arms) is visible without
        # waiting for the next breaker trip.
        get_registry().set_circuit_breaker_active(
            circuit_breaker_active(prior + [evaluated], now=event.timestamp)
        )
        return evaluated, reason, breaker_edge

    def _load_prior_events(self) -> list[PersonaAdaptiveEvent]:
        """Reconstruct typed events from prior persona_adaptation rows.

        Pulled into a helper so the test-only in-memory backend and the
        SQLite-backed path use the same code: both call
        :py:meth:`EpisodicMemory.by_kind` which scans by indexed key on
        the SQLite path.
        """
        rows = self._agent.episodic.by_kind(PERSONA_ADAPTATION_KIND)
        out: list[PersonaAdaptiveEvent] = []
        for row in rows:
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            event_dict = payload.get("event")
            if not isinstance(event_dict, dict):
                continue
            try:
                out.append(PersonaAdaptiveEvent(**event_dict))
            except Exception:  # noqa: BLE001 — corrupt row should not crash emit
                logger.warning(
                    "skipping malformed persona_adaptation row at ts=%s",
                    row.timestamp,
                )
        return out

    def _emit_circuit_breaker_notification(self) -> None:
        """One-shot drift_event for the operator when the breaker trips.

        Severity=high so the row is visually distinct in the audit log;
        reason carries the canonical ``rate_limit_circuit_breaker`` tag
        the operator dashboards / docs grep for. The notification itself
        is governed by the same rules and will land applied=False (the
        breaker is now active by definition), but the operator-facing
        row is the load-bearing artefact.
        """
        self._notifying_circuit_breaker = True
        try:
            event = PersonaAdaptiveEvent(
                kind="drift_event",
                capability=None,
                delta={
                    "severity": "high",
                    "reason": "rate_limit_circuit_breaker",
                },
                triggering_cause_ids=[_synth_cause_id()],
                operator_signature=None,
                timestamp=_utc_now(),
                applied=False,
            )
            self._emit(event, trigger="rate_limit_circuit_breaker")
        finally:
            self._notifying_circuit_breaker = False


def _event_to_payload(event: PersonaAdaptiveEvent) -> dict[str, Any]:
    """Serialise event into a JSON-safe dict for episodic.db payload."""
    return {
        "kind": event.kind,
        "capability": event.capability,
        "delta": dict(event.delta),
        "triggering_cause_ids": list(event.triggering_cause_ids),
        "operator_signature": event.operator_signature,
        "timestamp": event.timestamp.isoformat(),
        "schema_version": event.schema_version,
        "applied": event.applied,
    }


def _summarise(event: PersonaAdaptiveEvent, trigger: str, reason: str = "ok") -> str:
    suffix = f" applied=False:{reason}" if not event.applied else ""
    if event.kind == "skill_confidence_decrease":
        return (
            f"auto-emit skill_confidence_decrease {event.capability} ({trigger})"
            f"{suffix}"
        )
    if event.kind == "drift_event":
        severity = str(event.delta.get("severity", "?"))
        return f"auto-emit drift_event severity={severity} ({trigger}){suffix}"
    return f"auto-emit {event.kind} ({trigger}){suffix}"


def _synth_cause_id() -> str:
    """Fallback cause_id for callers that don't supply one.

    Wired callers (safety-floor wrapper, the Layer 3
    bundle ingest) supply the real episodic-source cause_id chain;
    bench / test paths fall back here to keep ``triggering_cause_ids``
    non-empty (chain-verification needs at least one element to verify against).
    """
    return f"auto:{uuid.uuid4()}"


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
