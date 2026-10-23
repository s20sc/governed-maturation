"""Pure-function resolver: fold ``PersonaCore`` + adaptation events into a
runtime ``PersonaProfile`` (the state contract).

The resolver is the only legal way to derive a ``PersonaProfile`` from
its constituent parts. This module implements:

- the typed-event fold producing ``PersonaAdaptive`` overlay (the state contract)
- the asymmetric-trust gate (signature-policy): relax-direction events without an
  ``operator_signature`` are rejected (the state contract)
- the cause-chain verification hook (chain-verification): events whose
  ``triggering_cause_ids`` cannot be verified are rejected. The resolver accepts a verifier callback; callers must provide
  an appropriate Ed25519 verifier at the trust boundary.
- the ``applied=False`` skip (event application flag):
  events written with ``applied=False`` (auto-emit clamped at the
  safety floor) stay in the audit chain but the fold ignores
  them, so resolver state does not move past the floor.
- the ``frozen`` parameter (the state contract): an operator-driven
  kill-switch that makes the resolver behave as ``(core, [])`` while
  events stay in episodic.db unchanged for reapplication on unfreeze.

Out of scope here (per child issue split):
- Real Ed25519 chain verify (supplies the verifier callable)
- Auto-emit hooks
- The runtime-state freeze flag itself lives in
  :mod:`aeros.runtime.persona_governance`; this resolver only consumes
  the boolean (so the resolver stays pure / IO-free).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Container, Sequence
from typing import TYPE_CHECKING

from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_event import (
    PersonaAdaptiveEvent,
    is_signature_required,
)

if TYPE_CHECKING:
    from aeros.audit.chain import SignedPersonaEvent
    from aeros.runtime.persona import PersonaProfile
    from aeros.runtime.persona_core import PersonaCore


class ResolverRejection(ValueError):
    """Resolver refused to fold an event (signature-policy / chain-verification violation).

    Two failure modes feed this:
      - relax-direction event without ``operator_signature`` (signature-policy)
      - cause-chain verifier returned False (chain-verification)
    The resolver itself is pure; the caller (route handler / CLI)
    chooses how to surface the rejection (log, 403, alert).
    """


# Cause-chain verifier: pure predicate over an event. The real Ed25519
# implementation is supplied externally; the base resolver + tests use the default
# permissive verifier so fold behaviour can be exercised without crypto
# setup. chain-verification is enforced regardless: the resolver always calls the
# verifier and rejects on False, so wiring the real verifier flips
# enforcement on without any resolver-side change.
CauseChainVerifier = Callable[[PersonaAdaptiveEvent], bool]


def _accept_all(_event: PersonaAdaptiveEvent) -> bool:
    return True


# the state contract maps the discrete ``PersonaCore.risk_appetite`` to a numeric
# cap on ``PersonaAdaptive.risk_appetite_actual``. The numbers are v1
# initial proposals; operational data may calibrate them.
_RISK_APPETITE_CAPS: dict[str, float] = {
    "conservative": 0.4,
    "balanced": 0.7,
    "aggressive": 1.0,
}


def _risk_cap(core: "PersonaCore") -> float:
    return _RISK_APPETITE_CAPS.get(core.risk_appetite, 0.7)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply_event(
    adaptive: PersonaAdaptive,
    event: PersonaAdaptiveEvent,
    risk_cap: float,
) -> PersonaAdaptive:
    """Return a new ``PersonaAdaptive`` with ``event``'s delta applied.

    Pure: never mutates ``adaptive``; copies the affected dict before
    writing the new entry.
    """
    kind = event.kind
    delta = event.delta

    if kind == "skill_confidence_decrease":
        cap = event.capability or ""
        magnitude = float(delta["magnitude"])
        new_conf = dict(adaptive.skill_confidence)
        # Skills not yet observed default to confidence 1.0; first
        # decrease clamps them down by ``magnitude``. Hard floor 0.0;
        # safety floor clamps with applied=False higher up.
        current = new_conf.get(cap, 1.0)
        new_conf[cap] = _clamp(current - magnitude, 0.0, 1.0)
        return dataclasses.replace(adaptive, skill_confidence=new_conf)

    if kind == "skill_confidence_increase":
        cap = event.capability or ""
        magnitude = float(delta["magnitude"])
        new_conf = dict(adaptive.skill_confidence)
        current = new_conf.get(cap, 1.0)
        new_conf[cap] = _clamp(current + magnitude, 0.0, 1.0)
        return dataclasses.replace(adaptive, skill_confidence=new_conf)

    if kind == "risk_appetite_calibration_tighten":
        magnitude = float(delta["magnitude"])
        new_value = _clamp(adaptive.risk_appetite_actual - magnitude, 0.0, risk_cap)
        return dataclasses.replace(adaptive, risk_appetite_actual=new_value)

    if kind == "risk_appetite_calibration_relax":
        magnitude = float(delta["magnitude"])
        new_value = _clamp(adaptive.risk_appetite_actual + magnitude, 0.0, risk_cap)
        return dataclasses.replace(adaptive, risk_appetite_actual=new_value)

    if kind in {"human_check_threshold_relax", "human_check_threshold_tighten"}:
        action_kind = str(delta["action_kind"])
        magnitude = float(delta["magnitude"])
        new_thresh = dict(adaptive.human_check_threshold)
        # No prior entry = inherit from PersonaCore at the resolver's
        # read time; for the fold step we treat that as 0.5 (neutral
        # midpoint) so a single event has visible effect.
        current = new_thresh.get(action_kind, 0.5)
        signed = -1.0 if kind == "human_check_threshold_relax" else 1.0
        new_thresh[action_kind] = _clamp(current + signed * magnitude, 0.0, 1.0)
        return dataclasses.replace(adaptive, human_check_threshold=new_thresh)

    if kind == "drift_event":
        # Drift is an informational tighten signal (the state contract).
        # v1 nudges ``risk_appetite_actual`` down by a severity-mapped
        # amount and records the reason in ``tendencies`` so operators
        # have a single dict-shaped surface for adaptive context the
        # numeric fields don't capture.
        severity = str(delta["severity"])
        nudge = {"low": 0.05, "medium": 0.10, "high": 0.20}.get(severity, 0.05)
        new_value = _clamp(adaptive.risk_appetite_actual - nudge, 0.0, risk_cap)
        new_tendencies = dict(adaptive.tendencies)
        # Append rather than overwrite: drift events are evidentiary,
        # so the chain matters. Trim to the last 8 entries to bound
        # state size (matches the state contract default chain depth).
        history = list(new_tendencies.get("drift_history", ()))
        history.append({"severity": severity, "reason": str(delta.get("reason", ""))})
        new_tendencies["drift_history"] = history[-8:]
        return dataclasses.replace(
            adaptive,
            risk_appetite_actual=new_value,
            tendencies=new_tendencies,
        )

    # Unreachable: the Pydantic model rejects unknown ``kind`` values at
    # construction time. Defensive return keeps the resolver pure if a
    # future kind is added without updating the fold.
    return adaptive


def _gate_event(
    event: PersonaAdaptiveEvent,
    verifier: CauseChainVerifier,
) -> None:
    """Raise ``ResolverRejection`` on signature-policy or chain-verification violations.

    signature-policy (the state contract): events whose kind is in the signature-required set
    must carry an ``operator_signature``. The signature itself is
    checked by the verifier; the base resolver enforces presence only.
    chain-verification: every event must verify the cause_id chain. The verifier is
    pluggable so a caller can wire real Ed25519; the default permissive
    verifier accepts everything for unit tests of pure-fold behaviour.
    """
    if is_signature_required(event.kind) and not event.operator_signature:
        raise ResolverRejection(
            f"event kind={event.kind!r} requires operator_signature (signature-policy)"
        )
    if not verifier(event):
        raise ResolverRejection(
            f"event kind={event.kind!r} cause_id chain verification failed (chain-verification)"
        )


def effective_persona(
    core: "PersonaCore",
    events: Sequence[PersonaAdaptiveEvent] = (),
    *,
    verifier: CauseChainVerifier | None = None,
    frozen: bool = False,
) -> "PersonaProfile":
    """Fold ``events`` over ``core`` and return the runtime read-model.

    Fold semantics:
      1. ``frozen=True`` (the state contract) returns
         ``PersonaProfile.from_core(core)`` and skips event processing
         entirely — gate, fold, and verifier are bypassed. Events stay
         in episodic.db; an unfreeze re-runs the fold.
      2. Gate every event up front via ``_gate_event`` (signature-policy + chain-verification).
         Rejection is fail-closed: ``ResolverRejection`` aborts the fold.
      3. Skip events with ``applied=False`` (Q5 floor / rate-limit clamp).
      4. Apply the remaining events in input order to a fresh
         ``PersonaAdaptive``.
      5. Build a ``PersonaProfile`` from ``(core, adaptive)``.

    The fold is pure: same inputs → same output, no I/O, no time
    coupling. Latency budget is the state contract (resolver p99 < 25 ms), met by
    the fold being O(len(events)) over a window the rate limit
    keeps small.
    """
    from aeros.runtime.persona import PersonaProfile
    from aeros.runtime.persona_metrics import (
        get_registry,
        record_adaptive_state,
        time_resolver_fold,
    )

    if frozen:
        return PersonaProfile.from_core(core)

    verify = verifier or _accept_all
    cap = _risk_cap(core)

    registry = get_registry()
    adaptive = PersonaAdaptive()
    with time_resolver_fold():
        for event in events:
            try:
                _gate_event(event, verify)
            except ResolverRejection as exc:
                # signature-policy / chain-verification reasons are encoded in the message tail:
                # we extract a stable tag for the metric label so a
                # dashboard can split signature rejections from chain
                # verification failures.
                reason = "signature_missing" if "signature-policy" in str(exc) else "chain_invalid"
                registry.inc_signature_rejection(reason)
                raise
            if not event.applied:
                continue
            adaptive = _apply_event(adaptive, event, cap)

    record_adaptive_state(adaptive)
    return PersonaProfile.from_core(core, adaptive)


def effective_persona_signed(
    core: "PersonaCore",
    signed_events: "Sequence[SignedPersonaEvent]" = (),
    *,
    trusted_pubkeys: Container[str],
    max_depth: int | None = None,
    frozen: bool = False,
) -> "PersonaProfile":
    """Resolver entry point for callers that already hold signed envelopes.

    Builds the cause-chain verifier from ``signed_events`` (the
    ``aeros.audit.make_chain_verifier``) and folds the events through
    :func:`effective_persona`. ``max_depth`` defaults to the state contract
    (``DEFAULT_MAX_CHAIN_DEPTH``). ``frozen`` is forwarded so the route
    handlers can wire the runtime-state freeze flag without rebuilding
    the verifier when frozen=True.

    ``trusted_pubkeys`` is a container of accepted hex-encoded Ed25519
    public keys for this host (typically
    ``set(load_trusted_pubkeys(~/.aeros/trusted_keys).keys())``). The
    parameter is required so the chain verifier rejects rows whose
    signer is not in the trusted set — chain-verification source-identity
    verification. Without it, a freshly-generated keypair could sign
    arbitrary adaptive deltas and the resolver would fold them.

    The route handlers and operator CLIs call this
    helper rather than constructing the verifier themselves, so chain-verification
    enforcement — both signature-correctness and signer-trust —
    stays in one place.
    """
    from aeros.audit.chain import DEFAULT_MAX_CHAIN_DEPTH, make_chain_verifier

    if frozen:
        return effective_persona(core, (), frozen=True)
    depth = DEFAULT_MAX_CHAIN_DEPTH if max_depth is None else max_depth
    verifier = make_chain_verifier(
        list(signed_events),
        trusted_pubkeys=trusted_pubkeys,
        max_depth=depth,
    )
    events = [se.event for se in signed_events]
    return effective_persona(core, events, verifier=verifier)
