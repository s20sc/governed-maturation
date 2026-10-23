"""Active Dreaming — offline reflection (the state contract).

Once per off-peak window (default 24h cap, reflection-rate-limit) the runner replays the
recent episodic ring + Layer 2 ``semantic_facts`` through an LLM
counterfactual reflector. The LLM proposes at most three tighten-direction
deltas; an EMA validation gate (capability-keyed failure-rate trend) drops
any whose direction disagrees with the episodic trend; the survivors emit
through :class:`PersonaAdaptationEmitter` so the existing §6.3 / §7.2 /
§8.3 governance gates apply unchanged. Either way, a single
``reflection_summary`` audit row lands in episodic.db so the operator can
see what the LLM saw, what it proposed, and which proposals were rejected.

Module surface (sub A/6):

* :class:`ActiveDreamReflectionEvent` — frozen Pydantic record persisted
  as a ``kind="reflection_summary"`` row. Per the long-form spec §3 it is
  NOT a ``PersonaAdaptiveEvent`` kind; reflection-derived typed events are
  separate rows whose ``triggering_cause_ids`` carry the reflection_id.
* :class:`ProposedDelta`, :class:`ValidationDetail` — frozen sub-models.
* :class:`AggregationBackend` — Protocol; v1 ships
  :class:`LLMCounterfactualBackend`.
* :class:`EMAValidationBackend` — capability-keyed EMA + asymmetric-trust
  rejection (relax direction always rejected per tightening-only).
* :class:`ActiveDreamingRunner` — orchestrates the full pass: kill-switch,
  rate-limit, episodic scan, LLM reflect, EMA validate, emit, audit row.

Kill switch (MUST-rollback): ``AEROS_ACTIVE_DREAMING_ENABLED`` defaults to
``"false"``. With the default unset / false the runner is a complete no-op
— no LLM call, no episodic write, no governance evaluation. Production
clusters opt in explicitly, per the 6-12-month default-off decision.

Determinism (deterministic-parameters): every reflection records the model_id, temperature
(must be 0.0), seed (must be 42), prompt_template_version, and
prompt_template_hash. Same input + same template version + same model
produces byte-identical proposed_deltas; the audit row makes that
provable post-hoc.

LLM timeout (reflection-timeout): single reflection is bounded to 55s wall-clock (5s
buffer beneath the 60s reflection-timeout cap). On timeout the audit row still lands
with ``validation_result="all_rejected"`` and ``reflection_text`` carries
the timeout marker so the operator notices. Three consecutive timeouts
flip ``runner.disabled`` so the scheduler (sub B) can stop calling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aeros.audit.signer import AuditSigner
from aeros.marketplace.signing import sign_payload
from aeros.runtime.persistent_agent import PersistentAgent
from aeros.runtime.persona_emit import PersonaAdaptationEmitter
from aeros.runtime.persona_event import (
    AUTOMATIC_TIGHTEN_KINDS,
    PersonaAdaptiveEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Constants                                                                #
# ---------------------------------------------------------------------- #


REFLECTION_SUMMARY_KIND = "reflection_summary"
DRIFT_EVENT_KIND = "drift_event"

ENABLED_ENV = "AEROS_ACTIVE_DREAMING_ENABLED"
LLM_MODEL_ENV = "AEROS_REFLECTOR_LLM"
DEFAULT_MODEL_ID = "glm-4.5-air"

PROMPT_TEMPLATE_VERSION = "1.0.0"
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "reflection_v1.txt"

# Same seed/temperature contract as deterministic-parameters; encoded so the runner refuses
# to start with anything else and the audit row can be verified post-hoc.
REQUIRED_TEMPERATURE = 0.0
REQUIRED_SEED = 42

# 60s reflection-timeout wall-clock budget with 5s buffer for the surrounding
# bookkeeping — episodic scan, EMA computation, signing, persist.
DEFAULT_LLM_TIMEOUT_SECONDS = 55.0

# 24h rate-limit window per reflection-rate-limit: at most one reflection per identity in
# any rolling 24h window. The window slides; the cap is enforced against
# the most recent ``reflection_summary`` row for the identity.
DEFAULT_RATE_LIMIT_SECONDS = 24 * 3600

# ProposedDelta.confidence floor — anything below is auto-rejected even
# before the EMA gate runs (long-form §3.1).
DEFAULT_CONFIDENCE_FLOOR = 0.5

# EMA bands on per-capability failure rate over the reflection's episodic
# window. Above the tighten band we expect tighten proposals to agree;
# below the relax band we treat the trend as relaxing (proposals that
# would tighten further disagree). The bands are deliberately wide — the
# point is to catch direction reversals, not parameter-tune the LLM.
EMA_TIGHTEN_BAND = 0.4
EMA_RELAX_BAND = 0.15

# Three consecutive LLM timeouts → runner self-disables until the bridge
# restarts. The scheduler (sub B) reads ``runner.disabled`` to stop the
# cron loop; this property only flips on timeout strikes, not on an EMA
# all_rejected pass (a healthy LLM that disagrees is still healthy).
TIMEOUT_STRIKES_LIMIT = 3


# ---------------------------------------------------------------------- #
# Schema                                                                   #
# ---------------------------------------------------------------------- #


class ProposedDelta(BaseModel):
    """One tighten-direction delta proposed by a reflection backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_kind: str
    target_capability: str | None = None
    proposed_delta: dict[str, float | int | str | bool] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationDetail(BaseModel):
    """EMA gate decision for a single :class:`ProposedDelta`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_capability: str | None = None
    ema_direction: Literal["tighten", "stable", "relax"] | None = None
    proposed_direction: Literal["tighten", "relax"]
    agreement: bool
    rejected: bool
    rejection_reason: str | None = None


class ActiveDreamReflectionEvent(BaseModel):
    """Frozen audit row written under ``kind="reflection_summary"``.

    NOT a ``PersonaAdaptiveEvent`` kind — long-form spec §3 keeps the
    seven-kind vocabulary closed. Reflection-derived typed events are
    separate ``persona_adaptation`` rows whose ``triggering_cause_ids``
    carry this row's ``reflection_id`` so a Layer 3 export retains the
    causal chain across deployments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["0.1.0"] = "0.1.0"

    reflection_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    triggering_episodic_ids: tuple[str, ...]
    cause_id_chain: tuple[str, ...]

    model_id: str
    temperature: float
    seed: int
    prompt_template_hash: str
    prompt_template_version: str

    reflection_text: str
    proposed_deltas: tuple[ProposedDelta, ...] = Field(default_factory=tuple)
    derived_facts: tuple[str, ...] = Field(default_factory=tuple)
    derived_typed_event_ids: tuple[str, ...] = Field(default_factory=tuple)

    validation_result: Literal["all_passed", "partial", "all_rejected"]
    validation_details: tuple[ValidationDetail, ...] = Field(default_factory=tuple)

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    wall_clock_seconds: float = 0.0

    @field_validator("temperature")
    @classmethod
    def _temperature_must_be_zero(cls, v: float) -> float:
        if v != REQUIRED_TEMPERATURE:
            raise ValueError(
                f"temperature must be {REQUIRED_TEMPERATURE} for determinism "
                f"(deterministic-parameters); got {v}"
            )
        return v

    @field_validator("seed")
    @classmethod
    def _seed_must_be_42(cls, v: int) -> int:
        if v != REQUIRED_SEED:
            raise ValueError(
                f"seed must be {REQUIRED_SEED} for determinism (deterministic-parameters); got {v}"
            )
        return v

    @field_validator("triggering_episodic_ids")
    @classmethod
    def _episodic_ids_non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("triggering_episodic_ids must be non-empty")
        return v


# ---------------------------------------------------------------------- #
# Reflection inputs                                                       #
# ---------------------------------------------------------------------- #


class ReflectionInputs(BaseModel):
    """Materialized window passed to a backend's ``reflect`` call.

    The runner builds this from the agent's episodic memory + Layer 2
    semantic_facts + current persona snapshot. Backends MUST treat it as
    read-only and MUST be deterministic functions of the inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_hash: str
    triggering_episodic_ids: tuple[str, ...]
    episodic_summary: str
    semantic_facts_summary: str
    current_confidences_summary: str
    capability_failure_rates: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Backend protocols + LLM implementation                                   #
# ---------------------------------------------------------------------- #


class AggregationBackend(Protocol):
    """Abstract protocol for the primary reflection backend (long-form §5).

    v1 ships :class:`LLMCounterfactualBackend`; future implementations
    (Bayesian, EMA-only) may plug here without resolver changes per
    forward-additive compat (the state contract).
    """

    model_id: str
    """Backend identifier persisted into the audit row."""

    def reflect(self, inputs: ReflectionInputs) -> "_BackendOutput":
        """Run one reflection pass.

        MUST be deterministic for the same input set. MUST NOT directly
        emit ``PersonaAdaptiveEvent`` rows — the runner is responsible
        for validation and emission.
        """
        ...


class _BackendOutput(BaseModel):
    """Internal carrier for a backend's reflection output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reflection_text: str
    proposed_deltas: tuple[ProposedDelta, ...] = Field(default_factory=tuple)
    derived_facts: tuple[str, ...] = Field(default_factory=tuple)
    input_tokens: int = 0
    output_tokens: int = 0


# Type for an LLM call: takes the rendered prompt and a deadline (seconds
# since epoch); returns the raw JSON response text + token counts.
# Tests inject a stub; production wires the GLM-4.5-Air HTTP adapter.
LLMCall = Callable[[str, float], tuple[str, int, int]]


class LLMCounterfactualBackend:
    """Default :class:`AggregationBackend` — frozen prompt + LLM call.

    The LLM call itself is injected at construction so tests can supply a
    deterministic stub and the production wiring (sub B) can supply the
    GLM-4.5-Air HTTP adapter. ``temperature=0.0`` and ``seed=42`` are
    hardcoded into the audit row regardless of what the backend received,
    so the determinism contract (deterministic-parameters) cannot drift via misconfiguration.
    """

    def __init__(
        self,
        llm_call: LLMCall,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_template_path: Path = DEFAULT_PROMPT_PATH,
        prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self._llm_call = llm_call
        self.model_id = model_id
        self._timeout_seconds = float(timeout_seconds)
        self._prompt_template_path = prompt_template_path
        self._prompt_template_version = prompt_template_version
        self._prompt_template_text = prompt_template_path.read_text(encoding="utf-8")
        self._prompt_template_hash = hashlib.sha256(
            self._prompt_template_text.encode("utf-8")
        ).hexdigest()

    @property
    def prompt_template_hash(self) -> str:
        return self._prompt_template_hash

    @property
    def prompt_template_version(self) -> str:
        return self._prompt_template_version

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def render_prompt(self, inputs: ReflectionInputs) -> str:
        return (
            self._prompt_template_text
            .replace("{{EPISODIC_SUMMARY}}", inputs.episodic_summary)
            .replace("{{SEMANTIC_FACTS}}", inputs.semantic_facts_summary)
            .replace("{{CURRENT_CONFIDENCES}}", inputs.current_confidences_summary)
        )

    def reflect(self, inputs: ReflectionInputs) -> _BackendOutput:
        prompt = self.render_prompt(inputs)
        deadline = time.monotonic() + self._timeout_seconds
        raw_text, in_tok, out_tok = self._llm_call(prompt, deadline)
        proposed, derived_facts, reflection_text = _parse_llm_json(raw_text)
        return _BackendOutput(
            reflection_text=reflection_text,
            proposed_deltas=tuple(proposed),
            derived_facts=tuple(derived_facts),
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
        )


def _parse_llm_json(
    raw: str,
) -> tuple[list[ProposedDelta], list[str], str]:
    """Parse the LLM response into typed proposals.

    Strict: missing keys, wrong types, or a kind not in
    :data:`AUTOMATIC_TIGHTEN_KINDS` cause a ``ValueError`` so the runner
    converts the failure into ``validation_result="all_rejected"``.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    text = payload.get("reflection_text", "")
    if not isinstance(text, str):
        raise ValueError("reflection_text must be a string")
    deltas_raw = payload.get("proposed_deltas", [])
    if not isinstance(deltas_raw, list):
        raise ValueError("proposed_deltas must be a list")
    facts_raw = payload.get("derived_facts", [])
    if not isinstance(facts_raw, list):
        raise ValueError("derived_facts must be a list")
    deltas: list[ProposedDelta] = []
    for raw_delta in deltas_raw:
        if not isinstance(raw_delta, dict):
            raise ValueError("each proposed delta must be an object")
        kind = raw_delta.get("target_kind")
        if kind not in AUTOMATIC_TIGHTEN_KINDS:
            # tightening-only + spec §3: only tighten kinds may be proposed. A
            # misbehaving LLM that emits a relax kind is rejected at
            # parse time so the EMA gate never has to see it.
            raise ValueError(
                f"target_kind={kind!r} is not in the tighten allowlist"
            )
        deltas.append(ProposedDelta(**raw_delta))
    return deltas, [str(f) for f in facts_raw], text


# ---------------------------------------------------------------------- #
# EMA validation backend                                                   #
# ---------------------------------------------------------------------- #


def _proposed_direction(target_kind: str) -> Literal["tighten", "relax"]:
    """Map an ``EventKind`` to its asymmetric-trust direction."""
    return "tighten" if target_kind in AUTOMATIC_TIGHTEN_KINDS else "relax"


def _ema_band(rate: float) -> Literal["tighten", "stable", "relax"]:
    """Classify a per-capability failure rate into a trend band."""
    if rate >= EMA_TIGHTEN_BAND:
        return "tighten"
    if rate <= EMA_RELAX_BAND:
        return "relax"
    return "stable"


class EMAValidationBackend:
    """Sanity-check gate for LLM-proposed deltas (long-form §5).

    Per-capability failure-rate is classified into one of three bands; a
    tighten proposal is accepted only when the EMA already trends tighten
    (tightening-only + asymmetric trust §6.3). A relax proposal is always rejected
    even if the LLM somehow slipped one past the parse-time allowlist.
    Confidence below :data:`DEFAULT_CONFIDENCE_FLOOR` is rejected before
    the band check runs.
    """

    def __init__(
        self,
        *,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    ) -> None:
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError(
                f"confidence_floor must be in [0, 1]; got {confidence_floor}"
            )
        self._confidence_floor = confidence_floor

    def validate(
        self,
        proposed: Sequence[ProposedDelta],
        *,
        capability_failure_rates: dict[str, float],
    ) -> list[ValidationDetail]:
        out: list[ValidationDetail] = []
        for delta in proposed:
            direction = _proposed_direction(delta.target_kind)
            cap = delta.target_capability
            rate = capability_failure_rates.get(cap or "", 0.0)
            ema_dir = _ema_band(rate) if cap else None

            if direction == "relax":
                out.append(
                    ValidationDetail(
                        target_capability=cap,
                        ema_direction=ema_dir,
                        proposed_direction=direction,
                        agreement=False,
                        rejected=True,
                        rejection_reason="relax_direction_requires_operator_signature",
                    )
                )
                continue

            if delta.confidence < self._confidence_floor:
                out.append(
                    ValidationDetail(
                        target_capability=cap,
                        ema_direction=ema_dir,
                        proposed_direction=direction,
                        agreement=False,
                        rejected=True,
                        rejection_reason=(
                            f"confidence {delta.confidence:.2f} below floor "
                            f"{self._confidence_floor:.2f}"
                        ),
                    )
                )
                continue

            # No capability key (e.g. risk_appetite_calibration_tighten or
            # drift_event) — EMA can't disambiguate a per-skill trend, so
            # we accept on the LLM's confidence alone. The downstream
            # safety floor in persona_governance still bounds the effect.
            if cap is None:
                out.append(
                    ValidationDetail(
                        target_capability=None,
                        ema_direction=None,
                        proposed_direction=direction,
                        agreement=True,
                        rejected=False,
                        rejection_reason=None,
                    )
                )
                continue

            agreement = ema_dir == "tighten"
            out.append(
                ValidationDetail(
                    target_capability=cap,
                    ema_direction=ema_dir,
                    proposed_direction=direction,
                    agreement=agreement,
                    rejected=not agreement,
                    rejection_reason=(
                        None
                        if agreement
                        else f"ema_direction={ema_dir} disagrees with proposed=tighten"
                    ),
                )
            )
        return out


# ---------------------------------------------------------------------- #
# Runner                                                                   #
# ---------------------------------------------------------------------- #


class ActiveDreamingRunner:
    """Orchestrates one reflection pass end-to-end.

    Public surface is intentionally narrow: ``run_reflection`` is the
    only entry point. All inputs (kill switch, rate limit, LLM call) are
    plumbed through the constructor so tests can swap each one without
    monkey-patching.
    """

    def __init__(
        self,
        agent: PersistentAgent,
        signer: AuditSigner,
        emitter: PersonaAdaptationEmitter,
        backend: AggregationBackend,
        *,
        ema_backend: EMAValidationBackend | None = None,
        prompt_template_hash: str | None = None,
        prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
        rate_limit_seconds: int = DEFAULT_RATE_LIMIT_SECONDS,
        episodic_window: int = 200,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._agent = agent
        self._signer = signer
        self._emitter = emitter
        self._backend = backend
        self._ema = ema_backend or EMAValidationBackend()
        self._prompt_template_hash = prompt_template_hash or getattr(
            backend, "prompt_template_hash", ""
        )
        self._prompt_template_version = prompt_template_version
        self._rate_limit_seconds = rate_limit_seconds
        self._episodic_window = max(1, episodic_window)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout_strikes = 0
        self._disabled = False
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        """True after three consecutive LLM timeouts (4.1 in infra plan)."""
        return self._disabled

    @property
    def timeout_strikes(self) -> int:
        return self._timeout_strikes

    # ---------------- Public entry ---------------- #

    def run_reflection(
        self, identity_hash: str
    ) -> ActiveDreamReflectionEvent | None:
        """Execute one reflection pass.

        Returns the reflection event when one ran (regardless of EMA
        outcome). Returns None when the kill switch is off, the rate
        limit blocks the pass, the runner is in the timeout-strike
        cooldown, or there is no episodic content to reflect on.
        """
        if not _kill_switch_enabled():
            logger.debug(
                "active dreaming disabled via %s; skipping reflection", ENABLED_ENV
            )
            return None
        with self._lock:
            if self._disabled:
                logger.warning(
                    "active dreaming runner disabled after %d timeout strikes",
                    self._timeout_strikes,
                )
                return None
        if self._reflected_within_rate_limit(identity_hash):
            logger.info(
                "active dreaming rate-limit window not elapsed for %s; skipping",
                identity_hash[:16],
            )
            return None

        episodic_records = self._collect_episodic(identity_hash)
        if not episodic_records:
            return None

        inputs = self._build_inputs(identity_hash, episodic_records)
        wall_start = time.monotonic()
        try:
            backend_out = self._backend.reflect(inputs)
            timed_out = False
            parse_error: str | None = None
        except _LLMTimeout:
            backend_out = None
            timed_out = True
            parse_error = None
            with self._lock:
                self._timeout_strikes += 1
                if self._timeout_strikes >= TIMEOUT_STRIKES_LIMIT:
                    self._disabled = True
                    self._emit_drift_pause_event()
        except (ValueError, KeyError, TypeError) as exc:
            backend_out = None
            timed_out = False
            parse_error = str(exc)
        else:
            with self._lock:
                self._timeout_strikes = 0

        wall_seconds = time.monotonic() - wall_start

        if backend_out is None:
            return self._write_unsuccessful_reflection(
                identity_hash=identity_hash,
                inputs=inputs,
                wall_seconds=wall_seconds,
                timed_out=timed_out,
                parse_error=parse_error,
            )

        return self._finalize_reflection(
            identity_hash=identity_hash,
            inputs=inputs,
            backend_out=backend_out,
            wall_seconds=wall_seconds,
        )

    # ---------------- Internals ---------------- #

    def _collect_episodic(self, identity_hash: str) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(episodic_id, payload)`` pairs for the reflection window.

        We use the agent's bounded ``recent`` slice rather than scanning
        the full DB so the prompt size stays predictable. The episodic_id
        we expose is the timestamp string — durable across restarts and
        unique per row in the in-memory backend.
        """
        records = self._agent.episodic.recent(self._episodic_window)
        out: list[tuple[str, dict[str, Any]]] = []
        for r in records:
            if r.kind == REFLECTION_SUMMARY_KIND:
                continue
            out.append((r.timestamp, {"kind": r.kind, "payload": r.payload}))
        return out

    def _reflected_within_rate_limit(self, identity_hash: str) -> bool:
        rows = self._agent.episodic.by_kind(REFLECTION_SUMMARY_KIND)
        if not rows:
            return False
        cutoff = self._clock() - timedelta(seconds=self._rate_limit_seconds)
        for row in rows:
            # Prefer the payload-recorded ``event.timestamp`` (the runner's
            # injected clock) over the row.timestamp (real wall clock).
            # Tests pin the runner clock; the row insertion path uses
            # ``datetime.now()`` so without this lookup a mocked future
            # clock would still see the wall-clock-stamped row as recent.
            ts: datetime | None = None
            payload = row.payload if isinstance(row.payload, dict) else {}
            payload_ts = payload.get("timestamp")
            if isinstance(payload_ts, str):
                try:
                    ts = datetime.fromisoformat(payload_ts)
                except ValueError:
                    ts = None
            if ts is None:
                try:
                    ts = datetime.fromisoformat(row.timestamp)
                except ValueError:
                    continue
            if ts >= cutoff:
                return True
        return False

    def _build_inputs(
        self,
        identity_hash: str,
        episodic_records: list[tuple[str, dict[str, Any]]],
    ) -> ReflectionInputs:
        episodic_ids = tuple(eid for eid, _ in episodic_records)
        episodic_summary = _summarize_episodic(episodic_records)
        semantic_summary = _summarize_semantic_facts(self._agent, identity_hash)
        confidences_summary = _summarize_confidences(self._agent, episodic_records)
        failure_rates = _capability_failure_rates(episodic_records)
        return ReflectionInputs(
            identity_hash=identity_hash,
            triggering_episodic_ids=episodic_ids,
            episodic_summary=episodic_summary,
            semantic_facts_summary=semantic_summary,
            current_confidences_summary=confidences_summary,
            capability_failure_rates=failure_rates,
        )

    def _finalize_reflection(
        self,
        *,
        identity_hash: str,
        inputs: ReflectionInputs,
        backend_out: _BackendOutput,
        wall_seconds: float,
    ) -> ActiveDreamReflectionEvent:
        validation_details = self._ema.validate(
            backend_out.proposed_deltas,
            capability_failure_rates=inputs.capability_failure_rates,
        )
        validation_result = _classify_validation(validation_details)

        reflection_id = generate_reflection_id()
        cause_id_chain = self._sign_reflection(
            reflection_id=reflection_id,
            triggering_episodic_ids=inputs.triggering_episodic_ids,
        )
        emitted_event_ids = self._emit_passed_deltas(
            reflection_id=reflection_id,
            episodic_ids=inputs.triggering_episodic_ids,
            proposed=backend_out.proposed_deltas,
            details=validation_details,
        )

        event = ActiveDreamReflectionEvent(
            reflection_id=reflection_id,
            timestamp=self._clock(),
            triggering_episodic_ids=inputs.triggering_episodic_ids,
            cause_id_chain=cause_id_chain,
            model_id=self._backend.model_id,
            temperature=REQUIRED_TEMPERATURE,
            seed=REQUIRED_SEED,
            prompt_template_hash=self._prompt_template_hash,
            prompt_template_version=self._prompt_template_version,
            reflection_text=backend_out.reflection_text,
            proposed_deltas=backend_out.proposed_deltas,
            derived_facts=backend_out.derived_facts,
            derived_typed_event_ids=tuple(emitted_event_ids),
            validation_result=validation_result,
            validation_details=tuple(validation_details),
            llm_input_tokens=backend_out.input_tokens,
            llm_output_tokens=backend_out.output_tokens,
            wall_clock_seconds=wall_seconds,
        )
        self._persist_reflection(event)
        return event

    def _write_unsuccessful_reflection(
        self,
        *,
        identity_hash: str,
        inputs: ReflectionInputs,
        wall_seconds: float,
        timed_out: bool,
        parse_error: str | None,
    ) -> ActiveDreamReflectionEvent:
        reflection_id = generate_reflection_id()
        cause_id_chain = self._sign_reflection(
            reflection_id=reflection_id,
            triggering_episodic_ids=inputs.triggering_episodic_ids,
        )
        if timed_out:
            text = "[reflection timeout] LLM did not respond within budget"
        elif parse_error:
            text = f"[reflection parse error] {parse_error}"
        else:
            text = "[reflection unsuccessful]"
        event = ActiveDreamReflectionEvent(
            reflection_id=reflection_id,
            timestamp=self._clock(),
            triggering_episodic_ids=inputs.triggering_episodic_ids,
            cause_id_chain=cause_id_chain,
            model_id=self._backend.model_id,
            temperature=REQUIRED_TEMPERATURE,
            seed=REQUIRED_SEED,
            prompt_template_hash=self._prompt_template_hash,
            prompt_template_version=self._prompt_template_version,
            reflection_text=text,
            proposed_deltas=(),
            derived_facts=(),
            derived_typed_event_ids=(),
            validation_result="all_rejected",
            validation_details=(),
            llm_input_tokens=0,
            llm_output_tokens=0,
            wall_clock_seconds=wall_seconds,
        )
        self._persist_reflection(event)
        return event

    def _sign_reflection(
        self,
        *,
        reflection_id: str,
        triggering_episodic_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        digest = compute_reflection_digest(
            reflection_id=reflection_id,
            triggering_episodic_ids=triggering_episodic_ids,
        )
        signature = sign_payload(self._signer.private_key_bytes, digest)
        return (signature.hex(), *triggering_episodic_ids)

    def _emit_passed_deltas(
        self,
        *,
        reflection_id: str,
        episodic_ids: tuple[str, ...],
        proposed: Sequence[ProposedDelta],
        details: Sequence[ValidationDetail],
    ) -> list[str]:
        emitted: list[str] = []
        for delta, detail in zip(proposed, details):
            if detail.rejected:
                continue
            try:
                event = _build_typed_event(
                    delta=delta,
                    cause_ids=(reflection_id, *episodic_ids),
                    timestamp=self._clock(),
                )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "reflection-derived delta failed PersonaAdaptiveEvent "
                    "validation kind=%s capability=%s: %s",
                    delta.target_kind,
                    delta.target_capability,
                    exc,
                )
                continue
            self._emitter.emit_reflection_derived(
                event,
                trigger=f"active_dreaming:{reflection_id}",
            )
            # Use the reflection_id + capability as the per-emitted-event
            # id; the actual episodic.db row id isn't exposed by the
            # backend, but this string is unique per reflection and
            # enough for an operator to grep both rows together.
            emitted.append(f"{reflection_id}:{delta.target_kind}:{delta.target_capability or '*'}")
        return emitted

    def _persist_reflection(self, event: ActiveDreamReflectionEvent) -> None:
        payload = event.model_dump(mode="json")
        summary = (event.reflection_text or "")[:200]
        self._agent.record(
            kind=REFLECTION_SUMMARY_KIND,
            intent_id=None,
            summary=summary,
            payload=payload,
        )

    def _emit_drift_pause_event(self) -> None:
        """Self-pause notification: three consecutive LLM timeouts."""
        try:
            self._emitter.record_drift_event(
                reason="active_dreaming_paused_three_timeouts",
                severity="high",
            )
        except Exception:  # noqa: BLE001 — drift emit must not poison the runner
            logger.exception("failed to emit active_dreaming pause drift event")


# ---------------------------------------------------------------------- #
# Helpers                                                                  #
# ---------------------------------------------------------------------- #


class _LLMTimeout(Exception):
    """Raised by an injected LLM call when the wall-clock budget expires."""


def _kill_switch_enabled() -> bool:
    raw = os.environ.get(ENABLED_ENV, "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def generate_reflection_id() -> str:
    """Return a UUID v7 string (RFC 9562) for a new reflection row.

    Time-ordered so naive ``ORDER BY reflection_id`` matches arrival
    order; cryptographically-random tail keeps collisions negligible
    across a fleet. Stdlib ``uuid`` does not yet expose v7 (Python 3.12),
    so we build the wire form by hand.
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.token_bytes(10)
    ts = ts_ms.to_bytes(6, "big")
    rand_a_high = (rand[0] & 0x0F) | 0x70
    rand_b_high = (rand[2] & 0x3F) | 0x80
    raw = ts + bytes([rand_a_high, rand[1], rand_b_high]) + rand[3:]
    h = raw.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def compute_reflection_digest(
    *, reflection_id: str, triggering_episodic_ids: tuple[str, ...]
) -> bytes:
    """SHA-256 over the canonical signing surface of a reflection row."""
    payload = json.dumps(
        {
            "reflection_id": reflection_id,
            "triggering_episodic_ids": list(triggering_episodic_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _classify_validation(
    details: Sequence[ValidationDetail],
) -> Literal["all_passed", "partial", "all_rejected"]:
    if not details:
        return "all_rejected"
    rejected = sum(1 for d in details if d.rejected)
    if rejected == 0:
        return "all_passed"
    if rejected == len(details):
        return "all_rejected"
    return "partial"


def _build_typed_event(
    *,
    delta: ProposedDelta,
    cause_ids: tuple[str, ...],
    timestamp: datetime,
) -> PersonaAdaptiveEvent:
    return PersonaAdaptiveEvent(
        kind=delta.target_kind,  # type: ignore[arg-type]
        capability=delta.target_capability,
        delta=dict(delta.proposed_delta),
        triggering_cause_ids=tuple(cause_ids),
        operator_signature=None,
        timestamp=timestamp,
    )


def _capability_failure_rates(
    records: list[tuple[str, dict[str, Any]]],
) -> dict[str, float]:
    """Per-capability failure rate over the reflection window.

    The window is the bounded ``recent`` slice the runner already pulled,
    so the dict is small (≤ episodic_window). Rows without an
    ``outcome``/``capability`` payload are skipped — the runner only
    needs episodic events that LLM reflection can attribute.
    """
    totals: dict[str, int] = {}
    failures: dict[str, int] = {}
    for _, row in records:
        payload = row.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        cap = payload.get("capability") or payload.get("skill_id")
        outcome = payload.get("outcome")
        if not cap or outcome not in ("success", "failure"):
            continue
        totals[cap] = totals.get(cap, 0) + 1
        if outcome == "failure":
            failures[cap] = failures.get(cap, 0) + 1
    return {
        cap: failures.get(cap, 0) / totals[cap]
        for cap in totals
        if totals[cap] > 0
    }


def _summarize_episodic(records: list[tuple[str, dict[str, Any]]]) -> str:
    if not records:
        return "(no episodic rows in window)"
    lines: list[str] = []
    for ts, row in records[-50:]:  # cap prompt size — last 50 rows is plenty
        kind = row.get("kind", "?")
        payload = row.get("payload") or {}
        # Compact one-liner per row; keep keys deterministic for hashing.
        keys = sorted(k for k in payload.keys() if k != "event")
        kv = ", ".join(f"{k}={payload[k]!r}" for k in keys[:6])
        lines.append(f"- {ts} kind={kind} {kv}".rstrip())
    return "\n".join(lines)


def _summarize_semantic_facts(agent: PersistentAgent, identity_hash: str) -> str:
    """Read ``semantic_facts`` rows for the identity, if Layer 2 is wired.

    No-op when the agent's episodic memory uses the in-memory fallback or
    when the ``semantic_facts`` table does not exist (Layer 2 disabled).
    """
    store = agent.episodic.store
    if store is None:
        return "(no semantic_facts; in-memory backend)"
    try:
        with store._lock:  # noqa: SLF001 — share connection lock
            cur = store._conn.execute(  # noqa: SLF001
                "SELECT fact_type, fact_key, confidence, value_json "
                "FROM semantic_facts WHERE identity_hash = ? "
                "ORDER BY confidence DESC LIMIT 50",
                (identity_hash,),
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — Layer 2 may be disabled
        return "(semantic_facts unavailable)"
    if not rows:
        return "(no semantic_facts in window)"
    return "\n".join(
        f"- {ft} {fk} confidence={conf:.2f} value={val}"
        for ft, fk, conf, val in rows
    )


def _summarize_confidences(
    agent: PersistentAgent,
    records: list[tuple[str, dict[str, Any]]],
) -> str:
    """Inline the per-capability success rate from the same window.

    The runner keeps the persona snapshot lightweight — it doesn't
    fold the resolver here (sub B will, once the scheduler decides what
    persona_core to wire). The per-capability rate already tells the LLM
    where to focus.
    """
    rates = _capability_failure_rates(records)
    if not rates:
        return "(no capability outcomes in window)"
    lines = [
        f"- {cap}: failure_rate={rate:.3f}"
        for cap, rate in sorted(rates.items())
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Public re-exports                                                        #
# ---------------------------------------------------------------------- #


__all__ = [
    "ActiveDreamReflectionEvent",
    "ActiveDreamingRunner",
    "AggregationBackend",
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_RATE_LIMIT_SECONDS",
    "EMAValidationBackend",
    "ENABLED_ENV",
    "LLMCounterfactualBackend",
    "PROMPT_TEMPLATE_VERSION",
    "ProposedDelta",
    "REFLECTION_SUMMARY_KIND",
    "REQUIRED_SEED",
    "REQUIRED_TEMPERATURE",
    "ReflectionInputs",
    "TIMEOUT_STRIKES_LIMIT",
    "ValidationDetail",
    "compute_reflection_digest",
    "generate_reflection_id",
]
