"""Opt-in authoritative adaptive state; NOT integrated into the existing runtime.

A SQLite FULL-synchronous transaction appends an Ed25519 ordered event, advances
an operator-signed checkpoint/tail manifest and prunes the tail atomically.
Eviction folds the oldest event into the checkpoint using the existing real
``_apply_event`` transition. Reads replay only the retained tail. Every envelope
binds a domain/version, agent, core hash, sequence, previous digest, key ID and
role, and the entire event payload (including applied). Automatic keys may sign
only AUTOMATIC_TIGHTEN_KINDS; operator keys may sign all seven kinds. The legacy
operator_signature string is not authority here; the outer signature is.

Callers must retain the returned Anchor independently, monotonically, and supply
it on every read/write/export/import. Exact anchor matching deliberately fails
closed on stale/concurrent calls and on uncertain commits; it is not an anchor
service. With trusted uncompromised keys and a current external anchor, altered,
replayed, reordered, dropped or rolled-back histories are rejected. Whole DB AND
anchor rollback is UNDETECTABLE without an external monotonic trusted anchor.
The checkpoint operator is trusted to attest evicted state: deleted historical
signatures cannot be independently re-audited. Signatures prove authorization,
not cause truth, behavioral safety, floors, rate limits, or adaptation quality.

Creation and restoration are explicit and never overwrite an existing store.
Only event-tail count is bounded; distinct skill/action state can grow (up to
schema limits). JSON only, closed schemas, finite bounded numbers; no pickle or
eval. SQLite durability assumes its filesystem and hardware honor sync. External
anchor advancement is not atomic with SQLite; after an uncertain commit callers
need independently trusted recovery, never silently adopt the DB's own head.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from aeros.runtime.persona_adaptive import PersonaAdaptive
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import AUTOMATIC_TIGHTEN_KINDS, PersonaAdaptiveEvent
from aeros.runtime.persona_resolver import _apply_event, _risk_cap

DOMAIN = "aeros.checkpointed-persona.v1"
ZERO = "0" * 64
MAX_JSON = 4 * 1024 * 1024


class IntegrityError(ValueError):
    """Invalid schema, authorization, history, or external anchor."""


def _json(value: Any) -> str:
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise IntegrityError("not canonical JSON") from exc
    if len(result) > MAX_JSON:
        raise IntegrityError("JSON size limit")
    return result


def _load(raw: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise IntegrityError("duplicate JSON key")
            result[key] = value
        return result
    if type(raw) is not str or len(raw) > MAX_JSON:
        raise IntegrityError("invalid JSON input")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(IntegrityError("nonfinite JSON")))
    except (ValueError, TypeError, RecursionError) as exc:
        raise IntegrityError("invalid JSON") from exc


def _fields(obj: Any, names: str) -> None:
    if type(obj) is not dict or set(obj) != set(names.split()):
        raise IntegrityError("unexpected schema fields")


def _text(value: Any) -> None:
    if type(value) is not str or not value or len(value) > 4096:
        raise IntegrityError("invalid text")


def _seq(value: Any) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise IntegrityError("invalid sequence")


def _digest(value: Any) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise IntegrityError("invalid digest")


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _number(value: Any, cap: float = 1.0) -> None:
    if type(value) not in (int, float) or not 0 <= value <= cap or not math.isfinite(value):
        raise IntegrityError("invalid bounded number")


def _state(value: Any, cap: float) -> PersonaAdaptive:
    _fields(value, "risk_appetite_actual skill_confidence human_check_threshold tendencies")
    _number(value["risk_appetite_actual"], cap)
    for name in ("skill_confidence", "human_check_threshold"):
        entries = value[name]
        if type(entries) is not dict or len(entries) > 4096:
            raise IntegrityError("invalid state map")
        for key, number in entries.items():
            _text(key)
            _number(number)
    tendencies = value["tendencies"]
    if type(tendencies) is not dict or set(tendencies) - {"drift_history"}:
        raise IntegrityError("invalid tendencies")
    history = tendencies.get("drift_history", [])
    if type(history) is not list or len(history) > 8:
        raise IntegrityError("invalid drift history")
    for entry in history:
        _fields(entry, "severity reason")
        if entry["severity"] not in ("low", "medium", "high"):
            raise IntegrityError("invalid severity")
        _text(entry["reason"])
    return PersonaAdaptive(**value)


def _event(payload: Any) -> PersonaAdaptiveEvent:
    _fields(payload, "kind capability delta triggering_cause_ids operator_signature timestamp schema_version applied")
    if type(payload["applied"]) is not bool or type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise IntegrityError("invalid event version/applied")
    _text(payload["kind"])
    _text(payload["timestamp"])
    for name in ("capability", "operator_signature"):
        if payload[name] is not None:
            _text(payload[name])
    causes = payload["triggering_cause_ids"]
    if type(causes) is not list or len(causes) > 128:
        raise IntegrityError("invalid causes")
    for cause in causes:
        _text(cause)
    delta = payload["delta"]
    if type(delta) is not dict:
        raise IntegrityError("invalid delta")
    if "magnitude" in delta:
        _number(delta["magnitude"], 0.5)
    for key in ("reason", "action_kind", "severity"):
        if key in delta:
            _text(delta[key])
    try:
        event = PersonaAdaptiveEvent.model_validate(payload)
    except ValueError as exc:
        raise IntegrityError("invalid typed event") from exc
    if event.timestamp.tzinfo is None:
        raise IntegrityError("timestamp must have timezone")
    return event


@dataclass(frozen=True)
class Anchor:
    """Caller-owned trust input; keep outside the rollback domain of the DB."""
    agent: str
    core_hash: str
    seq: int
    digest: str

    def __post_init__(self) -> None:
        _text(self.agent)
        _digest(self.core_hash)
        _seq(self.seq)
        _digest(self.digest)


@dataclass(frozen=True)
class TrustedKey:
    public_key: Ed25519PublicKey
    role: str

    def __post_init__(self) -> None:
        if self.role not in ("automatic", "operator") or not isinstance(self.public_key, Ed25519PublicKey):
            raise IntegrityError("invalid trust entry")


def _sign(body: dict, key_id: str, key: Ed25519PrivateKey) -> dict:
    _text(key_id)
    return {"body": body, "key_id": key_id, "signature": key.sign(_json({"key_id": key_id, "body": body}).encode()).hex()}


def _verify(envelope: Any, keys: Mapping[str, TrustedKey], role: str | None = None) -> dict:
    _fields(envelope, "body key_id signature")
    _text(envelope["key_id"])
    _text(envelope["signature"])
    trusted = keys.get(envelope["key_id"])
    if trusted is None or (role is not None and trusted.role != role):
        raise IntegrityError("untrusted key or wrong role")
    try:
        trusted.public_key.verify(bytes.fromhex(envelope["signature"]), _json({"key_id": envelope["key_id"], "body": envelope["body"]}).encode())
    except (ValueError, InvalidSignature) as exc:
        raise IntegrityError("invalid Ed25519 signature") from exc
    return envelope["body"]


class CheckpointedPersonaStore:
    """One agent/core per database. Public reads always require an exact anchor.

    Use ``create`` once, ``open`` on restart, ``sign_event`` then ``append``
    for mutations, and ``export``/``restore`` for explicit migration. Each append
    requires an operator checkpoint signer as well as the event's own signature;
    this minimal API is not an unattended automatic-checkpoint key service.
    """

    def __init__(self, path: str | Path, agent: str, core: PersonaCore,
                 keys: Mapping[str, TrustedKey], retention: int = 64) -> None:
        _text(agent)
        if type(retention) is not int or not 1 <= retention <= 4096:
            raise IntegrityError("retention must be 1..4096")
        # PersonaCore is a plain dataclass: validate rather than trust annotations.
        p = core.to_manifest_payload()
        for field, allowed in {"risk_appetite": ("conservative", "balanced", "aggressive"),
                               "human_check": ("never", "on_uncertainty", "on_high_risk", "always"),
                               "value_ordering": ("human_first", "task_first", "efficiency_first"),
                               "verbosity": ("terse", "balanced", "verbose")}.items():
            if p[field] not in allowed:
                raise IntegrityError("invalid core enum")
        _text(p["name"])
        if type(p["description"]) is not str or len(p["description"]) > 4096:
            raise IntegrityError("invalid description")
        if type(p["accept_unsoaked_ecm"]) is not bool or type(p["min_ecm_soak_days"]) is not int or p["min_ecm_soak_days"] < 0:
            raise IntegrityError("invalid core policy")
        if type(core.tags) is not tuple or len(core.tags) > 128:
            raise IntegrityError("invalid core tags")
        for tag in core.tags:
            _text(tag)
        self.agent, self.core_hash = agent, _hash(p)
        self.cap, self.retention = _risk_cap(core), retention
        self.keys = dict(keys)
        for key_id, trust in self.keys.items():
            _text(key_id)
            if not isinstance(trust, TrustedKey):
                raise IntegrityError("invalid trust map")
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=5000")

    @classmethod
    def create(cls, path: str | Path, agent: str, core: PersonaCore,
               keys: Mapping[str, TrustedKey], *, checkpoint_key_id: str,
               checkpoint_key: Ed25519PrivateKey, retention: int = 64) -> tuple[CheckpointedPersonaStore, Anchor]:
        store = cls(path, agent, core, keys, retention)
        try:
            state = dataclasses.asdict(PersonaAdaptive(risk_appetite_actual=min(0.5, store.cap)))
            genesis = _hash({"domain": DOMAIN, "agent": agent, "core_hash": store.core_hash, "initial_state": state})
            anchor = Anchor(agent, store.core_hash, 0, genesis)
            body = {"domain": DOMAIN, "agent": agent, "core_hash": store.core_hash,
                    "base_seq": 0, "base_digest": genesis, "state": state,
                    "tail": [], "head_seq": 0, "head_digest": genesis, "retention": retention}
            signed = _sign(body, checkpoint_key_id, checkpoint_key)
            store._validate(signed, anchor)
            store._install(signed)
            return store, anchor
        except Exception:
            store.close()
            raise

    @classmethod
    def open(cls, path: str | Path, agent: str, core: PersonaCore,
             keys: Mapping[str, TrustedKey], *, anchor: Anchor, retention: int = 64) -> CheckpointedPersonaStore:
        if not Path(path).is_file():
            raise IntegrityError("store missing; explicit create/restore required")
        store = cls(path, agent, core, keys, retention)
        try:
            store.read(anchor)
            return store
        except Exception:
            store.close()
            raise

    def _install(self, signed: dict) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("CREATE TABLE manifest (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL)")
                self._db.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, json TEXT NOT NULL)")
                self._db.execute("INSERT INTO manifest VALUES (1, ?)", (_json(signed),))
                for event in signed["body"]["tail"]:
                    self._db.execute("INSERT INTO events VALUES (?, ?)", (event["body"]["seq"], _json(event)))
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _binding(self, body: dict) -> None:
        if body["domain"] != DOMAIN or body["agent"] != self.agent or body["core_hash"] != self.core_hash:
            raise IntegrityError("domain/agent/core mismatch")

    def _step(self, envelope: dict, seq: int, digest: str, state: PersonaAdaptive) -> tuple[int, str, PersonaAdaptive]:
        body = _verify(envelope, self.keys)
        _fields(body, "domain agent core_hash seq previous role payload")
        self._binding(body)
        _seq(body["seq"])
        _digest(body["previous"])
        if body["seq"] != seq + 1 or body["previous"] != digest:
            raise IntegrityError("replay/reorder/drop or previous digest mismatch")
        role = self.keys[envelope["key_id"]].role
        if body["role"] != role:
            raise IntegrityError("wrong declared role")
        event = _event(body["payload"])
        if role == "automatic" and event.kind not in AUTOMATIC_TIGHTEN_KINDS:
            raise IntegrityError("operator event signed by automatic key")
        updated = _apply_event(state, event, self.cap) if event.applied else state
        _state(dataclasses.asdict(updated), self.cap)
        return body["seq"], _hash(envelope), updated

    def _validate(self, signed: dict, anchor: Anchor) -> PersonaAdaptive:
        if not isinstance(anchor, Anchor) or anchor.agent != self.agent or anchor.core_hash != self.core_hash:
            raise IntegrityError("external anchor binding mismatch")
        body = _verify(signed, self.keys, "operator")
        _fields(body, "domain agent core_hash base_seq base_digest state tail head_seq head_digest retention")
        self._binding(body)
        for name in ("base_seq", "head_seq"):
            _seq(body[name])
        for name in ("base_digest", "head_digest"):
            _digest(body[name])
        if type(body["retention"]) is not int or body["retention"] != self.retention:
            raise IntegrityError("retention mismatch")
        tail = body["tail"]
        if type(tail) is not list or len(tail) > self.retention:
            raise IntegrityError("invalid tail")
        state = _state(body["state"], self.cap)
        seq, digest = body["base_seq"], body["base_digest"]
        if seq == 0:
            initial = dataclasses.asdict(PersonaAdaptive(risk_appetite_actual=min(0.5, self.cap)))
            genesis = _hash({"domain": DOMAIN, "agent": self.agent, "core_hash": self.core_hash, "initial_state": initial})
            if body["state"] != initial or digest != genesis:
                raise IntegrityError("invalid genesis checkpoint")
        for event in tail:
            seq, digest, state = self._step(event, seq, digest, state)
        if (seq, digest) != (body["head_seq"], body["head_digest"]) or (seq, digest) != (anchor.seq, anchor.digest):
            raise IntegrityError("head differs from trusted external anchor")
        return state

    def _read(self, anchor: Anchor) -> tuple[dict, PersonaAdaptive]:
        rows = self._db.execute("SELECT id, json FROM manifest").fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise IntegrityError("missing/extra manifest")
        signed = _load(rows[0][1])
        state = self._validate(signed, anchor)
        rows = self._db.execute("SELECT seq, json FROM events ORDER BY seq").fetchall()
        expected = [(event["body"]["seq"], _json(event)) for event in signed["body"]["tail"]]
        if rows != expected:
            raise IntegrityError("event table differs from signed manifest")
        return signed, state

    def read(self, anchor: Anchor) -> PersonaAdaptive:
        with self._lock:
            self._db.execute("BEGIN")
            try:
                _, state = self._read(anchor)
                self._db.commit()
                return state
            except Exception:
                self._db.rollback()
                raise

    def sign_event(self, event: PersonaAdaptiveEvent, anchor: Anchor, *, key_id: str,
                   key: Ed25519PrivateKey) -> dict:
        """Prepare an envelope; does not commit or advance the external anchor."""
        self.read(anchor)
        trust = self.keys.get(key_id)
        if trust is None:
            raise IntegrityError("unknown signer")
        if type(event) is not PersonaAdaptiveEvent or not isinstance(event.timestamp, datetime):
            raise IntegrityError("expected typed event with datetime")
        if not isinstance(event.delta, Mapping) or type(event.triggering_cause_ids) is not tuple:
            raise IntegrityError("invalid event containers")
        # Read raw attributes: Pydantic's int serializer can turn bool into 1,
        # masking invalid model_copy/model_construct inputs before validation.
        payload = {name: getattr(event, name) for name in (
            "kind", "capability", "operator_signature", "schema_version", "applied")}
        payload.update(delta=dict(event.delta), triggering_cause_ids=list(event.triggering_cause_ids),
                       timestamp=event.timestamp.isoformat())
        _event(payload)
        body = {"domain": DOMAIN, "agent": self.agent, "core_hash": self.core_hash,
                "seq": anchor.seq + 1, "previous": anchor.digest, "role": trust.role, "payload": payload}
        envelope = _sign(body, key_id, key)
        self._step(envelope, anchor.seq, anchor.digest, self.read(anchor))
        return envelope

    def append(self, envelope: dict, anchor: Anchor, *, checkpoint_key_id: str,
               checkpoint_key: Ed25519PrivateKey) -> Anchor:
        # Detach caller-owned mutable structures before verifying and persisting.
        envelope = _load(_json(envelope))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                signed, state = self._read(anchor)
                seq, digest, _ = self._step(envelope, anchor.seq, anchor.digest, state)
                body = signed["body"]
                body["tail"].append(envelope)
                if len(body["tail"]) > self.retention:
                    oldest = body["tail"].pop(0)
                    base = _state(body["state"], self.cap)
                    bs, bd, base = self._step(oldest, body["base_seq"], body["base_digest"], base)
                    body.update(base_seq=bs, base_digest=bd, state=dataclasses.asdict(base))
                body.update(head_seq=seq, head_digest=digest)
                new_anchor = Anchor(self.agent, self.core_hash, seq, digest)
                signed = _sign(body, checkpoint_key_id, checkpoint_key)
                self._validate(signed, new_anchor)
                self._db.execute("INSERT INTO events VALUES (?, ?)", (seq, _json(envelope)))
                self._db.execute("UPDATE manifest SET json=? WHERE id=1", (_json(signed),))
                self._db.execute("DELETE FROM events WHERE seq<=?", (body["base_seq"],))
                self._db.commit()
                return new_anchor
            except Exception:
                self._db.rollback()
                raise

    def export(self, anchor: Anchor) -> str:
        """Return signed checkpoint+tail manifest, verified against caller anchor."""
        with self._lock:
            self._db.execute("BEGIN")
            try:
                signed, _ = self._read(anchor)
                result = _json(signed)
                self._db.commit()
                return result
            except Exception:
                self._db.rollback()
                raise

    @classmethod
    def restore(cls, path: str | Path, agent: str, core: PersonaCore,
                keys: Mapping[str, TrustedKey], bundle: str, *, anchor: Anchor,
                retention: int = 64) -> CheckpointedPersonaStore:
        store = cls(path, agent, core, keys, retention)
        try:
            signed = _load(bundle)
            store._validate(signed, anchor)
            store._install(signed)
            return store
        except Exception:
            store.close()
            raise

    def close(self) -> None:
        self._db.close()
