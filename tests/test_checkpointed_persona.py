"""Checkpoint recovery, authorization and retention regression tests."""
from __future__ import annotations

import copy
import dataclasses
import json
import sqlite3
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aeros.runtime.checkpointed_persona import (
    Anchor, CheckpointedPersonaStore, IntegrityError, TrustedKey, _sign,
)
from aeros.runtime.episodic_db import EpisodicStore
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_resolver import effective_persona


def event(kind="skill_confidence_decrease", *, applied=True, capability="grasp", delta=None):
    return PersonaAdaptiveEvent(kind=kind, capability=capability,
                                delta=delta or {"magnitude": 0.1}, applied=applied,
                                timestamp=datetime(2026, 10, 21, tzinfo=UTC))


@pytest.fixture
def setup(tmp_path):
    automatic, operator = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    keys = {"auto": TrustedKey(automatic.public_key(), "automatic"),
            "op": TrustedKey(operator.public_key(), "operator")}
    core = PersonaCore(risk_appetite="conservative")
    path = tmp_path / "state.sqlite"
    store, anchor = CheckpointedPersonaStore.create(
        path, "robot-A", core, keys, checkpoint_key_id="op", checkpoint_key=operator, retention=2)
    yield store, anchor, automatic, operator, keys, core, path
    # Some tests explicitly close/reopen; sqlite close is idempotent.
    store.close()


def append(store, anchor, ev, auto, op, *, operator=False):
    envelope = store.sign_event(ev, anchor, key_id="op" if operator else "auto", key=op if operator else auto)
    return store.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=op)


def restore_args(setup, tmp_path, bundle, anchor, **overrides):
    _, _, _, _, keys, core, _ = setup
    args = dict(path=tmp_path / "restored.sqlite", agent="robot-A", core=core,
                keys=keys, bundle=bundle, anchor=anchor, retention=2)
    args.update(overrides)
    return CheckpointedPersonaStore.restore(**args)


def test_old_fifo_fold_loses_evicted_state_negative_baseline(tmp_path):
    old = EpisodicStore(tmp_path / "old.sqlite")
    first = event()
    filler = event("risk_appetite_calibration_tighten")
    for ev in (first, filler, filler):
        old.append("robot-A", timestamp=ev.timestamp.isoformat(), kind=ev.kind,
                   intent_id=None, summary="probe", payload=ev.model_dump(mode="json"))
    before = effective_persona(PersonaCore(), [first, filler, filler]).adaptive.skill_confidence["grasp"]
    old.prune_to_capacity("robot-A", 2)
    surviving = [PersonaAdaptiveEvent.model_validate(row[-1]) for row in old.recent("robot-A", 10)]
    after = effective_persona(PersonaCore(), surviving).adaptive.skill_confidence.get("grasp", 1.0)
    old.close()
    print(f"OBSERVED OLD NEGATIVE PROBE: grasp before={before}, after FIFO eviction={after}")
    assert before == 0.9 and after == 1.0  # Assert the defect, never xfail the final suite.


def test_eviction_restart_export_restore_and_continued_sequence(setup, tmp_path):
    store, anchor, auto, op, keys, core, path = setup
    assert store.read(anchor).risk_appetite_actual == 0.4
    for ev in (event(), event("drift_event", delta={"severity": "low", "reason": "incident"}),
               event("risk_appetite_calibration_tighten"), event(applied=False)):
        anchor = append(store, anchor, ev, auto, op)
    state = store.read(anchor)
    assert state.skill_confidence["grasp"] == 0.9
    assert state.risk_appetite_actual == pytest.approx(0.25)
    assert state.tendencies["drift_history"] == [{"severity": "low", "reason": "incident"}]
    bundle = store.export(anchor)
    body = json.loads(bundle)["body"]
    assert (body["base_seq"], body["head_seq"], len(body["tail"])) == (2, 4, 2)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    store.close()
    reopened = CheckpointedPersonaStore.open(path, "robot-A", core, keys, anchor=anchor, retention=2)
    restored = restore_args(setup, tmp_path, bundle, anchor)
    assert reopened.read(anchor) == restored.read(anchor) == state
    updated = append(restored, anchor, event(), auto, op)
    assert updated.seq == 5
    assert restored.read(updated).skill_confidence["grasp"] == pytest.approx(0.8)
    # Caller mutation cannot change authoritative state.
    state.skill_confidence["grasp"] = 0
    assert reopened.read(anchor).skill_confidence["grasp"] == 0.9
    reopened.close()
    restored.close()


@pytest.mark.parametrize("kind,delta", [
    ("skill_confidence_increase", {"magnitude": 0.1}),
    ("human_check_threshold_relax", {"magnitude": 0.1, "action_kind": "move"}),
    ("human_check_threshold_tighten", {"magnitude": 0.1, "action_kind": "move"}),
    ("risk_appetite_calibration_relax", {"magnitude": 0.1}),
])
def test_role_policy_uses_outer_key_not_legacy_signature(setup, kind, delta):
    store, anchor, auto, op, *_ = setup
    ev = event(kind, delta=delta).model_copy(update={"operator_signature": "forged legacy string"})
    with pytest.raises(IntegrityError, match="operator event"):
        store.sign_event(ev, anchor, key_id="auto", key=auto)
    ev = ev.model_copy(update={"operator_signature": None})
    anchor = append(store, anchor, ev, auto, op, operator=True)
    assert anchor.seq == 1
    assert store.read(anchor).risk_appetite_actual <= 0.4


@pytest.mark.parametrize("field,value", [
    ("applied", False), ("delta", {"magnitude": 0.4}),
    ("timestamp", "2020-01-01T00:00:00Z"),
    ("triggering_cause_ids", ["invented"]),
])
def test_unsigned_payload_tampering(setup, field, value):
    store, anchor, auto, op, *_ = setup
    signed = store.sign_event(event(), anchor, key_id="auto", key=auto)
    signed["body"]["payload"][field] = value
    with pytest.raises(IntegrityError, match="signature"):
        store.append(signed, anchor, checkpoint_key_id="op", checkpoint_key=op)
    assert store.read(anchor).skill_confidence == {}


@pytest.mark.parametrize("mutation", ["agent", "core_hash", "seq", "previous", "role", "key_id"])
def test_envelope_binding_is_signed(setup, mutation):
    store, anchor, auto, op, *_ = setup
    signed = store.sign_event(event(), anchor, key_id="auto", key=auto)
    if mutation == "key_id":
        signed["key_id"] = "op"
    else:
        signed["body"][mutation] = {"agent": "robot-B", "core_hash": "f" * 64,
                                    "seq": 2, "previous": "f" * 64, "role": "operator"}[mutation]
    with pytest.raises(IntegrityError):
        store.append(signed, anchor, checkpoint_key_id="op", checkpoint_key=op)


def test_forged_signature_replay_reorder_and_stale_writer(setup):
    store, initial, auto, op, *_ = setup
    first = store.sign_event(event(), initial, key_id="auto", key=auto)
    forged = copy.deepcopy(first)
    forged["signature"] = "00" * 64
    with pytest.raises(IntegrityError, match="signature"):
        store.append(forged, initial, checkpoint_key_id="op", checkpoint_key=op)
    anchor = store.append(first, initial, checkpoint_key_id="op", checkpoint_key=op)
    with pytest.raises(IntegrityError, match="replay/reorder/drop"):
        store.append(first, anchor, checkpoint_key_id="op", checkpoint_key=op)
    with pytest.raises(IntegrityError, match="anchor"):
        store.append(first, initial, checkpoint_key_id="op", checkpoint_key=op)
    second = store.sign_event(event(), anchor, key_id="auto", key=auto)
    second["body"]["seq"] = 3
    second = _sign(second["body"], "auto", auto)
    with pytest.raises(IntegrityError, match="replay/reorder/drop"):
        store.append(second, anchor, checkpoint_key_id="op", checkpoint_key=op)


@pytest.mark.parametrize("mutation", ["reorder", "drop", "replay", "state", "extra", "applied"])
def test_bundle_mutations_even_with_resigned_manifest(setup, tmp_path, mutation):
    store, anchor, auto, op, *_ = setup
    for _ in range(3):
        anchor = append(store, anchor, event(), auto, op)
    bundle = json.loads(store.export(anchor))
    body = bundle["body"]
    if mutation == "reorder":
        body["tail"].reverse()
    elif mutation == "drop":
        body["tail"].pop()
    elif mutation == "replay":
        body["tail"][1] = body["tail"][0]
    elif mutation == "state":
        body["state"]["risk_appetite_actual"] = 0.9
    elif mutation == "extra":
        body["state"]["backdoor"] = True
    else:
        body["tail"][0]["body"]["payload"]["applied"] = False
    bundle = _sign(body, "op", op)
    with pytest.raises(IntegrityError):
        restore_args(setup, tmp_path, json.dumps(bundle), anchor)


def test_unsigned_checkpoint_tamper_and_wrong_checkpoint_role(setup, tmp_path):
    store, anchor, auto, op, *_ = setup
    anchor = append(store, anchor, event(), auto, op)
    bundle = json.loads(store.export(anchor))
    bundle["body"]["state"]["skill_confidence"]["injected"] = 0.5
    with pytest.raises(IntegrityError, match="signature"):
        restore_args(setup, tmp_path, json.dumps(bundle), anchor)
    signed = _sign(bundle["body"], "auto", auto)
    with pytest.raises(IntegrityError, match="role"):
        restore_args(setup, tmp_path, json.dumps(signed), anchor)
    ev = store.sign_event(event(), anchor, key_id="auto", key=auto)
    with pytest.raises(IntegrityError, match="role"):
        store.append(ev, anchor, checkpoint_key_id="auto", checkpoint_key=auto)
    assert store.read(anchor).skill_confidence["grasp"] == 0.9


def test_rollback_detected_only_with_current_external_anchor(setup, tmp_path):
    store, old_anchor, auto, op, keys, core, path = setup
    old_bundle = store.export(old_anchor)
    anchor = append(store, old_anchor, event(), auto, op)
    with pytest.raises(IntegrityError, match="anchor"):
        restore_args(setup, tmp_path, old_bundle, anchor)
    # Explicit observed limitation: rolling back both data and anchor is accepted.
    rolled = restore_args(setup, tmp_path, old_bundle, old_anchor)
    assert rolled.read(old_anchor).skill_confidence == {}
    rolled.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE manifest SET json=?", (old_bundle,))
        db.execute("DELETE FROM events")
    with pytest.raises(IntegrityError, match="anchor"):
        store.read(anchor)
    store.close()
    with pytest.raises(IntegrityError, match="anchor"):
        CheckpointedPersonaStore.open(path, "robot-A", core, keys, anchor=anchor, retention=2)


@pytest.mark.parametrize("mutation", ["delete", "payload", "seq"])
def test_event_table_tampering_detected(setup, mutation):
    store, anchor, auto, op, _, _, path = setup
    anchor = append(store, anchor, event(), auto, op)
    with sqlite3.connect(path) as db:
        if mutation == "delete":
            db.execute("DELETE FROM events")
        elif mutation == "seq":
            db.execute("UPDATE events SET seq=7")
        else:
            db.execute("UPDATE events SET json='{}'")
    with pytest.raises(IntegrityError, match="table"):
        store.read(anchor)


def test_sqlite_transaction_failure_rolls_back_append_and_checkpoint(setup):
    store, anchor, auto, op, *_ = setup
    for _ in range(2):
        anchor = append(store, anchor, event(), auto, op)
    before = store.export(anchor)
    # Force failure after event insertion and manifest update, during eviction.
    store._db.execute("CREATE TRIGGER fail_prune BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    envelope = store.sign_event(event(), anchor, key_id="auto", key=auto)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        store.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=op)
    assert store.export(anchor) == before
    store._db.execute("DROP TRIGGER fail_prune")
    newer = store.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=op)
    assert newer.seq == 3
    assert store.read(newer).skill_confidence["grasp"] == pytest.approx(0.7)


@pytest.mark.parametrize("field,value", [
    ("applied", "false"), ("applied", 1), ("schema_version", True),
    ("schema_version", 2), ("delta", {"magnitude": True}),
    ("delta", {"magnitude": float("nan")}), ("delta", {"magnitude": float("inf")}),
    ("delta", {"magnitude": "0.1"}), ("delta", {"magnitude": -0.1}),
    ("delta", {"magnitude": 0.1, "backdoor": 1}),
    ("timestamp", datetime(2026, 10, 21)), ("capability", ""),
])
def test_revalidate_untrusted_event_models(setup, field, value):
    store, anchor, auto, *_ = setup
    ev = event().model_copy(update={field: value})
    with pytest.raises(IntegrityError):
        store.sign_event(ev, anchor, key_id="auto", key=auto)


@pytest.mark.parametrize("state_field,value", [
    ("risk_appetite_actual", float("nan")),
    ("risk_appetite_actual", True),
    ("skill_confidence", {"grasp": "0.3"}),
    ("skill_confidence", {"grasp": -1}),
    ("human_check_threshold", {"move": 1.1}),
    ("tendencies", {"arbitrary": []}),
    ("tendencies", {"drift_history": [{"severity": "bad", "reason": "x"}]}),
])
def test_checkpoint_state_closed_schema(setup, tmp_path, state_field, value):
    store, anchor, auto, op, *_ = setup
    for _ in range(3):
        anchor = append(store, anchor, event(), auto, op)
    body = json.loads(store.export(anchor))["body"]
    body["state"][state_field] = value
    # NaN rejected by canonical encoder before verification.
    with pytest.raises(IntegrityError):
        signed = _sign(body, "op", op)
        restore_args(setup, tmp_path, json.dumps(signed), anchor)


def test_export_binding_unknown_key_and_existing_store_protection(setup, tmp_path):
    store, anchor, _, op, keys, core, path = setup
    bundle = store.export(anchor)
    for override in ({"agent": "robot-B"}, {"core": PersonaCore()}, {"retention": 1}, {"keys": {}}):
        with pytest.raises(IntegrityError):
            restore_args(setup, tmp_path, bundle, anchor, **override)
    with pytest.raises(sqlite3.OperationalError):
        restore_args(setup, tmp_path, bundle, anchor, path=path)
    with pytest.raises(sqlite3.OperationalError):
        CheckpointedPersonaStore.create(path, "robot-A", core, keys,
                                        checkpoint_key_id="op", checkpoint_key=op, retention=2)
    assert store.export(anchor) == bundle


def test_duplicate_json_keys_and_nonfinite_json_rejected(setup, tmp_path):
    store, anchor, *_ = setup
    for raw in ('{"body": {}, "body": {}}', '{"value": NaN}', '[]'):
        with pytest.raises(IntegrityError):
            restore_args(setup, tmp_path, raw, anchor)


def test_independent_connections_serialize_and_reject_stale_anchor(setup):
    store, anchor, auto, op, keys, core, path = setup
    other = CheckpointedPersonaStore.open(path, "robot-A", core, keys, anchor=anchor, retention=2)
    envelope = other.sign_event(event(), anchor, key_id="auto", key=auto)
    new_anchor = append(store, anchor, event(), auto, op)
    with pytest.raises(IntegrityError, match="anchor"):
        other.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=op)
    assert other.read(new_anchor).skill_confidence["grasp"] == 0.9
    other.close()


def test_skip_eviction_and_bounded_drift_history(setup):
    store, anchor, auto, op, *_ = setup
    anchor = append(store, anchor, event(applied=False), auto, op)
    for i in range(12):
        anchor = append(store, anchor, event("drift_event", delta={"severity": "low", "reason": str(i)}), auto, op)
    state = store.read(anchor)
    assert state.skill_confidence == {}
    assert state.risk_appetite_actual == 0
    assert [x["reason"] for x in state.tendencies["drift_history"]] == [str(i) for i in range(4, 12)]
    assert len(json.loads(store.export(anchor))["body"]["tail"]) == 2


def test_process_crash_during_eviction_recovers_committed_state(setup):
    """Kill the actual append transaction after INSERT/UPDATE, before prune."""
    import os
    import subprocess
    import sys
    from cryptography.hazmat.primitives import serialization

    store, anchor, auto, op, keys, core, path = setup
    for _ in range(2):
        anchor = append(store, anchor, event(), auto, op)
    before = store.export(anchor)
    raw = lambda key: key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                         serialization.NoEncryption()).hex()
    child = r'''
import dataclasses, json, os, sys
from datetime import UTC, datetime
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from aeros.runtime.checkpointed_persona import Anchor, CheckpointedPersonaStore, TrustedKey
from aeros.runtime.persona_core import PersonaCore
from aeros.runtime.persona_event import PersonaAdaptiveEvent
path, auto_hex, op_hex, anchor_json = sys.argv[1:]
auto, op = (Ed25519PrivateKey.from_private_bytes(bytes.fromhex(x)) for x in (auto_hex, op_hex))
keys = {"auto": TrustedKey(auto.public_key(), "automatic"), "op": TrustedKey(op.public_key(), "operator")}
a = Anchor(**json.loads(anchor_json))
s = CheckpointedPersonaStore.open(path, "robot-A", PersonaCore(risk_appetite="conservative"), keys, anchor=a, retention=2)
s._db.create_function("crash", 0, lambda: os._exit(23))
s._db.execute("CREATE TEMP TRIGGER crash_prune BEFORE DELETE ON events BEGIN SELECT crash(); END")
e = PersonaAdaptiveEvent(kind="skill_confidence_decrease", capability="grasp", delta={"magnitude": 0.1}, timestamp=datetime(2026,10,21,tzinfo=UTC))
signed = s.sign_event(e, a, key_id="auto", key=auto)
s.append(signed, a, checkpoint_key_id="op", checkpoint_key=op)
sys.exit(99)
'''
    completed = subprocess.run([sys.executable, "-c", child, str(path), raw(auto), raw(op),
                                json.dumps(dataclasses.asdict(anchor))],
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 23, completed.stderr
    store.close()
    reopened = CheckpointedPersonaStore.open(path, "robot-A", core, keys, anchor=anchor, retention=2)
    assert reopened.export(anchor) == before
    newer = append(reopened, anchor, event(), auto, op)
    assert newer.seq == 3
    assert reopened.read(newer).skill_confidence["grasp"] == pytest.approx(0.7)
    reopened.close()


@pytest.mark.parametrize("field,value", [
    ("agent", "robot-B"), ("core_hash", "f" * 64),
    ("role", "operator"), ("previous", "f" * 64),
])
def test_valid_crypto_cannot_bypass_envelope_binding_or_roles(setup, field, value):
    store, anchor, auto, op, *_ = setup
    signed = store.sign_event(event(), anchor, key_id="auto", key=auto)
    signed["body"][field] = value
    signed = _sign(signed["body"], "auto", auto)
    with pytest.raises(IntegrityError):
        store.append(signed, anchor, checkpoint_key_id="op", checkpoint_key=op)


def test_untrusted_signer_rejected_even_with_valid_signature(setup):
    store, anchor, auto, op, *_ = setup
    signed = store.sign_event(event(), anchor, key_id="auto", key=auto)
    signed = _sign(signed["body"], "unknown", Ed25519PrivateKey.generate())
    with pytest.raises(IntegrityError, match="untrusted"):
        store.append(signed, anchor, checkpoint_key_id="op", checkpoint_key=op)


@pytest.mark.parametrize("field,value", [
    ("seq", True), ("seq", -1), ("seq", 1.0),
    ("digest", "z" * 64), ("core_hash", "short"), ("agent", ""),
])
def test_anchor_schema_is_strict(setup, field, value):
    _, anchor, *_ = setup
    with pytest.raises(IntegrityError):
        dataclasses.replace(anchor, **{field: value})


def test_conservative_cap_holds_at_genesis_and_after_operator_relax(setup):
    store, anchor, auto, op, *_ = setup
    assert store.read(anchor).risk_appetite_actual == 0.4
    for _ in range(5):
        anchor = append(store, anchor, event("risk_appetite_calibration_relax", delta={"magnitude": 0.5}),
                        auto, op, operator=True)
    assert store.read(anchor).risk_appetite_actual == 0.4


@pytest.mark.parametrize("field,value", [("schema_version", True), ("applied", 1)])
def test_raw_signed_event_cannot_bypass_strict_types(setup, field, value):
    store, anchor, auto, op, *_ = setup
    envelope = store.sign_event(event(), anchor, key_id="auto", key=auto)
    envelope["body"]["payload"][field] = value
    envelope = _sign(envelope["body"], "auto", auto)
    with pytest.raises(IntegrityError):
        store.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=op)
    assert store.read(anchor).skill_confidence == {}


def test_wrong_checkpoint_private_key_does_not_commit(setup):
    store, anchor, auto, *_ = setup
    envelope = store.sign_event(event(), anchor, key_id="auto", key=auto)
    before = store.export(anchor)
    with pytest.raises(IntegrityError, match="signature"):
        store.append(envelope, anchor, checkpoint_key_id="op", checkpoint_key=Ed25519PrivateKey.generate())
    assert store.export(anchor) == before
