"""`aeros migrate` CLI — Layer 3 cross-deployment migration.

This module covers all three migration entry points: ``aeros migrate
export``, ``aeros migrate import`` / ``aeros migrate
verify``, and the audit-chain continuity rules
AC-1..AC-5 from spec §6 that mark the cross-deployment
boundary in the new deployment's episodic chain.

The bundle is a ``.aer`` tarball (gzip) containing 10 named files plus
a detached Ed25519 signature. The on-disk layout, signature input, and
identity-hash invariance proof are specified in
the Layer 3 design specification.

Three non-obvious details enforced here:

* **WAL checkpoint before copy** (WAL-checkpoint gotcha). ``episodic.db`` runs in
  WAL mode in production, so a naive ``shutil.copyfile`` of the main
  DB file misses any rows still sitting in the ``-wal`` sidecar. We
  ``PRAGMA wal_checkpoint(TRUNCATE)`` before the copy so the staged
  file is self-contained.
* **meta.json is in the signature digest** (spec hole #1 fix). The
  original §1.10 enumeration omitted ``meta.json`` from the file list,
  which would have allowed an attacker to swap timestamps or
  ``source_bridge_id`` post-signing without invalidating the bundle.
  ``SIGNATURE_FILES`` below includes it; ``test_export_signature_includes_meta_json``
  pins the fix.
* **Episodic.db is the canonical audit chain** (spec
  hole #2 fix). The bridge keeps a separate ``evolution_audit.db``
  hot-cache under ``~/.aeros/`` for fast ``GET /api/evolution/audit``
  reads, but that file is NOT in the bundle: it duplicates events that
  ``_record_evolution_audit`` already mirrors into episodic memory
  under ``kind="evolution.<action>"``. After COMMIT we append a
  single ``kind="migration_import"`` episodic record (AC-3..AC-5) so
  the new deployment's audit chain shows the cross-deployment boundary
  as a visible gap rather than silently re-rooting (Belief 5: audit is
  structural). The endpoint folds these markers back into its response
  on every request.

Import is a 9-step state machine (LOAD → VERIFY_SIG → VERIFY_IDENTITY_HASH
→ VERIFY_TEMPLATE_COMPAT → CHECK_DEPLOYMENT_CONFLICT → RECONSTRUCT →
MERGE_EPISODIC → SMOKE_TEST → COMMIT) with fail-fast exit codes 2/3/4/5/6.
Staging happens under ``<target>/.tmp-import/``; on
any failure the staging dir is rmtree'd so the target is left untouched.
``aeros migrate verify`` is a dry-run — only LOAD + VERIFY_SIG +
VERIFY_IDENTITY_HASH; never writes to a target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from aeros.audit.trusted_keys import (
    DEFAULT_RESIDENT_KEY_PATH as _DEFAULT_SIGNING_KEY,
    DEFAULT_TRUSTED_KEYS_DIR as _DEFAULT_TRUSTED_KEYS_DIR,
    ensure_resident_keypair,
    load_trusted_pubkeys,
)
from aeros.runtime.persistent_agent import (
    IdentityManifest,
    hash_ecm_registry,
    hash_env_policies,
    hash_persona,
    hash_policy_rules,
)
from aeros.runtime.persona import load_persona
from aeros.runtime.persona_adaptive import PersonaAdaptive

# Bundle constants — exposed publicly so import/verify can reuse.
BUNDLE_VERSION = "1.0.0"

# Files written into the bundle, in tarball-add order. semantic.db is a
# placeholder until Layer 2 lands; the tar entry exists for forward compat.
# persona_adaptive.json is the (the state contract) null-safe slot that
# carries the typed-event log.
BUNDLE_FILES: tuple[str, ...] = (
    "manifest.json",
    "system_prompt.txt",
    "persona.yaml",
    "persona_adaptive.json",
    "env_policies.yaml",
    "policy.yaml",
    "capability_set.json",
    "episodic.db",
    "semantic.db",
    "signature.sig",
    "meta.json",
)

# Files covered by the detached signature, in lexicographic order. Spec
# §1.10 enumerates these; meta.json is the fix for spec hole #1 (the
# original list omitted it). persona_adaptive.json is folded into the
# digest so a tampered overlay invalidates the bundle just like persona.yaml.
SIGNATURE_FILES: tuple[str, ...] = (
    "capability_set.json",
    "env_policies.yaml",
    "episodic.db",
    "manifest.json",
    "meta.json",
    "persona.yaml",
    "persona_adaptive.json",
    "policy.yaml",
    "semantic.db",
    "system_prompt.txt",
)

# The bridge's lifespan hard-codes this string into IdentityManifest.system_prompt.
# Kept in lockstep with bridge/app/main.py:lifespan; bumping it on the bridge
# side requires bumping it (and SOURCE_TEMPLATE_VERSION below) here too so
# import-side template-compat refuses bundles built off a different template.
DEFAULT_SYSTEM_PROMPT = "aeros-bridge governance v0.5"
SOURCE_TEMPLATE_VERSION = "v0.5"

# Capabilities the reference bridge registers — mirrors
# bridge/app/main.py:_REGISTERED_CAPABILITIES. Used as the fallback when
# bridge/configs/contracts/ is empty (e.g. bench harnesses).
_BRIDGE_DEFAULT_CAPABILITIES: tuple[str, ...] = (
    "perception.locate_object",
    "manipulation.grasp",
    "manipulation.place",
    "navigation.move_to",
    "manipulation.fold_and_pinch",
)

# Default disk locations. The signing-key + trusted-keys paths are
# imported from ``aeros.audit.trusted_keys`` (lift) so the
# resolver and operator persona-event CLIs share one canonical
# defaults surface.
_DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parents[3] / "bridge"
_DEFAULT_EPISODIC_DB = Path("~/.aeros/episodic.db").expanduser()

# Signature envelope schema. JSON with hex-encoded bytes; same shape as
# the ECM signing envelope but distinct schema_version so the two are
# never confused at parse time.
SIG_SCHEMA_VERSION = "v1-migrate"
SIG_ALGORITHM = "ed25519"


# --------------------- internal helpers --------------------- #


def _wal_checkpoint(db_path: Path) -> None:
    """Flush the WAL into the main DB file before copy (WAL-checkpoint gotcha).

    ``PRAGMA wal_checkpoint(TRUNCATE)`` blocks until pending writes are
    merged and removes the ``-wal`` / ``-shm`` sidecar files. Without
    this, a copy of the main DB file would miss in-flight rows. The DB
    must not have any active writers in another process; for the export
    use case the bridge is expected to be quiescent.
    """
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()


def _episodic_row_count(db_path: Path | None) -> int:
    """Return the row count of the ``episodic`` table, or 0 if absent."""
    if db_path is None or not db_path.exists() or db_path.stat().st_size == 0:
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM episodic")
        return int(cur.fetchone()[0])
    except sqlite3.DatabaseError:
        return 0
    finally:
        conn.close()


def _read_capabilities(bridge_dir: Path) -> list[str]:
    """Discover registered capability names from bridge/configs/contracts/.

    Each ``*.json`` declares one capability. Falls back to the reference
    bridge's hard-coded list when the directory is empty so bench / test
    harnesses (which install no contracts) still produce a sensible
    capability_set.json.
    """
    contracts_dir = bridge_dir / "configs" / "contracts"
    if not contracts_dir.is_dir():
        return list(_BRIDGE_DEFAULT_CAPABILITIES)
    found: list[str] = []
    for path in sorted(contracts_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("capability"), str):
            found.append(data["capability"])
    return found or list(_BRIDGE_DEFAULT_CAPABILITIES)


def _build_env_policies_for_hash(env_policies_yaml: Path) -> dict[str, Any]:
    """Reconstruct the dict shape that bridge/app/main.py feeds into hash_env_policies.

    Must mirror the lifespan exactly — any extra/missing key here would
    flip the env_policies_hash and break identity invariance across export.
    """
    raw = yaml.safe_load(env_policies_yaml.read_text(encoding="utf-8")) or {}
    env_blocks: dict[str, Any] = raw.get("environments") or {}
    out: dict[str, Any] = {}
    for name, block in env_blocks.items():
        out[name] = {
            "max_speed_mult": float(block.get("max_speed_mult", 1.0)),
            "max_force_mult": float(block.get("max_force_mult", 1.0)),
            "approval_required": bool(block.get("approval_required", False)),
            "max_retries": int(block.get("max_retries", 3)),
            "watcher_sensitivity": float(block.get("watcher_sensitivity", 0.5)),
            "forbidden_zones": list(block.get("forbidden_zones") or ()),
        }
    return out


def _read_policy_rules(policy_yaml: Path) -> list[dict[str, object]]:
    """Pull the ordered ``policies:`` list from policy.yaml.

    bridge/app/aeros/policy.py:_load_rules reads the same key; this
    helper duplicates it rather than importing because importing the
    bridge package starts dragging FastAPI / asyncio fixtures into the
    CLI's import graph.
    """
    raw = yaml.safe_load(policy_yaml.read_text(encoding="utf-8")) or {}
    rules = raw.get("policies") or []
    if not isinstance(rules, list):
        return []
    return [dict(rule) for rule in rules if isinstance(rule, dict)]


def _capability_set_payload(capabilities: Iterable[str]) -> dict[str, Any]:
    """Build the capability_set.json payload (sorted, deduped, hashed)."""
    sorted_caps = sorted(set(capabilities))
    return {
        "$schema": "https://aeros.dev/schemas/capability-set-v1.json",
        "capabilities": sorted_caps,
        "ecm_registry_hash": hash_ecm_registry(sorted_caps),
    }


def _manifest_dict(manifest: IdentityManifest) -> dict[str, Any]:
    """Serialise IdentityManifest into the spec §1.2 JSON shape.

    Includes ``identity_hash`` so import-side VI-1 can do a byte-equal
    compare without recomputing from scratch (it still recomputes for
    the integrity check; the field is for human / tool inspection).
    """
    return {
        "$schema": "https://aeros.dev/schemas/export-manifest-v1.json",
        "system_prompt": manifest.system_prompt,
        "env_policies_hash": manifest.env_policies_hash,
        "policy_rules_hash": manifest.policy_rules_hash,
        "ecm_registry_hash": manifest.ecm_registry_hash,
        "persona_hash": manifest.persona_hash,
        "runtime_version": manifest.runtime_version,
        "schema_version": manifest.schema_version,
        "identity_hash": manifest.identity_hash,
    }


def _bridge_id_for(bridge_dir: Path) -> str:
    """Synthesise a stable source_bridge_id for meta.json.

    Hostname + canonical bridge path + current timestamp, hashed and
    truncated. Distinguishes bundles produced from different physical
    hosts; downstream tooling never parses this back, only compares.
    """
    raw = (
        f"{socket.gethostname()}::"
        f"{bridge_dir.resolve()}::"
        f"{datetime.now(UTC).isoformat()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_signature_digest(staging_dir: Path) -> bytes:
    """Compute the SHA-256 digest the signature is taken over.

    Concatenates the raw bytes of ``SIGNATURE_FILES`` (lexicographic
    order, including ``meta.json`` per the spec hole #1 fix) and returns
    the 32-byte digest. Streamed in chunks so bundle size never has to
    fit in memory. Files that don't exist (e.g. semantic.db before
    Layer 2) contribute nothing — keeping the digest stable as features
    land.
    """
    h = hashlib.sha256()
    for name in SIGNATURE_FILES:
        path = staging_dir / name
        if not path.exists():
            continue
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.digest()


def _write_signature_file(
    staging_dir: Path,
    *,
    digest: bytes,
    signature: bytes,
    public_key_bytes: bytes,
    signer_name: str | None,
) -> None:
    """Write signature.sig as a JSON envelope around the Ed25519 detached sig."""
    payload = {
        "schema_version": SIG_SCHEMA_VERSION,
        "algorithm": SIG_ALGORITHM,
        "digest_sha256": digest.hex(),
        "signature": signature.hex(),
        "signer_pubkey": public_key_bytes.hex(),
        "signer_name": signer_name,
        "signed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (staging_dir / "signature.sig").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def verify_bundle_signature(bundle_path: Path) -> bool:
    """Recompute the digest from a ``.aer`` bundle and verify the signature.

    Lightweight building block used by tests (and reused by the import path
    import). Returns True iff the cryptographic signature matches the
    digest recomputed from the bundle's current bytes — which means
    tampering with any file in ``SIGNATURE_FILES`` (notably ``meta.json``
    after the spec hole #1 fix) makes this return False.

    Trust resolution (matching against ``trusted_keys/``) is the
    import-side concern and lives in the import path.
    """
    from aeros.marketplace.signing import verify_signature

    with tempfile.TemporaryDirectory(prefix="aeros-verify-") as workdir_str:
        workdir = Path(workdir_str)
        with tarfile.open(bundle_path, "r:gz") as tf:
            tf.extractall(workdir, filter="data")
        sig_path = workdir / "signature.sig"
        if not sig_path.is_file():
            return False
        try:
            envelope = json.loads(sig_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        try:
            pubkey = bytes.fromhex(envelope["signer_pubkey"])
            signature = bytes.fromhex(envelope["signature"])
        except (KeyError, ValueError):
            return False
        digest = compute_signature_digest(workdir)
        if digest.hex() != envelope.get("digest_sha256"):
            return False
        return verify_signature(pubkey, digest, signature)


# --------------------- import-side helpers --------------------- #


def _recompute_manifest_from_bundle(
    staging: Path, manifest_data: dict[str, Any]
) -> IdentityManifest:
    """Rebuild an ``IdentityManifest`` from a staged bundle's config files.

    The same hash inputs the exporter consumed, applied to the bytes
    that arrived in the bundle. If the result's ``identity_hash`` does
    not byte-equal ``manifest_data["identity_hash"]`` then either the
    bundle has been tampered or the source hashed something we no
    longer hash the same way — both are import-time fail-fasts.
    """
    persona_yaml = staging / "persona.yaml"
    env_policies_yaml = staging / "env_policies.yaml"
    policy_yaml = staging / "policy.yaml"
    capability_set_path = staging / "capability_set.json"
    system_prompt_path = staging / "system_prompt.txt"

    # the state contract / identity-invariance: hash reads only PersonaCore.
    persona_payload = load_persona(persona_yaml).core.to_manifest_payload()
    env_policies_dict = _build_env_policies_for_hash(env_policies_yaml)
    policy_rules = _read_policy_rules(policy_yaml)

    capability_set = json.loads(capability_set_path.read_text(encoding="utf-8"))
    capabilities = list(capability_set.get("capabilities", []))

    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    runtime_version = manifest_data.get("runtime_version", "0.5.0")
    schema_version = manifest_data.get("schema_version", "v1")

    return IdentityManifest(
        system_prompt=system_prompt,
        env_policies_hash=hash_env_policies(env_policies_dict),
        policy_rules_hash=hash_policy_rules(policy_rules),
        ecm_registry_hash=hash_ecm_registry(capabilities),
        persona_hash=hash_persona(persona_payload),
        runtime_version=runtime_version,
        schema_version=schema_version,
    )


def _verify_envelope_against_trust(
    sig_path: Path, trusted_keys_dir: Path
) -> tuple[bytes, bytes, bytes] | tuple[None, None, None]:
    """Parse signature.sig and confirm signer + cryptographic verify.

    Returns ``(pubkey, signature, embedded_digest)`` on success, or a
    triple of ``None`` on any envelope / trust / cryptographic failure.
    The caller raises the appropriate fail-fast exit on ``None``; this
    helper does not print or exit so import / verify share it cleanly.
    """
    from aeros.marketplace.signing import verify_signature

    if not sig_path.is_file():
        return None, None, None
    try:
        envelope = json.loads(sig_path.read_text(encoding="utf-8"))
        pubkey = bytes.fromhex(envelope["signer_pubkey"])
        signature = bytes.fromhex(envelope["signature"])
        embedded_digest = bytes.fromhex(envelope["digest_sha256"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None, None, None
    if envelope.get("schema_version") != SIG_SCHEMA_VERSION:
        return None, None, None
    if envelope.get("algorithm") != SIG_ALGORITHM:
        return None, None, None
    trusted = load_trusted_pubkeys(trusted_keys_dir)
    if pubkey.hex() not in trusted:
        return None, None, None
    if not verify_signature(pubkey, embedded_digest, signature):
        return None, None, None
    return pubkey, signature, embedded_digest


def _record_migration_import(
    *,
    target_episodic_db: Path,
    identity_hash: str,
    source_identity_hash: str,
    bundle_version: str,
    source_bridge_id: str,
    episodic_rows_imported: int,
    semantic_rows_imported: int,
    import_tool_version: str,
) -> None:
    """Append a single AC-3..AC-5 migration_import episodic record.

    Spec §6.1: Import event MUST produce one new episodic record with
    ``kind="migration_import"``, ``identity_hash`` equal to the
    post-migration hash (AC-4 — which §5 proves byte-equals the source
    hash), and ``timestamp`` at the import moment (AC-5). Belief 5
    (audit is structural): this entry makes the cross-deployment
    boundary observable in the chain rather than silently re-rooting at
    the new deployment's first locally-generated event.

    Runs after COMMIT so a failed import never leaves a partial
    migration_import row behind. The summary template here is exactly
    the one ``test_layer3_migration_e2e`` greps for; do not edit casually.
    """
    from aeros.runtime.episodic_db import EpisodicStore

    store = EpisodicStore(target_episodic_db)
    try:
        store.append(
            identity_hash,
            timestamp=datetime.now(UTC).isoformat(),
            kind="migration_import",
            intent_id=None,
            summary=f"Agent migrated from bridge {source_bridge_id}",
            payload={
                "source_identity_hash": source_identity_hash,
                "bundle_version": bundle_version,
                "episodic_rows_imported": episodic_rows_imported,
                "semantic_rows_imported": semantic_rows_imported,
                "import_tool_version": import_tool_version,
            },
        )
    finally:
        store.close()


def _merge_episodic_dedup(existing_db: Path, staging_db: Path) -> int:
    """Fold *existing_db* rows into *staging_db* via INSERT OR IGNORE.

    The unique key is ``(identity_hash, timestamp, kind)`` per spec; a
    UNIQUE INDEX is added to the staging copy if absent so SQLite's OR
    IGNORE does the dedup natively. Returns the count of rows actually
    inserted (rows from existing that were not already in staging).
    """
    target_conn = sqlite3.connect(str(staging_db))
    src_conn = sqlite3.connect(str(existing_db))
    try:
        target_conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_dedup "
            "ON episodic(identity_hash, timestamp, kind)"
        )
        try:
            rows = src_conn.execute(
                "SELECT identity_hash, timestamp, kind, intent_id, summary, payload_json "
                "FROM episodic"
            ).fetchall()
        except sqlite3.DatabaseError:
            rows = []
        before = target_conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
        target_conn.executemany(
            "INSERT OR IGNORE INTO episodic "
            "(identity_hash, timestamp, kind, intent_id, summary, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        target_conn.commit()
        after = target_conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
    finally:
        target_conn.close()
        src_conn.close()
    return int(after) - int(before)


# --------------------- export --------------------- #


def cmd_export(args: argparse.Namespace) -> int:
    """Implement ``aeros migrate export``.

    Reads the bridge configs, snapshots episodic.db (after a WAL
    checkpoint), assembles the 10-file bundle, signs the SHA-256 of the
    file concatenation with Ed25519, and writes the gzip tarball to
    ``--output``. Returns 0 on success, non-zero on a fail-fast abort.
    """
    bridge_dir = Path(args.bridge_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    persona_yaml = bridge_dir / "configs" / "persona.yaml"
    env_policies_yaml = bridge_dir / "configs" / "env_policies.yaml"
    policy_yaml = bridge_dir / "configs" / "policy.yaml"

    for required in (persona_yaml, env_policies_yaml, policy_yaml):
        if not required.is_file():
            print(f"error: missing required config: {required}", file=sys.stderr)
            return 1

    # the state contract / identity-invariance: hash reads only PersonaCore.
    persona_payload = load_persona(persona_yaml).core.to_manifest_payload()
    env_policies_dict = _build_env_policies_for_hash(env_policies_yaml)
    policy_rules = _read_policy_rules(policy_yaml)
    capabilities = _read_capabilities(bridge_dir)
    capability_set = _capability_set_payload(capabilities)
    system_prompt = DEFAULT_SYSTEM_PROMPT

    runtime_version = args.runtime_version or "0.5.0"
    manifest = IdentityManifest(
        system_prompt=system_prompt,
        env_policies_hash=hash_env_policies(env_policies_dict),
        policy_rules_hash=hash_policy_rules(policy_rules),
        ecm_registry_hash=hash_ecm_registry(capabilities),
        persona_hash=hash_persona(persona_payload),
        runtime_version=runtime_version,
    )

    # Identity hash sanity check — a callsite-provided expected value
    # MUST match the freshly-computed manifest. Catches "configs on disk
    # disagree with the running bridge" before bytes hit the bundle.
    expected = getattr(args, "expected_identity_hash", None)
    if expected and expected != manifest.identity_hash:
        print(
            f"error: identity_hash mismatch: configs on disk hash to "
            f"{manifest.identity_hash} but --expected-identity-hash was "
            f"{expected}",
            file=sys.stderr,
        )
        return 3

    episodic_src = (
        Path(args.episodic_db).expanduser() if args.episodic_db else _DEFAULT_EPISODIC_DB
    )

    # WAL checkpoint MUST run before the file copy (WAL-checkpoint gotcha).
    _wal_checkpoint(episodic_src)
    episodic_count = _episodic_row_count(episodic_src)

    signing_key_path = (
        Path(args.signing_key).expanduser().resolve()
        if args.signing_key
        else _DEFAULT_SIGNING_KEY
    )
    trusted_keys_dir = (
        Path(args.trusted_keys_dir).expanduser().resolve()
        if args.trusted_keys_dir
        else _DEFAULT_TRUSTED_KEYS_DIR
    )
    try:
        priv_bytes, pub_bytes = ensure_resident_keypair(
            signing_key_path,
            trusted_keys_dir=trusted_keys_dir,
            auto_generate=bool(args.generate_key),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="aeros-export-") as staging_str:
        staging = Path(staging_str)

        (staging / "manifest.json").write_text(
            json.dumps(_manifest_dict(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (staging / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        shutil.copyfile(persona_yaml, staging / "persona.yaml")
        # the state contract: null-safe slot for the typed-event log that
        # lands in PersonaAdaptive. Empty here so no adaptation
        # state crosses deployments yet; the slot exists so the import side
        # can tolerate an absent event-log field.
        (staging / "persona_adaptive.json").write_text(
            json.dumps(PersonaAdaptive().to_bundle_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        shutil.copyfile(env_policies_yaml, staging / "env_policies.yaml")
        shutil.copyfile(policy_yaml, staging / "policy.yaml")
        (staging / "capability_set.json").write_text(
            json.dumps(capability_set, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # episodic.db: copy after the checkpoint above, OR write an
        # empty file when no DB exists (fresh bridge with no recorded
        # episodes — still a valid bundle).
        if episodic_src.exists():
            shutil.copyfile(episodic_src, staging / "episodic.db")
        else:
            (staging / "episodic.db").write_bytes(b"")
        # semantic.db is reserved for Layer 2; empty placeholder.
        (staging / "semantic.db").write_bytes(b"")

        meta_payload = {
            "bundle_version": BUNDLE_VERSION,
            "source_runtime_version": runtime_version,
            "source_system_prompt_template_version": SOURCE_TEMPLATE_VERSION,
            "exported_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_bridge_id": _bridge_id_for(bridge_dir),
            "episodic_row_count": episodic_count,
            "semantic_db_present": False,
            "semantic_row_count": 0,
            "export_tool_version": runtime_version,
        }
        (staging / "meta.json").write_text(
            json.dumps(meta_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Sign over the digest of all SIGNATURE_FILES (now including
        # meta.json — spec hole #1 fix).
        digest = compute_signature_digest(staging)
        from aeros.marketplace.signing import sign_payload

        signature = sign_payload(priv_bytes, digest)
        _write_signature_file(
            staging,
            digest=digest,
            signature=signature,
            public_key_bytes=pub_bytes,
            signer_name=args.signer_name,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tf:
            for name in BUNDLE_FILES:
                path = staging / name
                if path.exists():
                    tf.add(path, arcname=name)

    print(f"exported: {output}", file=sys.stderr)
    print(f"  identity_hash={manifest.identity_hash}", file=sys.stderr)
    print(f"  episodic_rows={episodic_count}", file=sys.stderr)
    print(f"  bundle_size={output.stat().st_size} bytes", file=sys.stderr)
    return 0


# --------------------- import + verify --------------------- #

# Files copied from staging into the final target on COMMIT. Excludes
# signature.sig (kept inside the bundle, not part of a deployed bridge's
# config tree) and the empty semantic.db placeholder (Layer 2).
_IMPORT_INSTALL_FILES: tuple[str, ...] = (
    "manifest.json",
    "system_prompt.txt",
    "persona.yaml",
    "persona_adaptive.json",
    "env_policies.yaml",
    "policy.yaml",
    "capability_set.json",
    "meta.json",
    "episodic.db",
    "semantic.db",
)


def cmd_import(args: argparse.Namespace) -> int:
    """Implement ``aeros migrate import`` — 9-step state machine.

    LOAD → VERIFY_SIG (exit 2) → VERIFY_IDENTITY_HASH (exit 3) →
    VERIFY_TEMPLATE_COMPAT (exit 4) → CHECK_DEPLOYMENT_CONFLICT (exit 5)
    → RECONSTRUCT → MERGE_EPISODIC (only when same identity already on
    target) → SMOKE_TEST (exit 6) → COMMIT. Staging happens under
    ``<target>/.tmp-import/``; on any failure we ``shutil.rmtree`` it
    so the existing target is left untouched.
    """
    bundle_path = Path(args.bundle).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    trusted_keys_dir = (
        Path(args.trusted_keys_dir).expanduser().resolve()
        if args.trusted_keys_dir
        else _DEFAULT_TRUSTED_KEYS_DIR
    )
    allow_migrate = bool(args.migrate)

    if not bundle_path.is_file():
        print(f"error: bundle not found: {bundle_path}", file=sys.stderr)
        return 1

    target.mkdir(parents=True, exist_ok=True)
    staging = target / ".tmp-import"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    def _rollback() -> None:
        # Single-source rollback per spec.
        shutil.rmtree(staging, ignore_errors=True)

    try:
        # ---------------- LOAD ----------------
        with tarfile.open(bundle_path, "r:gz") as tf:
            tf.extractall(staging, filter="data")
        manifest_path = staging / "manifest.json"
        meta_path = staging / "meta.json"
        sig_path = staging / "signature.sig"
        for required in (manifest_path, meta_path, sig_path):
            if not required.is_file():
                print(
                    f"error: bundle missing required file: {required.name}",
                    file=sys.stderr,
                )
                _rollback()
                return 1
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))

        # ---------------- VERIFY_SIG ----------------
        pubkey, _signature, embedded_digest = _verify_envelope_against_trust(
            sig_path, trusted_keys_dir
        )
        if pubkey is None or embedded_digest is None:
            print(
                "error: signature.sig is missing, malformed, untrusted, or invalid; "
                f"trusted_keys_dir={trusted_keys_dir}",
                file=sys.stderr,
            )
            _rollback()
            return 2

        # ---------------- VERIFY_IDENTITY_HASH ----------------
        recomputed_digest = compute_signature_digest(staging)
        if recomputed_digest != embedded_digest:
            print(
                "error: bundle digest mismatch — files were altered after signing "
                f"(envelope={embedded_digest.hex()}, recomputed={recomputed_digest.hex()})",
                file=sys.stderr,
            )
            _rollback()
            return 3
        try:
            recomputed_manifest = _recompute_manifest_from_bundle(
                staging, manifest_data
            )
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            print(
                f"error: cannot recompute identity from bundle configs: {exc}",
                file=sys.stderr,
            )
            _rollback()
            return 3
        claimed_identity = manifest_data.get("identity_hash")
        if recomputed_manifest.identity_hash != claimed_identity:
            print(
                f"error: identity_hash mismatch (manifest claims {claimed_identity}, "
                f"recomputed {recomputed_manifest.identity_hash})",
                file=sys.stderr,
            )
            _rollback()
            return 3

        # ---------------- VERIFY_TEMPLATE_COMPAT ----------------
        bundle_template = meta_data.get("source_system_prompt_template_version")
        if bundle_template != SOURCE_TEMPLATE_VERSION:
            print(
                f"error: template version mismatch (bundle={bundle_template!r}, "
                f"local bridge={SOURCE_TEMPLATE_VERSION!r})",
                file=sys.stderr,
            )
            _rollback()
            return 4

        # ---------------- CHECK_DEPLOYMENT_CONFLICT ----------------
        existing_manifest = target / "manifest.json"
        existing_identity_hash: str | None = None
        if existing_manifest.is_file():
            try:
                existing_data = json.loads(existing_manifest.read_text(encoding="utf-8"))
                existing_identity_hash = existing_data.get("identity_hash")
            except (json.JSONDecodeError, OSError):
                existing_identity_hash = None
        if (
            existing_identity_hash is not None
            and existing_identity_hash != recomputed_manifest.identity_hash
            and not allow_migrate
        ):
            print(
                f"error: target already deployed with different identity_hash "
                f"{existing_identity_hash}; pass --migrate to override",
                file=sys.stderr,
            )
            _rollback()
            return 5

        # ---------------- RECONSTRUCT ----------------
        # Tarball already extracted to staging/; nothing further to assemble.
        # Capture the source-side episodic row count BEFORE MERGE_EPISODIC
        # folds existing target rows back in: the spec §6.1 payload
        # ``episodic_rows_imported`` is "rows arriving from the bundle",
        # not "rows in the staged DB after the merge".
        staged_episodic_path = staging / "episodic.db"
        source_episodic_rows = _episodic_row_count(staged_episodic_path)

        # ---------------- MERGE_EPISODIC ----------------
        same_identity = (
            existing_identity_hash is not None
            and existing_identity_hash == recomputed_manifest.identity_hash
        )
        if same_identity:
            existing_episodic = target / "episodic.db"
            staged_episodic = staging / "episodic.db"
            if (
                existing_episodic.exists()
                and existing_episodic.stat().st_size > 0
                and staged_episodic.exists()
                and staged_episodic.stat().st_size > 0
            ):
                _merge_episodic_dedup(existing_episodic, staged_episodic)

        # ---------------- SMOKE_TEST ----------------
        # Construct a PersistentAgent over the freshly-loaded manifest (no
        # SQLite path → in-memory backend) and confirm identity_drift is
        # zero. Catches any path where reconstruct + merge silently drove
        # identity off the claimed value (would also fail VERIFY_IDENTITY_HASH;
        # SMOKE_TEST is the belt to that test's braces).
        from aeros.runtime.persistent_agent import PersistentAgent

        agent = PersistentAgent(recomputed_manifest, episodic_db=None)
        try:
            if agent.identity_drift() != 0.0:
                print(
                    "error: smoke test failed — identity drift detected after import",
                    file=sys.stderr,
                )
                _rollback()
                return 6
        finally:
            agent.close()

        # ---------------- COMMIT ----------------
        # Per-file rename out of staging into target so partially-existing
        # targets (the same-identity merge case) get atomically swapped
        # one entry at a time. signature.sig is intentionally not
        # installed — it lives only inside the .aer bundle.
        for name in _IMPORT_INSTALL_FILES:
            src = staging / name
            if not src.exists():
                continue
            dst = target / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.replace(src, dst)
        shutil.rmtree(staging, ignore_errors=True)

        # ---------------- AUDIT_CHAIN_CONTINUITY (spec §6 / AC-3..AC-5) ----------------
        # AC-3: emit a single migration_import episodic record into the
        # newly-installed target/episodic.db. The row's identity_hash is
        # the post-migration hash (AC-4 — equal to the source hash by §5),
        # the timestamp is now-UTC (AC-5), and the payload captures the
        # cross-deployment lineage operators need for forensic queries.
        # Done after COMMIT so a failed import never leaves a stranded
        # marker in a half-migrated target.
        _record_migration_import(
            target_episodic_db=target / "episodic.db",
            identity_hash=recomputed_manifest.identity_hash,
            source_identity_hash=str(claimed_identity or ""),
            bundle_version=str(meta_data.get("bundle_version", "")),
            source_bridge_id=str(meta_data.get("source_bridge_id", "")),
            episodic_rows_imported=int(source_episodic_rows),
            semantic_rows_imported=int(meta_data.get("semantic_row_count", 0) or 0),
            import_tool_version=str(meta_data.get("export_tool_version", "")) or "unknown",
        )

    except Exception as exc:  # noqa: BLE001 — fail-fast rollback
        _rollback()
        print(f"error: import failed: {exc}", file=sys.stderr)
        return 1

    print(f"imported: {bundle_path} → {target}", file=sys.stderr)
    print(f"  identity_hash={recomputed_manifest.identity_hash}", file=sys.stderr)
    print(f"  episodic_rows_imported={source_episodic_rows}", file=sys.stderr)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Implement ``aeros migrate verify`` — dry-run.

    Runs only LOAD + VERIFY_SIG + VERIFY_IDENTITY_HASH and never writes
    to a target directory. Returns the same fail-fast exit codes as
    ``cmd_import`` for the steps it performs (2 for sig issues, 3 for
    identity / digest mismatch). Useful as a pre-flight before a real
    import, or as a CI gate on signed bundles in transit.
    """
    bundle_path = Path(args.bundle).expanduser().resolve()
    trusted_keys_dir = (
        Path(args.trusted_keys_dir).expanduser().resolve()
        if args.trusted_keys_dir
        else _DEFAULT_TRUSTED_KEYS_DIR
    )

    if not bundle_path.is_file():
        print(f"error: bundle not found: {bundle_path}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="aeros-verify-") as workdir_str:
        workdir = Path(workdir_str)
        with tarfile.open(bundle_path, "r:gz") as tf:
            tf.extractall(workdir, filter="data")

        manifest_path = workdir / "manifest.json"
        sig_path = workdir / "signature.sig"
        if not manifest_path.is_file() or not sig_path.is_file():
            print(
                "error: bundle missing manifest.json or signature.sig",
                file=sys.stderr,
            )
            return 1
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        pubkey, _signature, embedded_digest = _verify_envelope_against_trust(
            sig_path, trusted_keys_dir
        )
        if pubkey is None or embedded_digest is None:
            print(
                "error: signature.sig is missing, malformed, untrusted, or invalid; "
                f"trusted_keys_dir={trusted_keys_dir}",
                file=sys.stderr,
            )
            return 2

        recomputed_digest = compute_signature_digest(workdir)
        if recomputed_digest != embedded_digest:
            print(
                "error: bundle digest mismatch — files were altered after signing",
                file=sys.stderr,
            )
            return 3
        try:
            recomputed_manifest = _recompute_manifest_from_bundle(
                workdir, manifest_data
            )
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            print(f"error: cannot recompute identity: {exc}", file=sys.stderr)
            return 3
        claimed_identity = manifest_data.get("identity_hash")
        if recomputed_manifest.identity_hash != claimed_identity:
            print(
                f"error: identity_hash mismatch (manifest claims {claimed_identity}, "
                f"recomputed {recomputed_manifest.identity_hash})",
                file=sys.stderr,
            )
            return 3

    print(f"verified: {bundle_path}", file=sys.stderr)
    print(f"  identity_hash={claimed_identity}", file=sys.stderr)
    return 0


# --------------------- argparse wiring --------------------- #


def register_migrate_commands(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    """Wire the ``aeros migrate <subcommand>`` tree onto the top-level parser.

    Registers ``export`` plus ``import`` / ``verify``
   . Audit-chain continuity is still
    deferred and will register on this same sub-subparser group.
    """
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Cross-deployment migration: export / import / verify .aer bundles",
    )
    migrate_sub = migrate_parser.add_subparsers(dest="migrate_command", required=True)

    exp = migrate_sub.add_parser(
        "export",
        help="Export the running bridge's identity + episodic state as a .aer bundle",
    )
    exp.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the .aer bundle (gzip tarball)",
    )
    exp.add_argument(
        "--bridge-dir",
        default=str(_DEFAULT_BRIDGE_DIR),
        help="Bridge root directory (default: bridge/ in the repo)",
    )
    exp.add_argument(
        "--episodic-db",
        default=None,
        help="Override episodic.db path (default: ~/.aeros/episodic.db)",
    )
    exp.add_argument(
        "--signing-key",
        default=None,
        help="Override Ed25519 signing key path (default: ~/.aeros/keys/export_signing.key)",
    )
    exp.add_argument(
        "--trusted-keys-dir",
        default=None,
        help="Override trusted-keys dir for self-trust bootstrap "
        "(default: ~/.aeros/trusted_keys/)",
    )
    exp.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a fresh Ed25519 signing key if one does not exist "
        "(also installs the matching pubkey into --trusted-keys-dir)",
    )
    exp.add_argument(
        "--signer-name",
        default=None,
        help="Optional human-readable signer label embedded in signature.sig",
    )
    exp.add_argument(
        "--runtime-version",
        default=None,
        help="Override the runtime_version field in IdentityManifest "
        "(default: 0.5.0, matching bridge lifespan)",
    )
    exp.add_argument(
        "--expected-identity-hash",
        default=None,
        help="Abort if the computed identity_hash doesn't match this value "
        "(used to confirm parity with a running bridge)",
    )
    exp.set_defaults(func=cmd_export)

    imp = migrate_sub.add_parser(
        "import",
        help="Import a .aer bundle into a target deployment "
        "(verify + reconstruct + merge episodic + smoke-test + commit)",
    )
    imp.add_argument(
        "--bundle",
        required=True,
        help="Path to the .aer bundle to import",
    )
    imp.add_argument(
        "--target",
        required=True,
        help="Target directory to install configs + episodic.db into "
        "(e.g. ~/.aeros/imports/<robot>/)",
    )
    imp.add_argument(
        "--trusted-keys-dir",
        default=None,
        help="Override trusted-keys dir for signature verification "
        "(default: ~/.aeros/trusted_keys/)",
    )
    imp.add_argument(
        "--migrate",
        action="store_true",
        help="Allow takeover when the target already holds a different "
        "identity_hash (refuses without this flag — exits 5)",
    )
    imp.set_defaults(func=cmd_import)

    ver = migrate_sub.add_parser(
        "verify",
        help="Dry-run a .aer bundle: signature + identity hash; never writes",
    )
    ver.add_argument(
        "--bundle",
        required=True,
        help="Path to the .aer bundle to verify",
    )
    ver.add_argument(
        "--trusted-keys-dir",
        default=None,
        help="Override trusted-keys dir for signature verification "
        "(default: ~/.aeros/trusted_keys/)",
    )
    ver.set_defaults(func=cmd_verify)
