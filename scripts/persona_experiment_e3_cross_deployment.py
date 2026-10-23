#!/usr/bin/env python3
"""E3 driver — cross-deployment migration round-trip (paper artifact).

Drives the Layer 3 ``aeros migrate
export`` / ``aeros migrate import`` flow with a one-year accumulated
reflection chain so the paper can cite the the state contract invariance:
identity hash byte-equal across the migration boundary, the resolver
fold yields a byte-equal :class:`PersonaProfile.adaptive` overlay at
the destination, and the bundle stays under the
10 MB envelope despite a year of audit rows.

Stages:

1. **Bridge A scaffold** — write a self-contained ``bridge_dir/configs``
   tree (persona / env_policies / policy / one capability contract)
   plus an empty staging episodic.db.
2. **Accumulate** — instantiate :class:`PersistentAgent` against the
   staging DB and run ~10 reflections / year via the in-process
   ``LLMCounterfactualBackend`` stub (same shape as the E1 driver but
   trimmed to a one-year horizon for E3 cost discipline). Capture
   bridge A's identity_hash + folded adaptive overlay.
3. **Export** — call ``aeros.cli.migrate.cmd_export`` against the same
   bridge_dir; stat the bundle for the 10 MB envelope check.
4. **Import** — call ``aeros.cli.migrate.cmd_import`` into bridge B.
   The migrate CLI verifies signature, recomputes identity hash, and
   refuses any tamper.
5. **Re-fold** — load bridge B's installed manifest + episodic chain,
   re-fold persona_adaptation events through the resolver against a
   freshly-loaded :class:`PersonaCore`, capture bridge B's adaptive
   overlay.
6. **Compare** — assert identity_hash byte-equal (Theorem 1) and the
   adaptive overlays byte-equal under JSON canonical form (the state contract
   reflection summary continuity).

Output JSON lands at ``out/e3_results.json`` with all three
checks plus bundle size delta. Last stdout line is a single-line JSON
metrics object so a CI runner can grep it.
"""

# Identity labels are fixed synthetic inputs to the manifest digest.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aeros.audit.signer import AuditSigner
from aeros.cli import migrate
from aeros.runtime.active_dreaming import (
    DEFAULT_PROMPT_PATH,
    ENABLED_ENV,
    REFLECTION_SUMMARY_KIND,
    ActiveDreamingRunner,
    LLMCounterfactualBackend,
)
from aeros.runtime.persistent_agent import (
    IdentityManifest,
    PersistentAgent,
    hash_ecm_registry,
    hash_env_policies,
    hash_persona,
    hash_policy_rules,
)
from aeros.runtime.persona import load_persona
from aeros.runtime.persona_emit import (
    PERSONA_ADAPTATION_KIND,
    PersonaAdaptationEmitter,
)
from aeros.runtime.persona_event import PersonaAdaptiveEvent
from aeros.runtime.persona_metrics import reset_for_tests
from aeros.runtime.persona_resolver import effective_persona


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "out" / "e3_results.json"

DEFAULT_TOTAL_DAYS = 365
QUICK_TOTAL_DAYS = 73  # two reflection ticks at d=36 / d=72
DAYS_PER_REFLECTION = 36
EXECUTIONS_PER_DAY = 12
HOT_FAILURE_RATE = 0.5
COLD_FAILURE_RATE = 0.02
DEFAULT_SEED = 42
BUNDLE_SIZE_LIMIT_MB = 10.0

# Capabilities that must agree with the contracts/*.json the bridge
# scaffold writes — the migrate CLI hashes the contract directory into
# the manifest.
CAPABILITIES: tuple[str, ...] = (
    "manipulation.grasp",
    "manipulation.place",
    "navigation.move_to",
    "perception.locate_object",
)


def _write_bridge_configs(bridge_dir: Path) -> None:
    """Mirror ``tests/cli/test_migrate.py::_write_bridge_configs``.

    Identical config bytes for both deployments so the recomputed
    identity_hash on the import side matches the source manifest. The
    contracts directory needs at least one ``*.json`` so the CLI's
    ``_read_capabilities`` fallback does not kick in.
    """
    cfg = bridge_dir / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "persona.yaml").write_text(
        "persona:\n"
        "  name: persona-E3\n"
        "  risk_appetite: balanced\n"
        "  human_check: on_high_risk\n"
        "  value_ordering: human_first\n"
        "  verbosity: balanced\n"
        "  tags: [persona, cross_deployment]\n",
        encoding="utf-8",
    )
    (cfg / "env_policies.yaml").write_text(
        "default_environment: sim_relaxed\n"
        "environments:\n"
        "  sim_relaxed:\n"
        "    description: persona E3 source environment\n"
        "    max_speed_mult: 1.0\n"
        "    max_force_mult: 1.0\n"
        "    approval_required: false\n"
        "    max_retries: 3\n"
        "    watcher_sensitivity: 0.5\n"
        "    forbidden_zones: []\n",
        encoding="utf-8",
    )
    (cfg / "policy.yaml").write_text(
        "policies:\n"
        "  - name: deny_restricted\n"
        "    when: { zone: restricted }\n"
        "    action: deny\n",
        encoding="utf-8",
    )
    contracts = cfg / "contracts"
    contracts.mkdir(exist_ok=True)
    for cap in CAPABILITIES:
        slug = cap.replace(".", "_")
        (contracts / f"{slug}.json").write_text(
            json.dumps(
                {
                    "capability": cap,
                    "ecm_name": f"PersonaECM_{slug}",
                    "contract_version": "1.0.0",
                }
            ),
            encoding="utf-8",
        )


def _manifest_from_bridge_configs(bridge_dir: Path) -> IdentityManifest:
    """Recompute the manifest the migrate CLI would assemble from disk.

    Delegates to the migrate CLI's private helpers so the bytes that
    enter ``hash_env_policies`` / ``hash_policy_rules`` /
    ``hash_ecm_registry`` are identical to the ones cmd_export uses;
    if we re-derived them in this script the identity hash would drift
    on any future config-shape addition.
    """
    persona_yaml = bridge_dir / "configs" / "persona.yaml"
    env_yaml = bridge_dir / "configs" / "env_policies.yaml"
    policy_yaml = bridge_dir / "configs" / "policy.yaml"

    persona_payload = load_persona(persona_yaml).core.to_manifest_payload()
    env_dict = migrate._build_env_policies_for_hash(env_yaml)
    policy_rules = migrate._read_policy_rules(policy_yaml)
    capabilities = migrate._read_capabilities(bridge_dir)

    return IdentityManifest(
        system_prompt=migrate.DEFAULT_SYSTEM_PROMPT,
        env_policies_hash=hash_env_policies(env_dict),
        policy_rules_hash=hash_policy_rules(policy_rules),
        ecm_registry_hash=hash_ecm_registry(capabilities),
        persona_hash=hash_persona(persona_payload),
        runtime_version="0.5.0",
    )


def _signer_for_seed(seed: int) -> AuditSigner:
    rng = random.Random(seed)
    priv_obj = Ed25519PrivateKey.from_private_bytes(rng.randbytes(32))
    priv_bytes = priv_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = priv_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return AuditSigner(private_key_bytes=priv_bytes, public_key_bytes=pub_bytes)


def _hot_capability_for_day(day: int) -> str:
    return CAPABILITIES[(day // 30) % len(CAPABILITIES)]


def _seed_day_episodes(
    agent: PersistentAgent,
    *,
    rng: random.Random,
    hot_capability: str,
) -> None:
    for _ in range(EXECUTIONS_PER_DAY):
        cap = rng.choice(CAPABILITIES)
        if cap == hot_capability:
            outcome = "failure" if rng.random() < HOT_FAILURE_RATE else "success"
        else:
            outcome = "failure" if rng.random() < COLD_FAILURE_RATE else "success"
        agent.record(
            kind="execution_result",
            summary=f"{cap}:{outcome}",
            payload={"capability": cap, "outcome": outcome},
        )


def _reflection_llm_call(prompt: str, deadline: float) -> tuple[str, int, int]:
    """Stub identical in shape to E1's — pick the hottest capability."""
    best: tuple[float, str] = (-1.0, CAPABILITIES[0])
    for line in prompt.splitlines():
        if "failure_rate=" not in line or ":" not in line:
            continue
        try:
            cap_part, rate_part = line.split(":", 1)
            cap = cap_part.strip().lstrip("- ").strip()
            rate = float(rate_part.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if rate > best[0]:
            best = (rate, cap)
    target = best[1]
    body = json.dumps(
        {
            "reflection_text": (
                f"E3 maturation reflection — proposing tighten on "
                f"{target} based on observed failure cluster."
            ),
            "proposed_deltas": [
                {
                    "target_kind": "skill_confidence_decrease",
                    "target_capability": target,
                    "proposed_delta": {"magnitude": 0.05},
                    "confidence": 0.7,
                }
            ],
            "derived_facts": [
                f"capability={target} drift_window=last_36_days"
            ],
        },
        sort_keys=True,
    )
    return body, len(prompt), len(body)


def _accumulate_year(
    *,
    bridge_dir: Path,
    episodic_db: Path,
    seed: int,
    total_days: int,
) -> dict[str, Any]:
    """Drive a real PersistentAgent against the on-disk episodic.db.

    Returns the source identity_hash, reflection counts, and the folded
    adaptive overlay so the post-import comparison has a baseline.
    """
    manifest = _manifest_from_bridge_configs(bridge_dir)

    episodic_db.parent.mkdir(parents=True, exist_ok=True)
    if episodic_db.exists():
        episodic_db.unlink()

    rng = random.Random(seed)
    signer = _signer_for_seed(seed)
    agent = PersistentAgent(
        manifest, episodic_capacity=16384, episodic_db=episodic_db
    )
    persona_core = load_persona(bridge_dir / "configs" / "persona.yaml").core
    emitter = PersonaAdaptationEmitter(
        agent,
        signer,
        failure_threshold=3,
        confidence_decrement=0.05,
        persona_core=persona_core,
    )

    clock_ref = [datetime(2026, 1, 1, tzinfo=UTC)]
    backend = LLMCounterfactualBackend(
        _reflection_llm_call, prompt_template_path=DEFAULT_PROMPT_PATH
    )
    runner = ActiveDreamingRunner(
        agent,
        signer,
        emitter,
        backend,
        episodic_window=200,
        clock=lambda: clock_ref[0],
    )

    start = clock_ref[0]
    reflection_attempts = 0
    reflections_with_event = 0
    for day in range(total_days):
        clock_ref[0] = start + timedelta(days=day, hours=2)
        hot = _hot_capability_for_day(day)
        _seed_day_episodes(agent, rng=rng, hot_capability=hot)
        if day % DAYS_PER_REFLECTION == 0 and day > 0:
            reflection_attempts += 1
            event = runner.run_reflection(agent.identity_hash)
            if event is not None and event.validation_result != "all_rejected":
                reflections_with_event += 1

    events = _persona_events(agent)
    profile = effective_persona(persona_core, events)
    adaptive_payload = profile.adaptive.to_bundle_payload()

    summary = {
        "source_identity_hash": agent.identity_hash,
        "reflection_attempts": reflection_attempts,
        "reflections_with_event": reflections_with_event,
        "reflection_summary_rows": len(
            agent.episodic.by_kind(REFLECTION_SUMMARY_KIND)
        ),
        "persona_adaptation_rows": len(events),
        "source_adaptive_payload": adaptive_payload,
        "source_adaptive_hash": _hash_adaptive(adaptive_payload),
    }

    agent.close()
    return summary


def _persona_events(agent: PersistentAgent) -> list[PersonaAdaptiveEvent]:
    out: list[PersonaAdaptiveEvent] = []
    for row in agent.episodic.by_kind(PERSONA_ADAPTATION_KIND):
        try:
            out.append(PersonaAdaptiveEvent(**row.payload["event"]))
        except (TypeError, KeyError, ValueError):
            continue
    return out


def _persona_events_from_db(
    db_path: Path, *, identity_hash: str
) -> list[PersonaAdaptiveEvent]:
    """Read persona_adaptation rows from the imported episodic.db.

    The migrate CLI installs the bundle's episodic.db verbatim into the
    target — partitioned by ``identity_hash`` — so the same scan the
    runtime would do reproduces the source event log on bridge B.
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        return []
    out: list[PersonaAdaptiveEvent] = []
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT payload_json FROM episodic "
            "WHERE identity_hash = ? AND kind = ? "
            "ORDER BY id",
            (identity_hash, PERSONA_ADAPTATION_KIND),
        )
        for (payload_json,) in cur.fetchall():
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            ev_payload = payload.get("event") if isinstance(payload, dict) else None
            if not isinstance(ev_payload, dict):
                continue
            try:
                out.append(PersonaAdaptiveEvent(**ev_payload))
            except (TypeError, KeyError, ValueError):
                continue
    finally:
        conn.close()
    return out


def _hash_adaptive(payload: dict[str, Any]) -> str:
    """SHA-256 over the JSON canonical form of an adaptive overlay.

    Used as the byte-equal proxy for the ``effective_persona``
    invariance claim. Equal hashes ⇒ equal payloads, modulo SHA-256
    collisions (vanishingly negligible at the 32-byte digest).
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _empty_baseline_bundle_size(workdir: Path) -> int:
    """Export a fresh bridge with an empty episodic.db for the size delta.

    The "delta" the parent acceptance criterion bounds is the size cost
    of one year of accumulated reflections vs. a baseline empty bundle.
    Both bundles are produced with identical bridge configs so the only
    moving piece is the episodic chain.
    """
    bridge_dir = workdir / "baseline_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    _write_bridge_configs(bridge_dir)
    empty_db = workdir / "baseline_episodic.db"
    empty_db.write_bytes(b"")
    bundle = workdir / "baseline_bundle.aer"
    keys_dir = workdir / "baseline_keys"
    trusted = workdir / "baseline_trusted"
    rc = migrate.cmd_export(
        _ns_export(
            output=str(bundle),
            bridge_dir=str(bridge_dir),
            episodic_db=str(empty_db),
            signing_key=str(keys_dir / "signing.key"),
            trusted_keys_dir=str(trusted),
        )
    )
    if rc != 0:
        raise RuntimeError(f"baseline export failed rc={rc}")
    return bundle.stat().st_size


def _ns_export(**overrides: Any):
    import argparse as _argparse

    defaults: dict[str, Any] = {
        "output": None,
        "bridge_dir": None,
        "episodic_db": None,
        "signing_key": None,
        "trusted_keys_dir": None,
        "generate_key": True,
        "signer_name": None,
        "runtime_version": None,
        "expected_identity_hash": None,
    }
    defaults.update(overrides)
    return _argparse.Namespace(**defaults)


def _ns_import(**overrides: Any):
    import argparse as _argparse

    defaults: dict[str, Any] = {
        "bundle": None,
        "target": None,
        "trusted_keys_dir": None,
        "migrate": False,
    }
    defaults.update(overrides)
    return _argparse.Namespace(**defaults)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E3 driver: cross-deployment migration "
            "round-trip with one year of reflections."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a one-month accumulation instead of the full year",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="deterministic seed for episodic activity (default 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="path to JSON metrics file (default out/e3_results.json)",
    )
    args = parser.parse_args(argv)

    started_at = time.monotonic()
    os.environ[ENABLED_ENV] = "true"
    reset_for_tests()

    total_days = QUICK_TOTAL_DAYS if args.quick else DEFAULT_TOTAL_DAYS

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="persona-E3-") as tmp_str:
        tmp = Path(tmp_str)
        bridge_a = tmp / "bridge_A"
        bridge_b = tmp / "bridge_B"
        bridge_a.mkdir(parents=True)
        _write_bridge_configs(bridge_a)
        episodic_a = bridge_a / "data" / "episodic.db"

        accumulation = _accumulate_year(
            bridge_dir=bridge_a,
            episodic_db=episodic_a,
            seed=args.seed,
            total_days=total_days,
        )

        bundle_path = tmp / "persona_e3.aer"
        signing_key = tmp / "keys" / "export_signing.key"
        trusted_keys_dir = tmp / "trusted_keys"
        rc = migrate.cmd_export(
            _ns_export(
                output=str(bundle_path),
                bridge_dir=str(bridge_a),
                episodic_db=str(episodic_a),
                signing_key=str(signing_key),
                trusted_keys_dir=str(trusted_keys_dir),
                expected_identity_hash=accumulation["source_identity_hash"],
            )
        )
        if rc != 0:
            print(f"ERROR: cmd_export rc={rc}", file=sys.stderr)
            return 2
        bundle_bytes = bundle_path.stat().st_size

        baseline_bytes = _empty_baseline_bundle_size(tmp / "baseline_workdir")
        bundle_delta_bytes = bundle_bytes - baseline_bytes
        bundle_size_mb = bundle_bytes / (1024 * 1024)
        bundle_delta_mb = bundle_delta_bytes / (1024 * 1024)

        rc = migrate.cmd_import(
            _ns_import(
                bundle=str(bundle_path),
                target=str(bridge_b),
                trusted_keys_dir=str(trusted_keys_dir),
            )
        )
        if rc != 0:
            print(f"ERROR: cmd_import rc={rc}", file=sys.stderr)
            return 2

        installed_manifest = json.loads(
            (bridge_b / "manifest.json").read_text(encoding="utf-8")
        )
        dest_identity_hash = installed_manifest["identity_hash"]
        dest_episodic_db = bridge_b / "episodic.db"

        # Bridge B re-loads PersonaCore from the installed persona.yaml
        # exactly as a freshly-booted bridge would; the resolver fold
        # over the imported events must produce a byte-equal overlay.
        dest_persona_core = load_persona(bridge_b / "persona.yaml").core
        dest_events = _persona_events_from_db(
            dest_episodic_db, identity_hash=dest_identity_hash
        )
        dest_profile = effective_persona(dest_persona_core, dest_events)
        dest_adaptive_payload = dest_profile.adaptive.to_bundle_payload()
        dest_adaptive_hash = _hash_adaptive(dest_adaptive_payload)

        identity_hash_byte_equal = (
            accumulation["source_identity_hash"] == dest_identity_hash
        )
        adaptive_byte_equal = (
            accumulation["source_adaptive_hash"] == dest_adaptive_hash
        )
        bundle_under_envelope = bundle_delta_mb < BUNDLE_SIZE_LIMIT_MB

        if not identity_hash_byte_equal:
            failures.append(
                f"identity_hash mismatch: source={accumulation['source_identity_hash']} "
                f"dest={dest_identity_hash}"
            )
        if not adaptive_byte_equal:
            failures.append(
                f"adaptive overlay mismatch: source={accumulation['source_adaptive_hash']} "
                f"dest={dest_adaptive_hash}"
            )
        if not bundle_under_envelope:
            failures.append(
                f"bundle delta {bundle_delta_mb:.3f} MB ≥ {BUNDLE_SIZE_LIMIT_MB} MB envelope"
            )

        metrics = {
            "experiment": "e3_cross_deployment",
            "seed": args.seed,
            "quick_mode": args.quick,
            "total_days": total_days,
            "source_identity_hash": accumulation["source_identity_hash"],
            "dest_identity_hash": dest_identity_hash,
            "identity_hash_byte_equal": identity_hash_byte_equal,
            "source_adaptive_hash": accumulation["source_adaptive_hash"],
            "dest_adaptive_hash": dest_adaptive_hash,
            "effective_persona_byte_equal": adaptive_byte_equal,
            "source_adaptive_payload": accumulation["source_adaptive_payload"],
            "dest_adaptive_payload": dest_adaptive_payload,
            "reflection_summary_rows": accumulation["reflection_summary_rows"],
            "persona_adaptation_rows_source": accumulation["persona_adaptation_rows"],
            "persona_adaptation_rows_dest": len(dest_events),
            "bundle_size_bytes": bundle_bytes,
            "baseline_bundle_size_bytes": baseline_bytes,
            "bundle_size_mb": round(bundle_size_mb, 4),
            "bundle_size_delta_mb": round(bundle_delta_mb, 4),
            "bundle_size_limit_mb": BUNDLE_SIZE_LIMIT_MB,
            "bundle_size_under_envelope": bundle_under_envelope,
            "wall_clock_seconds": round(time.monotonic() - started_at, 4),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

    if failures:
        print(json.dumps(metrics, sort_keys=True))
        for msg in failures:
            print(f"ERROR: E3 driver: {msg}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
