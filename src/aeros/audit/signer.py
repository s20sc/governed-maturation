"""Ed25519 signer for ``PersonaAdaptiveEvent`` cause_id chain entries.

the state contract promotes signing to mandatory from the first
``persona_adaptation`` row, so every auto-emit and operator
sign-event routes through this module. Reuses the marketplace
Ed25519 primitives (``aeros.marketplace.signing``) — keypair generation,
PEM load, ``sign_payload`` / ``verify_signature`` — so we don't ship a
parallel implementation and the same algorithmic surface keeps the audit
trail interoperable with Layer 3 export signing.

The signed bytes are a canonical JSON serialisation of the event's
structural surface (``kind``, ``capability``, ``delta``,
``triggering_cause_ids``, ``timestamp``, ``schema_version``, ``applied``).
Tampering with any of these breaks the signature without re-signing.
The Pydantic model itself is frozen with ``extra="forbid"``,
so injecting an unsigned field via JSON also fails validation upstream.

What this module does NOT do (deferred to the chain verifier):
- Verifier callable wired into ``persona_resolver.effective_persona``.
- Per-row signature persistence schema in ``episodic.db`` (stored
  in the existing ``payload_json`` blob; a later change may promote
  to a typed column for indexed verify queries).
- Discovery of trusted-keys directory at resolver read time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aeros.marketplace.signing import (
    KeyFormatError,
    load_private_key,
    sign_payload,
    verify_signature,
)

if TYPE_CHECKING:
    from aeros.runtime.persona_event import PersonaAdaptiveEvent


@dataclass(frozen=True)
class EventSignature:
    """Per-row signature artefacts written into ``episodic.db`` payload.

    ``fingerprint`` is the SHA-256 of the public key truncated to the
    first 16 hex chars — short enough for log lines, long enough that
    collisions across a fleet are negligible.
    """

    signature_hex: str
    signer_pubkey_hex: str
    fingerprint: str


def canonical_event_payload(event: "PersonaAdaptiveEvent") -> bytes:
    """Return the deterministic byte sequence signed for ``event``.

    Includes everything that affects fold semantics:
      - kind, capability, delta, triggering_cause_ids
      - timestamp (ISO 8601), schema_version
      - applied (Q5 flag — flipping it post-sign would silently bypass
        the safety-floor skip, so it must enter the signed surface)

    ``operator_signature`` is deliberately excluded: relax-direction
    events sign the same surface, then attach the operator
    signature. The auto-emit path leaves the field None.
    """
    payload = {
        "kind": event.kind,
        "capability": event.capability,
        # ``event.delta`` is a ``MappingProxyType`` (deep immutability);
        # unwrap to a real dict so ``json.dumps`` can serialize it.
        # ``sort_keys=True`` below makes the resulting bytes byte-equal
        # to the pre-immutability representation.
        "delta": dict(event.delta),
        "triggering_cause_ids": list(event.triggering_cause_ids),
        "timestamp": event.timestamp.isoformat(),
        "schema_version": event.schema_version,
        "applied": event.applied,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_event_digest(event: "PersonaAdaptiveEvent") -> bytes:
    """SHA-256 over :func:`canonical_event_payload`. Returns 32 raw bytes."""
    return hashlib.sha256(canonical_event_payload(event)).digest()


def key_fingerprint(public_key_bytes: bytes) -> str:
    """Short stable identifier for a public key (first 16 hex chars of SHA-256)."""
    if len(public_key_bytes) != 32:
        raise KeyFormatError(
            f"Ed25519 public key must be 32 bytes (got {len(public_key_bytes)})"
        )
    return hashlib.sha256(public_key_bytes).hexdigest()[:16]


@dataclass(frozen=True)
class AuditSigner:
    """Holds an Ed25519 keypair for the agent's resident signing key.

    The resident key path is the same one Layer 3 export signing uses
    (``~/.aeros/keys/export_signing.key`` by default; see
    ``aeros.cli.migrate._ensure_signing_key``). Co-locating audit and
    export signing under one key means a Layer 3 import that trusts the
    source's pubkey also trusts the source's auto-emitted persona
    events, with no second trust pass.
    """

    private_key_bytes: bytes
    public_key_bytes: bytes

    def __post_init__(self) -> None:
        if len(self.private_key_bytes) != 32:
            raise KeyFormatError(
                f"Ed25519 private key must be 32 bytes "
                f"(got {len(self.private_key_bytes)})"
            )
        if len(self.public_key_bytes) != 32:
            raise KeyFormatError(
                f"Ed25519 public key must be 32 bytes "
                f"(got {len(self.public_key_bytes)})"
            )

    @classmethod
    def from_private_key_file(cls, path: Path) -> "AuditSigner":
        priv_raw = load_private_key(path)
        priv = Ed25519PrivateKey.from_private_bytes(priv_raw)
        pub_raw = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(private_key_bytes=priv_raw, public_key_bytes=pub_raw)

    @property
    def public_key_hex(self) -> str:
        return self.public_key_bytes.hex()

    @property
    def fingerprint(self) -> str:
        return key_fingerprint(self.public_key_bytes)

    def sign(self, event: "PersonaAdaptiveEvent") -> EventSignature:
        digest = compute_event_digest(event)
        sig = sign_payload(self.private_key_bytes, digest)
        return EventSignature(
            signature_hex=sig.hex(),
            signer_pubkey_hex=self.public_key_bytes.hex(),
            fingerprint=self.fingerprint,
        )


def load_resident_signer(private_key_path: Path) -> AuditSigner:
    """Load the resident Ed25519 keypair from a PEM file.

    Caller's responsibility to provision the key file (CLIs
    plus the existing ``aeros.cli.migrate`` keygen path do this; the
    bridge lifespan reads the same path).
    """
    return AuditSigner.from_private_key_file(private_key_path)


def sign_event(
    event: "PersonaAdaptiveEvent",
    *,
    private_key_bytes: bytes,
    public_key_bytes: bytes,
) -> EventSignature:
    return AuditSigner(
        private_key_bytes=private_key_bytes,
        public_key_bytes=public_key_bytes,
    ).sign(event)


def verify_event(
    event: "PersonaAdaptiveEvent",
    *,
    signature_hex: str,
    signer_pubkey_hex: str,
) -> bool:
    """Return True iff ``signature_hex`` is a valid signature for ``event``.

    Fail-closed on malformed hex or wrong key length. The chain verifier wires this
    into ``persona_resolver`` as the ``CauseChainVerifier`` callable so
    chain-verification enforcement flips on without any resolver-side change.
    """
    try:
        sig_bytes = bytes.fromhex(signature_hex)
        pub_bytes = bytes.fromhex(signer_pubkey_hex)
    except ValueError:
        return False
    if len(pub_bytes) != 32:
        return False
    digest = compute_event_digest(event)
    try:
        return verify_signature(pub_bytes, digest, sig_bytes)
    except KeyFormatError:
        return False
