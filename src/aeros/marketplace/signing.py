"""Ed25519 signing primitives for ECM packages.

The marketplace's first piece of cryptographic infrastructure: an author
generates a keypair, signs an ``.aecm`` archive with it, and a bridge
operator verifies the signature against a directory of trusted public
keys before installing.

Format choices:

* **Algorithm.** Ed25519 — 32-byte private key, 32-byte public key,
  64-byte signature. Modern, fast, no nonce-reuse failure mode.
* **Serialization.** Private keys are PEM-encoded PKCS8 (no passphrase
  by default; the file's mode is 0600). Public keys are PEM-encoded
  SubjectPublicKeyInfo. Both are flat-file artifacts that
  ``cryptography`` reads natively.
* **Detached signature.** A JSON file ``<pkg>.aecm.sig`` sits next to
  the ``.aecm`` archive carrying the SHA-256 of the archive bytes, the
  signature over that digest, and the signer's raw public key bytes
  (hex). The archive itself is unchanged, so existing tooling that
  unzips ``.aecm`` files keeps working without signature awareness.
* **Trust resolution.** Verification compares the signer's public key
  bytes to the on-disk PEM files in a ``trusted_keys/`` directory. Any
  match accepts the signature. Untrusted keys reject even with a
  cryptographically valid signature.

Features not implemented by this module:
* Sigstore / transparency log integration.
* Revocation lists (CRL / OCSP).
* Network registry publish / fetch.
* Quota enforcement.


"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIG_SCHEMA_VERSION = "v1"
SIG_ALGORITHM = "ed25519"
SIG_FILE_SUFFIX = ".sig"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SigningError(Exception):
    """Base class for signing / verification failures."""


class KeyFormatError(SigningError):
    """Raised when a key file is unreadable or not Ed25519."""


class SignatureFileError(SigningError):
    """Raised when the .sig JSON is missing fields or malformed."""


class SignatureMismatchError(SigningError):
    """Raised when the cryptographic verify fails."""


class UntrustedKeyError(SigningError):
    """Raised when the signer's public key is not in the trust dir."""


# ---------------------------------------------------------------------------
# Keypair generation + serialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keypair:
    """An Ed25519 keypair held in raw 32-byte form.

    Use :meth:`generate` to create a fresh pair, :meth:`save` to persist
    (private as PKCS8 PEM with mode 0600, public as SubjectPublicKeyInfo
    PEM), or pair the raw bytes with the ``load_*`` helpers.
    """

    private_key_bytes: bytes
    public_key_bytes: bytes

    @classmethod
    def generate(cls) -> Keypair:
        priv = Ed25519PrivateKey.generate()
        priv_raw = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_raw = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(private_key_bytes=priv_raw, public_key_bytes=pub_raw)

    @property
    def public_key_hex(self) -> str:
        return self.public_key_bytes.hex()

    def save(self, *, private_path: Path, public_path: Path | None = None) -> None:
        """Write private key (PKCS8 PEM) and optionally a public key (PEM).

        Public key path defaults to ``<private_path>.pub``. The private
        file's mode is forced to 0600 after write.
        """
        priv_obj = Ed25519PrivateKey.from_private_bytes(self.private_key_bytes)
        priv_pem = priv_obj.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_bytes(priv_pem)
        os.chmod(private_path, stat.S_IRUSR | stat.S_IWUSR)

        if public_path is None:
            public_path = private_path.with_suffix(private_path.suffix + ".pub")
        pub_obj = priv_obj.public_key()
        pub_pem = pub_obj.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        public_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.write_bytes(pub_pem)


def load_private_key(path: Path) -> bytes:
    """Read a PEM-encoded Ed25519 private key from disk; return raw 32 bytes."""
    pem = path.read_bytes()
    try:
        priv = serialization.load_pem_private_key(pem, password=None)
    except (ValueError, TypeError) as exc:
        raise KeyFormatError(f"cannot parse private key {path}: {exc}") from exc
    if not isinstance(priv, Ed25519PrivateKey):
        raise KeyFormatError(f"key at {path} is not Ed25519 (got {type(priv).__name__})")
    return priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def load_public_key(path: Path) -> bytes:
    """Read a PEM-encoded Ed25519 public key from disk; return raw 32 bytes."""
    pem = path.read_bytes()
    try:
        pub = serialization.load_pem_public_key(pem)
    except (ValueError, TypeError) as exc:
        raise KeyFormatError(f"cannot parse public key {path}: {exc}") from exc
    if not isinstance(pub, Ed25519PublicKey):
        raise KeyFormatError(f"key at {path} is not Ed25519 (got {type(pub).__name__})")
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# ---------------------------------------------------------------------------
# Sign / verify primitives
# ---------------------------------------------------------------------------


def compute_aecm_digest(aecm_path: Path) -> bytes:
    """Return the SHA-256 digest of an ``.aecm`` archive (raw 32 bytes)."""
    h = hashlib.sha256()
    with aecm_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def sign_payload(private_key_bytes: bytes, payload: bytes) -> bytes:
    """Sign *payload* with a raw 32-byte Ed25519 private key. Returns 64-byte signature."""
    if len(private_key_bytes) != 32:
        raise KeyFormatError(f"Ed25519 private key must be 32 bytes (got {len(private_key_bytes)})")
    priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return priv.sign(payload)


def verify_signature(public_key_bytes: bytes, payload: bytes, signature: bytes) -> bool:
    """Return True iff *signature* is a valid Ed25519 sig of *payload* under *public_key*."""
    if len(public_key_bytes) != 32:
        raise KeyFormatError(f"Ed25519 public key must be 32 bytes (got {len(public_key_bytes)})")
    if len(signature) != 64:
        return False
    pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        pub.verify(signature, payload)
    except InvalidSignature:
        return False
    return True


# ---------------------------------------------------------------------------
# .sig file: JSON envelope around (digest, signature, signer pubkey, metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureEnvelope:
    """Contents of an ``.aecm.sig`` file. JSON-serialisable."""

    schema_version: str
    algorithm: str
    aecm_sha256_hex: str
    signature_hex: str
    signer_pubkey_hex: str
    signed_at: str
    signer_name: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "algorithm": self.algorithm,
                "aecm_sha256": self.aecm_sha256_hex,
                "signature": self.signature_hex,
                "signer_pubkey": self.signer_pubkey_hex,
                "signed_at": self.signed_at,
                "signer_name": self.signer_name,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> SignatureEnvelope:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SignatureFileError(f"signature file is not valid JSON: {exc}") from exc
        for required in (
            "schema_version",
            "algorithm",
            "aecm_sha256",
            "signature",
            "signer_pubkey",
            "signed_at",
        ):
            if required not in data:
                raise SignatureFileError(f"signature file missing required field {required!r}")
        if data["schema_version"] != SIG_SCHEMA_VERSION:
            raise SignatureFileError(
                f"unsupported signature schema_version {data['schema_version']!r};"
                f" this build understands {SIG_SCHEMA_VERSION!r}"
            )
        if data["algorithm"] != SIG_ALGORITHM:
            raise SignatureFileError(
                f"unsupported signature algorithm {data['algorithm']!r};"
                f" this build understands {SIG_ALGORITHM!r}"
            )
        return cls(
            schema_version=data["schema_version"],
            algorithm=data["algorithm"],
            aecm_sha256_hex=data["aecm_sha256"],
            signature_hex=data["signature"],
            signer_pubkey_hex=data["signer_pubkey"],
            signed_at=data["signed_at"],
            signer_name=data.get("signer_name"),
        )


def sign_aecm(
    aecm_path: Path,
    *,
    private_key_bytes: bytes,
    public_key_bytes: bytes,
    signer_name: str | None = None,
    sig_path: Path | None = None,
) -> Path:
    """Sign an ``.aecm`` archive; return the path to the written ``.sig`` file.

    The signer's public key is embedded in the envelope so verification
    can both check the signature AND map the signer to the trusted-keys
    directory by raw public key bytes.
    """
    digest = compute_aecm_digest(aecm_path)
    signature = sign_payload(private_key_bytes, digest)
    envelope = SignatureEnvelope(
        schema_version=SIG_SCHEMA_VERSION,
        algorithm=SIG_ALGORITHM,
        aecm_sha256_hex=digest.hex(),
        signature_hex=signature.hex(),
        signer_pubkey_hex=public_key_bytes.hex(),
        signed_at=datetime.now(UTC).isoformat(),
        signer_name=signer_name,
    )
    target = sig_path or aecm_path.with_suffix(aecm_path.suffix + SIG_FILE_SUFFIX)
    target.write_text(envelope.to_json(), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Verification + trust resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a verify call, including which trusted key matched."""

    valid: bool
    signer_pubkey_hex: str
    matched_trusted_key_path: Path | None = None
    trusted: bool = False
    digest_hex: str = ""
    error: str | None = None
    signer_name: str | None = None
    signed_at: str | None = None
    notes: list[str] = field(default_factory=list)


def _load_trusted_keys(trusted_dir: Path) -> dict[str, Path]:
    """Map raw-pubkey-hex → file path for each ``*.pub`` PEM in *trusted_dir*.

    Returns an empty dict if the directory is missing or empty. Files
    that don't parse as Ed25519 public keys are skipped with their
    error retained as a note in the calling :class:`VerifyResult`.
    """
    if not trusted_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(trusted_dir.glob("*.pub")):
        try:
            raw = load_public_key(path)
        except KeyFormatError:
            continue
        out[raw.hex()] = path
    return out


def verify_aecm(
    aecm_path: Path,
    *,
    sig_path: Path | None = None,
    trusted_keys_dir: Path | None = None,
) -> VerifyResult:
    """Verify the detached signature on *aecm_path*.

    Returns a :class:`VerifyResult` describing both the cryptographic
    outcome and the trust resolution. Does NOT raise on failure — the
    caller decides whether to gate install on ``valid`` and ``trusted``.

    A missing ``.sig`` file returns ``valid=False`` with an ``error``
    string; the install-gate caller treats that as "unsigned" rather
    than as a tamper claim.
    """
    sig_target = sig_path or aecm_path.with_suffix(aecm_path.suffix + SIG_FILE_SUFFIX)
    if not sig_target.is_file():
        return VerifyResult(
            valid=False,
            signer_pubkey_hex="",
            error=f"signature file not found: {sig_target}",
        )
    try:
        envelope = SignatureEnvelope.from_json(sig_target.read_text(encoding="utf-8"))
    except SignatureFileError as exc:
        return VerifyResult(
            valid=False,
            signer_pubkey_hex="",
            error=str(exc),
        )

    digest = compute_aecm_digest(aecm_path)
    if digest.hex() != envelope.aecm_sha256_hex:
        return VerifyResult(
            valid=False,
            signer_pubkey_hex=envelope.signer_pubkey_hex,
            digest_hex=digest.hex(),
            signer_name=envelope.signer_name,
            signed_at=envelope.signed_at,
            error=(
                "aecm digest mismatch: archive content has changed since signing"
                f" (expected {envelope.aecm_sha256_hex}, got {digest.hex()})"
            ),
        )

    try:
        pubkey_bytes = bytes.fromhex(envelope.signer_pubkey_hex)
        signature = bytes.fromhex(envelope.signature_hex)
    except ValueError as exc:
        return VerifyResult(
            valid=False,
            signer_pubkey_hex=envelope.signer_pubkey_hex,
            digest_hex=digest.hex(),
            error=f"hex decode failed: {exc}",
        )

    if not verify_signature(pubkey_bytes, digest, signature):
        return VerifyResult(
            valid=False,
            signer_pubkey_hex=envelope.signer_pubkey_hex,
            digest_hex=digest.hex(),
            signer_name=envelope.signer_name,
            signed_at=envelope.signed_at,
            error="cryptographic signature does not verify against signer pubkey",
        )

    matched: Path | None = None
    notes: list[str] = []
    if trusted_keys_dir is not None:
        trusted = _load_trusted_keys(trusted_keys_dir)
        if not trusted:
            notes.append(f"trusted_keys directory has no usable PEMs: {trusted_keys_dir}")
        matched = trusted.get(envelope.signer_pubkey_hex)
    return VerifyResult(
        valid=True,
        signer_pubkey_hex=envelope.signer_pubkey_hex,
        matched_trusted_key_path=matched,
        trusted=matched is not None,
        digest_hex=digest.hex(),
        signer_name=envelope.signer_name,
        signed_at=envelope.signed_at,
        notes=notes,
    )
