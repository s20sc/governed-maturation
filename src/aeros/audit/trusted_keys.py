"""Trusted Ed25519 public-key directory + resident-key bootstrap.

This module lifts the trust-directory code out of ``aeros.cli.migrate`` so the
migrate CLI, the operator persona-event CLIs, and the
resolver's chain verifier all share one canonical surface for "load
the trusted ``*.pub`` files" and "ensure the agent's resident signing
key exists". Before the lift, ``migrate.py`` was the only home for
these helpers; reusing them from non-migrate callers meant either
importing private names or duplicating PEM-walk logic.

The on-disk layout is:

* ``~/.aeros/keys/export_signing.key`` — PEM-encoded Ed25519 private
  key, mode ``0600``. Same key signs Layer 3 export bundles AND
  ``persona_adaptation`` rows (the state contract reuse).
* ``~/.aeros/keys/export_signing.key.pub`` — matching public key.
* ``~/.aeros/trusted_keys/*.pub`` — every public key whose
  signatures this host accepts. Bootstrap puts the host's own pubkey
  here so a self-import works without manual key distribution.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aeros.marketplace.signing import (
    KeyFormatError,
    Keypair,
    load_private_key,
    load_public_key,
)

DEFAULT_RESIDENT_KEY_PATH = Path("~/.aeros/keys/export_signing.key").expanduser()
DEFAULT_TRUSTED_KEYS_DIR = Path("~/.aeros/trusted_keys").expanduser()


def load_trusted_pubkeys(trusted_dir: Path) -> dict[str, Path]:
    """Map raw-pubkey-hex → file path for each ``*.pub`` PEM in ``trusted_dir``.

    Returns an empty dict if the directory is missing or holds no
    parseable Ed25519 keys; the caller treats an empty dict as "nothing
    is trusted" rather than as a successful trust resolution. Files
    that fail to parse as Ed25519 public keys are skipped silently —
    a stray non-key file in the dir is a misconfiguration the caller
    surfaces, not a hard fail at load time.
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


def is_trusted_pubkey(trusted_dir: Path, pubkey_hex: str) -> bool:
    """Convenience wrapper: True iff ``pubkey_hex`` matches a PEM in ``trusted_dir``."""
    return pubkey_hex in load_trusted_pubkeys(trusted_dir)


def ensure_resident_keypair(
    signing_key_path: Path,
    *,
    trusted_keys_dir: Path,
    auto_generate: bool,
) -> tuple[bytes, bytes]:
    """Resolve / bootstrap the resident Ed25519 keypair.

    Returns ``(private_bytes_32, public_bytes_32)``. When the key file
    exists, loads it; otherwise (and only when ``auto_generate=True``)
    generates a fresh Ed25519 keypair, writes the private key to
    ``signing_key_path`` (mode 0600), and drops the matching public key
    into ``trusted_keys_dir`` so this host trusts its own bundles
    (installer-style bootstrap of trusted_keys/).
    """
    if signing_key_path.exists():
        priv = load_private_key(signing_key_path)
        pub_path = signing_key_path.with_suffix(signing_key_path.suffix + ".pub")
        if pub_path.exists():
            pub = load_public_key(pub_path)
        else:
            pub = Ed25519PrivateKey.from_private_bytes(priv).public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        return priv, pub

    if not auto_generate:
        raise FileNotFoundError(
            f"resident signing key not found: {signing_key_path}\n"
            "Pass --signing-key <path>, or run with --generate-key to "
            "create one (writes a fresh Ed25519 keypair under "
            f"{signing_key_path.parent} and trusts the matching pubkey "
            f"in {trusted_keys_dir})."
        )

    signing_key_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_keys_dir.mkdir(parents=True, exist_ok=True)
    kp = Keypair.generate()
    pub_path = signing_key_path.with_suffix(signing_key_path.suffix + ".pub")
    kp.save(private_path=signing_key_path, public_path=pub_path)
    os.chmod(signing_key_path, stat.S_IRUSR | stat.S_IWUSR)
    trusted_copy = trusted_keys_dir / f"{signing_key_path.stem}.pub"
    if not trusted_copy.exists():
        shutil.copyfile(pub_path, trusted_copy)
    return kp.private_key_bytes, kp.public_key_bytes
