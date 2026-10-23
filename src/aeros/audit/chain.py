"""Cause-id chain verification for ``PersonaAdaptiveEvent`` rows.

the state contract: every ``persona_adaptation`` row carries an Ed25519
signature over its structural surface (``aeros.audit.signer``) plus a
``triggering_cause_ids`` list pointing at the episodic-row predecessors
that justify it. The resolver folds an event into ``PersonaAdaptive``
only after the chain is verified — chain-verification enforcement (the state contract, point 6).

This module is the chain verifier the resolver reserved a hook for. The
shape is deliberately small:

* :class:`SignedPersonaEvent` — the ``(cause_id, event, signature_hex,
  signer_pubkey_hex)`` tuple a persona-adaptation row produces.
  ``cause_id`` is the row's stable id under which it was stored, so
  child events can reference it via ``triggering_cause_ids``.
* :func:`verify_chain` — recurse from a head envelope through every
  ancestor whose cause_id resolves into ``by_cause_id``. Bounded by
  ``max_depth`` (the state contract default ``8``) and gated by
  ``trusted_pubkeys``: a row whose ``signer_pubkey_hex`` is not in
  the caller-supplied trusted set is rejected even when the embedded
  signature is otherwise valid (chain-verification source-identity verification).
* :func:`make_chain_verifier` — adapt the chain check into the
  resolver's :data:`aeros.runtime.persona_resolver.CauseChainVerifier`
  callable shape (event → bool) so wiring it on flips chain-verification enforcement
  with no resolver edit.

Trust model: ``trusted_pubkeys`` is a ``Container[str]`` of the raw
hex-encoded Ed25519 public keys this host accepts (typically the
``.keys()`` of :func:`aeros.audit.trusted_keys.load_trusted_pubkeys`
applied to ``~/.aeros/trusted_keys/``). The parameter is required —
without it, a freshly-generated keypair could sign arbitrary adaptive
deltas and slide them past the resolver as legitimate. Callers that
genuinely have no trust constraint (test scaffolding, exploratory
fixtures) must opt in by passing the row's own pubkey or a curated
set; passing ``frozenset()`` short-circuits everything to ``False``.

Non-persona predecessors (e.g. an episodic message id, an evolution
audit row) are terminal: a cause_id that does not appear in
``by_cause_id`` is treated as a leaf rather than a verification
failure. This matches the state contract's "cause_id traces back to verifiable
episodic sources" — a persona-adaptation row signs its own surface and
its persona-adaptation parents; non-persona ancestors are out of scope
for the chain check.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass

from aeros.audit.signer import key_fingerprint, verify_event
from aeros.runtime.persona_event import PersonaAdaptiveEvent

DEFAULT_MAX_CHAIN_DEPTH = 8


@dataclass(frozen=True)
class SignedPersonaEvent:
    """Per-row audit envelope for a ``persona_adaptation`` episodic entry.

    ``cause_id`` is the episodic-row id under which the row is stored.
    Children reference it through ``PersonaAdaptiveEvent.triggering_cause_ids``;
    the chain verifier follows the references until it reaches either
    a leaf (a cause_id absent from ``by_cause_id``) or the configured
    depth cap.
    """

    cause_id: str
    event: PersonaAdaptiveEvent
    signature_hex: str
    signer_pubkey_hex: str

    @property
    def fingerprint(self) -> str:
        """Short pubkey fingerprint, or empty string when ``signer_pubkey_hex`` is malformed."""
        try:
            return key_fingerprint(bytes.fromhex(self.signer_pubkey_hex))
        except (ValueError, Exception):
            # ``key_fingerprint`` raises ``KeyFormatError`` (a subclass
            # of Exception) when the pubkey is the wrong length; an
            # empty/odd-length hex string raises ValueError on
            # ``bytes.fromhex``. Both cases are surfaced as "no
            # fingerprint", letting log lines render without breaking
            # the verifier path.
            return ""


def verify_chain(
    head: SignedPersonaEvent,
    *,
    by_cause_id: Mapping[str, SignedPersonaEvent],
    trusted_pubkeys: Container[str],
    max_depth: int = DEFAULT_MAX_CHAIN_DEPTH,
) -> bool:
    """Verify ``head`` and every persona-adaptation ancestor up to ``max_depth``.

    Returns False on any of:

    * ``head``'s structural signature does not verify under
      ``head.signer_pubkey_hex``;
    * any persona-adaptation ancestor reachable through
      ``triggering_cause_ids`` fails its own signature verify;
    * any envelope carries a malformed (non-hex / wrong-length)
      ``signer_pubkey_hex`` (chain-verification fail-closed: a row whose pubkey is
      unverifiable cannot be folded);
    * any envelope's ``signer_pubkey_hex`` is not in ``trusted_pubkeys``
      (chain-verification source-identity verification: a syntactically valid
      signature from an untrusted key is still rejected, so a
      freshly-generated keypair cannot fold arbitrary adaptive
      deltas);
    * the recursion exceeds ``max_depth`` before reaching leaves
      (the state contract default ``8``).

    ``trusted_pubkeys`` accepts any container that supports ``in``
    (typically a ``set[str]`` / ``frozenset[str]`` derived from
    :func:`aeros.audit.trusted_keys.load_trusted_pubkeys`). Pubkey hex
    matching is byte-exact: callers must pass lowercase 64-char hex
    strings (the format produced by ``Keypair.public_key_hex`` and
    :class:`AuditSigner.public_key_hex`).

    A reference cycle (chain loops back to an already-visited cause_id)
    is treated as a leaf to avoid infinite recursion. The calling
    resolver fold treats any False return as chain-verification rejection.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    return _verify_recursive(
        head,
        by_cause_id=by_cause_id,
        trusted_pubkeys=trusted_pubkeys,
        max_depth=max_depth,
        depth=0,
        visited=set(),
    )


def _verify_recursive(
    node: SignedPersonaEvent,
    *,
    by_cause_id: Mapping[str, SignedPersonaEvent],
    trusted_pubkeys: Container[str],
    max_depth: int,
    depth: int,
    visited: set[str],
) -> bool:
    if depth > max_depth:
        return False
    if node.cause_id in visited:
        # Cycle: treat as terminal so the verifier returns the result
        # already accumulated for this branch rather than recursing
        # forever. The signature on the head was checked the first
        # time the cause_id was visited.
        return True
    visited.add(node.cause_id)

    # Pubkey must decode cleanly. ``verify_event`` is fail-closed on
    # malformed hex but we surface that explicitly so a missing-pubkey
    # row is rejected even if its signature happens to be empty bytes
    # (which Ed25519 verify might handle differently across libs).
    try:
        pubkey_bytes = bytes.fromhex(node.signer_pubkey_hex)
    except ValueError:
        return False
    if len(pubkey_bytes) != 32:
        return False

    # chain-verification source-identity verification: even a structurally valid
    # signature is rejected when the signer is not in the trusted set
    # for this host. Without this check, an actor with episodic.db
    # write access could fabricate a fresh keypair, sign whatever
    # adaptive delta they want, embed the pubkey, and the resolver
    # would fold it as legitimate.
    if node.signer_pubkey_hex not in trusted_pubkeys:
        return False

    if not verify_event(
        node.event,
        signature_hex=node.signature_hex,
        signer_pubkey_hex=node.signer_pubkey_hex,
    ):
        return False

    for cid in node.event.triggering_cause_ids:
        parent = by_cause_id.get(cid)
        if parent is None:
            # Non-persona ancestor (episodic message, evolution row).
            # Out of scope for the chain — terminal.
            continue
        if not _verify_recursive(
            parent,
            by_cause_id=by_cause_id,
            trusted_pubkeys=trusted_pubkeys,
            max_depth=max_depth,
            depth=depth + 1,
            visited=visited,
        ):
            return False
    return True


def make_chain_verifier(
    signed_events: list[SignedPersonaEvent],
    *,
    trusted_pubkeys: Container[str],
    max_depth: int = DEFAULT_MAX_CHAIN_DEPTH,
) -> Callable[[PersonaAdaptiveEvent], bool]:
    """Build a resolver-shaped verifier from a list of signed envelopes.

    The resolver hook (``CauseChainVerifier``) receives only the
    in-memory event; signature artefacts live alongside in the row's
    payload. This adapter indexes the envelopes by Python identity so
    the resolver fold can pass plain :class:`PersonaAdaptiveEvent`
    instances and the verifier still finds the matching signature.

    ``trusted_pubkeys`` propagates into every recursive verify so each
    row in the chain is gated on signer trust, not just the head
    (chain-verification source-identity verification).

    Returns a callable ``verifier(event: PersonaAdaptiveEvent) -> bool``
    suitable for ``effective_persona(..., verifier=...)``. Events not in ``signed_events`` return False
    (chain-verification fail-closed: an unsigned row never folds).
    """
    by_cause_id = {se.cause_id: se for se in signed_events}
    by_identity: dict[int, SignedPersonaEvent] = {
        id(se.event): se for se in signed_events
    }

    def _verifier(event: PersonaAdaptiveEvent) -> bool:
        envelope = by_identity.get(id(event))
        if envelope is None:
            return False
        return verify_chain(
            envelope,
            by_cause_id=by_cause_id,
            trusted_pubkeys=trusted_pubkeys,
            max_depth=max_depth,
        )

    return _verifier
