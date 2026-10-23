"""SQLite-backed storage layer for EpisodicMemory — PersistentAgent Layer 1.

Holds the connection, schema migration, and the small surface
(`append` / `recent` / `count` / `prune_to_capacity` / `close`) that
`EpisodicMemory` swaps in for its in-process list when a `db_path` is
provided.

Synchronous by design (the state contract): `EpisodicMemory.append` is
called from sync code paths (`PersistentAgent.record`) inside async
route handlers. Using `sqlite3` keeps the call site sync — with the
`(identity_hash, timestamp DESC)` index each insert is microseconds, so
the event loop is held for negligible time.

Rows are partitioned by `identity_hash` so a single shared DB file can
host multiple robots without cross-contamination; queries always lead
with that key.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_hash TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    intent_id     TEXT,
    summary       TEXT    NOT NULL,
    payload_json  TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodic_lookup
    ON episodic (identity_hash, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_kind
    ON episodic (identity_hash, kind, timestamp DESC);
"""

_INSERT = (
    "INSERT INTO episodic (identity_hash, timestamp, kind, intent_id, "
    "summary, payload_json) VALUES (?, ?, ?, ?, ?, ?)"
)

_RECENT = (
    "SELECT timestamp, kind, intent_id, summary, payload_json FROM episodic "
    "WHERE identity_hash = ? ORDER BY id DESC LIMIT ?"
)

# Layer 3: scan-by-kind for migration_import marker
# lookup. Used by /api/evolution/audit to surface the cross-deployment
# audit chain marker without having to walk the full ring buffer in
# Python. Index `idx_episodic_kind` covers the WHERE.
_BY_KIND = (
    "SELECT timestamp, kind, intent_id, summary, payload_json FROM episodic "
    "WHERE identity_hash = ? AND kind = ? ORDER BY id ASC"
)

_COUNT = "SELECT COUNT(*) FROM episodic WHERE identity_hash = ?"

# Delete the `excess` oldest rows for this identity. ORDER BY id ASC rides
# the autoincrement so it serves as the FIFO order the buffer used to
# implement; LIMIT keeps the inner scan bounded by the deletion size, not
# table size.
_PRUNE = (
    "DELETE FROM episodic WHERE identity_hash = ? AND id IN ("
    "SELECT id FROM episodic WHERE identity_hash = ? ORDER BY id ASC LIMIT ?"
    ")"
)


class EpisodicStore:
    """Thread-safe sync SQLite store for episodic records.

    A single instance owns one connection. Concurrent writers serialise
    on `_lock`; readers go through the same lock so they observe a
    committed snapshot regardless of caller thread.
    """

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        # Auto-create the parent directory so first-run on a fresh host
        # (no ~/.aeros yet) does not crash the lifespan startup.
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        # check_same_thread=False — sync writes happen from whichever
        # thread the FastAPI handler ran on; we provide our own _lock.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        # WAL gives us concurrent readers + a single writer with much
        # better restart durability than the default rollback journal.
        # PRAGMA persists at the file level; safe to set on every open.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._lock = threading.Lock()
        logger.debug("EpisodicStore opened at %s", self._db_path)

    @property
    def path(self) -> str:
        return self._db_path

    def append(
        self,
        identity_hash: str,
        *,
        timestamp: str,
        kind: str,
        intent_id: str | None,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        payload_json = json.dumps(payload, default=str, sort_keys=True)
        with self._lock:
            self._conn.execute(
                _INSERT,
                (identity_hash, timestamp, kind, intent_id, summary, payload_json),
            )
            self._conn.commit()

    def recent(self, identity_hash: str, n: int) -> list[tuple[Any, ...]]:
        """Return up to n most-recent rows for identity, oldest-first.

        Output ordering matches the historical in-memory `recent(n)`
        slice (`self._records[-n:]`) so swapping backends is transparent
        to callers.
        """
        if n <= 0:
            return []
        with self._lock:
            cur = self._conn.execute(_RECENT, (identity_hash, n))
            rows = cur.fetchall()
        rows.reverse()
        return [
            (
                ts,
                kind,
                intent_id,
                summary,
                json.loads(payload_json) if payload_json else {},
            )
            for (ts, kind, intent_id, summary, payload_json) in rows
        ]

    def by_kind(self, identity_hash: str, kind: str) -> list[tuple[Any, ...]]:
        """Return all records matching ``(identity_hash, kind)``, oldest-first.

        Layer 3 needs the migration_import audit marker
        to be queryable cheaply: ``GET /api/evolution/audit`` calls this
        on every request to fold cross-deployment markers into the audit
        chain response. Tuple shape mirrors :py:meth:`recent` so the
        caller maps both with one EpisodicRecord constructor.
        """
        with self._lock:
            cur = self._conn.execute(_BY_KIND, (identity_hash, kind))
            rows = cur.fetchall()
        return [
            (
                ts,
                row_kind,
                intent_id,
                summary,
                json.loads(payload_json) if payload_json else {},
            )
            for (ts, row_kind, intent_id, summary, payload_json) in rows
        ]

    def count(self, identity_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(_COUNT, (identity_hash,))
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def prune_to_capacity(self, identity_hash: str, capacity: int) -> int:
        """Drop the oldest rows so count(identity_hash) == min(count, capacity).

        Returns the number of rows deleted. No-op when already at or
        below capacity.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        with self._lock:
            cur = self._conn.execute(_COUNT, (identity_hash,))
            row = cur.fetchone()
            count = int(row[0]) if row else 0
            if count <= capacity:
                return 0
            excess = count - capacity
            cur = self._conn.execute(_PRUNE, (identity_hash, identity_hash, excess))
            deleted = cur.rowcount
            self._conn.commit()
        return int(deleted)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.debug("EpisodicStore closed at %s", self._db_path)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 — finalizer must not raise
            pass
