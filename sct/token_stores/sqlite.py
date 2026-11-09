"""Keep reversible maps encrypted in an application-owned local database."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import stat
from typing import Protocol

from sct.utils.anonymization_map import AnonymizationMap


class TokenStoreError(RuntimeError):
    """The store could not safely read or write an entry."""


class KeyProvider(Protocol):
    """Supply externally owned keys without storing them in SQLite."""

    def current_key_id(self) -> str: ...

    def fernet_key(self, key_id: str) -> bytes: ...


@dataclass(frozen=True)
class StoreEvent:
    """Value-free notification after a committed storage operation."""

    operation: str
    reference: str | None = None
    count: int = 1


class EncryptedSQLiteTokenStore:
    """Store maps under a trusted scope; deletion is logical, not secure erasure."""

    def __init__(
        self,
        path: Path,
        *,
        scope: str,
        key_provider: KeyProvider,
        ttl_seconds: int,
        clock: Callable[[], float],
        audit: Callable[[StoreEvent], None] | None = None,
    ):
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        if not isinstance(scope, str) or not scope:
            raise ValueError("scope must be a nonempty trusted identifier")
        self._path = Path(path)
        self._scope = scope
        self._keys = key_provider
        self._ttl = ttl_seconds
        self._clock = clock
        self._audit = audit
        try:
            from cryptography.fernet import Fernet

            self._fernet = Fernet
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self._path.parent.is_symlink() or self._path.parent.stat().st_mode & 0o077:
                raise ValueError("unprotected directory")
            fd = os.open(self._path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                    raise ValueError("unprotected database")
            finally:
                os.close(fd)
            with self._connection() as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS tokens (
                    reference TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    expires_at REAL NOT NULL, key_id TEXT NOT NULL,
                    ciphertext BLOB NOT NULL)""")
        except Exception:
            raise TokenStoreError("Cannot initialize encrypted token store") from None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(self._path, timeout=30)
            with conn:
                yield conn
        except Exception:
            raise TokenStoreError("Encrypted token store operation failed") from None
        finally:
            if conn is not None:
                conn.close()

    def _now(self) -> float:
        try:
            now = self._clock()
            if type(now) not in (int, float) or not math.isfinite(now):
                raise ValueError("invalid clock")
            return now
        except Exception:
            raise TokenStoreError("Invalid token store clock") from None

    def _encrypt(self, payload: dict) -> tuple[str, bytes]:
        try:
            key_id = self._keys.current_key_id()
            if not isinstance(key_id, str) or not key_id:
                raise ValueError("invalid key identifier")
            token = self._fernet(self._keys.fernet_key(key_id)).encrypt(json.dumps(payload, allow_nan=False).encode("utf-8"))
            return key_id, token
        except Exception:
            raise TokenStoreError("Cannot encrypt token store entry") from None

    def _decode(self, reference: str, row: tuple) -> dict:
        expiry, key_id, ciphertext = row
        try:
            payload = json.loads(self._fernet(self._keys.fernet_key(key_id)).decrypt(ciphertext))
            if (
                type(payload["version"]) is not int
                or payload["version"] != 1
                or payload["scope"] != self._scope
                or payload["reference"] != reference
                or payload["expires_at"] != expiry
                or not isinstance(expiry, (int, float))
                or not math.isfinite(expiry)
            ):
                raise ValueError("invalid envelope")
            data = payload["map"]
            if not isinstance(data["session_id"], str):
                raise ValueError("invalid map")
            if not isinstance(data["reserved"], list) or not all(isinstance(x, str) for x in data["reserved"]):
                raise ValueError("invalid reservations")
            if not isinstance(data["entries"], list):
                raise ValueError("invalid entries")
            for entry in data["entries"]:
                if (
                    not all(isinstance(entry[k], str) for k in ("placeholder", "original", "entity_type"))
                    or type(entry["start"]) is not int
                    or type(entry["end"]) is not int
                    or entry["start"] < 0
                    or entry["end"] < entry["start"]
                ):
                    raise ValueError("invalid entry")
            counters = payload["counters"]
            if not isinstance(counters, dict) or not all(
                isinstance(k, str) and type(v) is int and v >= 0 for k, v in counters.items()
            ):
                raise ValueError("invalid counters")
            restored = AnonymizationMap.from_dict(data)
            if any(counters.get(k, 0) < v for k, v in restored._counter.items()):
                raise ValueError("counter precedes restored entries")
            return payload
        except Exception:
            raise TokenStoreError("Cannot authenticate token store entry") from None

    def _event(self, operation: str, reference: str | None = None, count: int = 1) -> None:
        if self._audit is not None:
            try:
                self._audit(StoreEvent(operation, reference, count))
            except Exception:
                raise TokenStoreError("Token store audit callback failed after commit") from None

    def save(self, anon_map: AnonymizationMap) -> str:
        reference = secrets.token_urlsafe(32)
        expiry = self._now() + self._ttl
        payload = dict(
            version=1,
            scope=self._scope,
            reference=reference,
            expires_at=expiry,
            map=anon_map.to_dict(),
            counters=anon_map._counter,
        )
        key_id, ciphertext = self._encrypt(payload)
        with self._connection() as conn:
            conn.execute("INSERT INTO tokens VALUES (?, ?, ?, ?, ?)", (reference, self._scope, expiry, key_id, ciphertext))
        self._event("save", reference)
        return reference

    def _row(self, conn: sqlite3.Connection, reference: str) -> tuple | None:
        return conn.execute(
            "SELECT expires_at, key_id, ciphertext FROM tokens WHERE reference = ? AND scope = ?", (reference, self._scope)
        ).fetchone()

    def load(self, reference: str) -> AnonymizationMap | None:
        with self._connection() as conn:
            row = self._row(conn, reference)
            if row is None:
                return None
            payload = self._decode(reference, row)
            if payload["expires_at"] <= self._now():
                return None
            mapping = AnonymizationMap.from_dict(payload["map"])
            mapping._counter = payload["counters"]
        self._event("load", reference)
        return mapping

    def delete(self, reference: str) -> None:
        with self._connection() as conn:
            count = conn.execute("DELETE FROM tokens WHERE reference = ? AND scope = ?", (reference, self._scope)).rowcount
        self._event("delete", count=count)

    def rotate(self, ref: str) -> bool:
        """Re-encrypt one live entry without extending its lifetime."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, ref)
            if row is None:
                return False
            payload = self._decode(ref, row)
            if payload["expires_at"] <= self._now():
                return False
            key_id, ciphertext = self._encrypt(payload)
            conn.execute(
                "UPDATE tokens SET key_id = ?, ciphertext = ? WHERE reference = ? AND scope = ?",
                (key_id, ciphertext, ref, self._scope),
            )
        self._event("rotate", ref)
        return True

    def purge_expired(self) -> int:
        """Logically delete expired entries in the current scope."""
        with self._connection() as conn:
            count = conn.execute("DELETE FROM tokens WHERE scope = ? AND expires_at <= ?", (self._scope, self._now())).rowcount
        self._event("purge_expired", count=count)
        return count
