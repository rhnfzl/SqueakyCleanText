# Encrypted token maps

Applications can restore reversible replacements after a restart through an opaque reference.
The map still contains personal data, so the application must protect its keys and control access.
`EncryptedSQLiteTokenStore` is an opt-in local reference integration using authenticated Fernet encryption and SQLite transactions.
The default store behavior and `TokenStore.save/load/delete` interface do not change.

## Install and connect

```sh
pip install 'SqueakyCleanText[token-store]'
```

The application supplies a key provider, a trusted scope, a positive integer lifetime, and a clock:

```python
from pathlib import Path
from time import time
from sct.token_stores import EncryptedSQLiteTokenStore

# app_keys implements current_key_id() and fernet_key(key_id).
# The application obtains this provider from its own protected key service.
store = EncryptedSQLiteTokenStore(
    Path('/srv/my-app/private-token-maps/tokens.db'),
    scope=authorized_tenant_id,
    key_provider=app_keys,
    ttl_seconds=3600,
    clock=time,
    audit=record_store_event,
)
```

`app_keys`, `authorized_tenant_id`, and `record_store_event` are application-owned dependencies, not SCT services.
The provider returns an opaque key ID and Fernet key bytes. Keys are never written into SQLite.
Do not put keys, raw maps, or restored values in logs, command arguments, or source files.
Pass `token_store=store` to `TextCleaner` with `replacement_mode='reversible'`.
Read the reference from `result.metadata['anon_map_reference']`; see [Privacy](PRIVACY.md).
SCT does not automatically write raw map files.

## Scope and authorization

The trusted application assigns `scope` after authenticating the caller and authorizing its tenant or session.
Never accept scope directly from an arbitrary RPC client, request body, or model-generated argument.
A reference is an identifier, not proof of authorization.
The application authorizes every save, restore, rotation, and deletion before calling the store.

The encrypted envelope binds the scope, reference, expiry, map, and allocation counters.
A wrong scope or missing reference returns `None` from `load`, or `False` from `rotate`.
An authenticated entry that has expired behaves the same way.
Malformed entries, ciphertext tampering, unavailable keys, and storage failures raise value-free `TokenStoreError` exceptions.
An expired row with an unavailable key or corrupt ciphertext can still raise an authentication error during `load` or `rotate`.
Expiry checks do not bypass authentication.

Database readers can see opaque references, scope identifiers, expiry times, key IDs, ciphertext sizes, and access patterns.
Use non-sensitive scope and key identifiers. Encryption does not hide this metadata.
Processes with database write access can delete rows or alter unauthenticated lookup metadata.
The store cannot prevent denial of service by a process that controls the files.

## Rotate keys without extending retention

1. Add the new key to the application's provider.
2. Make `current_key_id()` return the new key ID.
3. Keep old keys available through `fernet_key(old_key_id)`.
4. Call `store.rotate(reference)` for each live map that needs migration.
5. Verify migration and backup requirements before retiring any old key.

New saves use the current key. Existing maps remain readable using their recorded key ID while that key remains available.
`rotate(ref) -> bool` re-encrypts one live map in a write transaction.
It preserves the reference and original expiry. It returns `False` for missing, wrong-scope, or expired entries.
Concurrent readers see a complete committed map or an error, never a partially updated map.
A failed encryption or database operation leaves the previous committed entry intact.
The provider must coordinate key changes across application workers and remain safe for their concurrency model.

Removing an old key makes all entries and backups encrypted with it unreadable until that key is restored.
Rotation does not update old backups or other references automatically.
SCT does not retire or destroy application keys.
The application owns key recovery, version inventory, backup encryption, and disaster recovery tests.

## Expiry, deletion, and audit

`load` rejects entries at or after their expiry. It does not delete them.
`purge_expired() -> int` transactionally deletes expired rows only in the store's scope and returns the deleted count.
The application schedules purge and selects a trusted clock and retention policy.
`delete(reference)` logically deletes the current-scope row and returns `None`.

Logical deletion, expiry, purge, and rotation do not securely erase SQLite pages, journals, WAL files, backups, snapshots, or storage media.
Storage encryption and backup retention remain application responsibilities.
Neither this integration nor its tests provide compliance certification.

An optional audit callback receives a frozen `StoreEvent(operation, reference=None, count=1)` after commit.
Successful saves, loads, and rotations include a reference. Deletes and purges report counts without a reference.
Missing or expired loads and unsuccessful rotations do not emit events.
The callback does not provide a complete authorization or failed-access audit trail.
The application records access decisions and protects its audit sink separately.
If the callback fails, SCT raises `TokenStoreError` after the database operation has committed.
A failed save response can therefore leave a stored entry whose reference the caller did not receive.
Design application retry, reconciliation, and retention around that possibility.

## Files, processes, and limits

Use a dedicated local directory owned by the application account.
The immediate parent directory must exclude group and other access (normally mode `0700`).
The database must be a regular file excluding group and other access (normally mode `0600`).
The store rejects a symlink database and a symlink immediate parent.
The application must also protect ancestor directories and avoid untrusted filesystem writers.
These checks do not provide a sandbox or prevent filesystem replacement races by privileged processes.

Each operation opens and closes its own SQLite connection; no connection is retained across a fork.
Fork requires a quiescent parent without active threads or inherited provider resources.
Use `spawn` or `forkserver` when the application already has threads. SCT does not provide general fork safety.
SQLite's default rollback journal and transactions serialize writers, with a 30-second connection timeout.
No WAL mode is enabled by SCT. Protect any journals, WAL files created externally, and backup copies with the same access policy.
This is local storage for mutually trusted application processes, not a network filesystem or distributed database integration.
Do not share a mutable map across writers while it is being saved.
Construct worker-local providers and stores in application deployments, especially when providers contain locks or external clients.

The implementation targets Python 3.12+ on POSIX systems with `os.O_NOFOLLOW` and Unix permission semantics.
Windows, network filesystems, hostile shared directories, and multi-host coordination are outside the qualified scope.
Operational tests use four workers, small synthetic maps, concurrent reads/writes, rotation, rollback, and scope-bound expiry cleanup.
They cover the multiprocessing start methods available on the tested runtime, including Linux `fork`, `spawn`, and `forkserver`.
The fork case runs in a fresh Python subprocess to avoid native threads left by other tests.
These bounded checks do not establish sustained-load capacity or resistance to every local attacker.
Review deployment permissions, authorization, key management, and recovery independently before production use.
An independent security-focused review must pass before this reference integration is labelled ready for supported production use.
