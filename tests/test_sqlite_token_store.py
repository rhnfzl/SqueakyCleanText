"""Encrypted storage preserves maps without exposing reversible values."""

import importlib.util
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.fernet import Fernet

from sct.utils.anonymization_map import AnonymizationMap


class Keys:
    def __init__(self):
        self.keys = {"first": Fernet.generate_key()}
        self.current = "first"

    def current_key_id(self):
        return self.current

    def fernet_key(self, key_id):
        return self.keys[key_id]


def test_provider_exists():
    assert importlib.util.find_spec("sct.token_stores.sqlite") is not None, "encrypted store is missing"


@pytest.fixture
def api():
    from sct.token_stores import EncryptedSQLiteTokenStore, TokenStoreError

    return EncryptedSQLiteTokenStore, TokenStoreError


@pytest.fixture
def setup_store(tmp_path, api):
    cls, error = api
    keys = Keys()
    now = [100.0]
    events = []
    path = tmp_path / "private" / "tokens.db"
    store = cls(path, scope="tenant", key_provider=keys, ttl_seconds=60, clock=lambda: now[0], audit=events.append)
    mapping = AnonymizationMap(session_id="private-session", reserved={"<PERSON_0>"})
    marker = mapping.next_placeholder("PERSON")
    mapping.add(marker, "Alice Sensitive", "PERSON", 0, 15)
    return store, mapping, marker, keys, now, events, path, error


def test_s1_acceptance(setup_store, api):
    store, mapping, marker, keys, now, events, path, _ = setup_store
    ref = store.save(mapping)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
from sct.token_stores import EncryptedSQLiteTokenStore
data = json.load(sys.stdin)
class Keys:
    def current_key_id(self): return 'first'
    def fernet_key(self, key_id): return data['key'].encode()
store = EncryptedSQLiteTokenStore(Path(data['path']), scope='tenant', key_provider=Keys(),
                                 ttl_seconds=60, clock=lambda: 100)
mapping = store.load(data['reference'])
assert mapping.deanonymize('<PERSON_1>') == 'Alice Sensitive'
assert mapping.next_placeholder('PERSON') == '<PERSON_2>'
""",
        ],
        input=json.dumps({"path": str(path), "key": keys.keys["first"].decode(), "reference": ref}),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    restarted = api[0](path, scope="tenant", key_provider=keys, ttl_seconds=60, clock=lambda: now[0])
    loaded = restarted.load(ref)
    assert loaded.deanonymize(marker) == "Alice Sensitive"
    assert loaded.next_placeholder("PERSON") == "<PERSON_2>"
    assert loaded.reserved == {"<PERSON_0>"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    for secret in (b"Alice Sensitive", b"private-session", keys.keys["first"]):
        assert secret not in path.read_bytes()
        assert secret.decode() not in repr((store, ref, events))
    assert restarted.load("missing") is None
    other = api[0](path, scope="other", key_provider=keys, ttl_seconds=60, clock=lambda: now[0])
    assert other.load(ref) is None
    other.delete(ref)
    assert restarted.load(ref) is not None
    now[0] = 160
    assert restarted.load(ref) is None


@pytest.mark.parametrize("mutation", ["ciphertext", "key", "scope", "reference", "expiry", "map"])
def test_tampering_fails_without_sensitive_errors(setup_store, mutation):
    store, mapping, _, keys, _, _, path, error = setup_store
    ref = store.save(mapping)
    with sqlite3.connect(path) as conn:
        (ciphertext,) = conn.execute("SELECT ciphertext FROM tokens WHERE reference = ?", (ref,)).fetchone()
        if mutation == "key":
            keys.keys["first"] = Fernet.generate_key()
        elif mutation == "ciphertext":
            conn.execute("UPDATE tokens SET ciphertext = ?", (b"invalid",))
        else:
            fernet = Fernet(keys.keys["first"])
            payload = json.loads(fernet.decrypt(ciphertext))
            payload[{"expiry": "expires_at"}.get(mutation, mutation)] = "invalid"
            conn.execute("UPDATE tokens SET ciphertext = ?", (fernet.encrypt(json.dumps(payload).encode()),))
    with pytest.raises(error) as exc:
        store.load(ref)
    assert "Alice Sensitive" not in str(exc.value)
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("ttl", [0, -1, True, 1.5])
def test_invalid_ttl_fails(tmp_path, api, ttl):
    with pytest.raises(ValueError):
        api[0](tmp_path / "tokens.db", scope="tenant", key_provider=Keys(), ttl_seconds=ttl, clock=lambda: 0)


def test_concurrent_saves_rotation_expiry_and_delete(setup_store):
    store, mapping, _, keys, now, _, _, _ = setup_store
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(lambda _: store.save(mapping), range(12)))
    assert len(set(refs)) == 12
    keys.keys["second"] = Fernet.generate_key()
    keys.current = "second"
    assert store.rotate(refs[0]) is True
    del keys.keys["first"]
    assert store.load(refs[0]).entries[0].original == "Alice Sensitive"
    store.delete(refs[0])
    assert store.load(refs[0]) is None
    assert store.rotate(refs[0]) is False
    now[0] = 160
    assert store.purge_expired() == 11


@pytest.mark.parametrize("counter", [{"PERSON": 0}, {"PERSON": True}, {"PERSON": -1}, []])
def test_authenticated_invalid_counters_fail(setup_store, counter):
    store, mapping, _, keys, _, _, path, error = setup_store
    ref = store.save(mapping)
    with sqlite3.connect(path) as conn:
        (ciphertext,) = conn.execute("SELECT ciphertext FROM tokens").fetchone()
        fernet = Fernet(keys.keys["first"])
        payload = json.loads(fernet.decrypt(ciphertext))
        payload["counters"] = counter
        conn.execute("UPDATE tokens SET ciphertext = ?", (fernet.encrypt(json.dumps(payload).encode()),))
    with pytest.raises(error):
        store.load(ref)


def test_unallocated_counters_survive_restart(setup_store):
    store, mapping, _, _, _, _, _, _ = setup_store
    mapping.next_placeholder("PERSON")
    ref = store.save(mapping)
    assert store.load(ref).next_placeholder("PERSON") == "<PERSON_3>"


def test_clock_error_does_not_expose_values(setup_store):
    store, mapping, _, _, _, _, _, error = setup_store

    def broken_clock():
        raise RuntimeError("Alice Sensitive")

    store._clock = broken_clock
    with pytest.raises(error) as exc:
        store.save(mapping)
    assert "Alice Sensitive" not in str(exc.value)


def test_unprotected_files_and_symlinks_fail(tmp_path, api):
    cls, error = api
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    path = parent / "tokens.db"
    path.touch(mode=0o644)
    kwargs = dict(scope="tenant", key_provider=Keys(), ttl_seconds=60, clock=lambda: 0)
    with pytest.raises(error):
        cls(path, **kwargs)
    link = parent / "link.db"
    link.symlink_to(path)
    with pytest.raises(error):
        cls(link, **kwargs)
    parent.chmod(0o755)
    with pytest.raises(error):
        cls(parent / "new.db", **kwargs)
