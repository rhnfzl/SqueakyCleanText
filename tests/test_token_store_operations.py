"""Bounded operational checks use synthetic maps and application-owned keys."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
import multiprocessing
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier

import pytest

if importlib.util.find_spec("cryptography") is None and os.environ.get("SCT_OPTIONAL_PROFILE") != "token-store":
    pytest.skip("install the token-store extra", allow_module_level=True)

from cryptography.fernet import Fernet

from sct.token_stores import EncryptedSQLiteTokenStore, TokenStoreError
from sct.utils.anonymization_map import AnonymizationMap


class Keys:
    def __init__(self):
        self.keys = {"old": Fernet.generate_key(), "new": Fernet.generate_key()}
        self.current = "old"

    def current_key_id(self):
        return self.current

    def fernet_key(self, key_id):
        return self.keys[key_id]


def fixed_clock():
    return 100


def make_map(label):
    mapping = AnonymizationMap(session_id=f"synthetic-{label}", reserved={"<PERSON_0>"})
    for index in range(8):
        value = f"synthetic-person-{label}-{index}"
        mapping.add(mapping.next_placeholder("PERSON"), value, "PERSON", index * 40, index * 40 + len(value))
    mapping.next_placeholder("PERSON")
    return mapping


def check_map(store, ref, label):
    restored = store.load(ref)
    assert restored is not None
    assert restored.to_dict() == make_map(label).to_dict()
    assert restored.next_placeholder("PERSON") == "<PERSON_10>"


def exercise_worker(store, barrier, label, shared_ref):
    barrier.wait(timeout=30)
    refs = []
    for index in range(4):
        name = f"{label}-{index}"
        ref = store.save(make_map(name))
        check_map(store, ref, name)
        check_map(store, shared_ref, "shared")
        refs.append((ref, name))
    return refs


def process_worker(store, barrier, label, shared_ref, queue):
    # Observe real connections while exercising a store warmed in the parent.
    connect = sqlite3.connect
    owners = []

    def owned_connect(*args, **kwargs):
        owners.append(os.getpid())
        return connect(*args, **kwargs)

    sqlite3.connect = owned_connect
    try:
        refs = exercise_worker(store, barrier, label, shared_ref)
        assert len(owners) == 12 and set(owners) == {os.getpid()}
        queue.put((True, refs))
    except BaseException as exc:
        queue.put((False, type(exc).__name__))
        raise
    finally:
        sqlite3.connect = connect


@pytest.fixture
def setup_store(tmp_path):
    keys = Keys()
    path = tmp_path / "private" / "tokens.db"
    store = EncryptedSQLiteTokenStore(path, scope="trusted-app-tenant", key_provider=keys,
                                     ttl_seconds=60, clock=fixed_clock)
    return store, keys, path


def test_s2_acceptance(setup_store):
    store, _, _ = setup_store
    shared_ref = store.save(make_map("shared"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        barrier = Barrier(4)
        futures = [pool.submit(exercise_worker, store, barrier, f"thread-{i}", shared_ref) for i in range(4)]
        results = [future.result(timeout=60) for future in futures]
    refs = [ref for result in results for ref, _ in result]
    assert len(set(refs + [shared_ref])) == 17
    for result in results:
        for ref, label in result:
            assert len(ref) >= 32 and label not in ref
            check_map(store, ref, label)


def exercise_processes(store, method):
    shared_ref = store.save(make_map("shared"))
    check_map(store, shared_ref, "shared")
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    barrier = context.Barrier(4)
    children = [context.Process(target=process_worker, args=(store, barrier, str(i), shared_ref, queue)) for i in range(4)]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=60) for _ in children]
        for child in children:
            child.join(timeout=60)
            assert child.exitcode == 0
        assert all(ok for ok, _ in results)
        refs = [ref for _, result in results for ref, _ in result]
        assert len(set(refs + [shared_ref])) == 17
        for _, result in results:
            for ref, label in result:
                check_map(store, ref, label)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=10)
        queue.close()
        queue.join_thread()


def run_fork_probe(script, data, *, timeout):
    child = subprocess.Popen(  # noqa: S603 - scripts are trusted test fixtures, never application input
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(json.dumps(data), timeout=timeout)
    except subprocess.TimeoutExpired:
        # The fresh session owns this group, including any fork workers.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGTERM:
                try:
                    child.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        child.communicate(timeout=5)
        raise
    return subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_processes_use_fresh_connections_and_complete_maps(setup_store, method):
    store, _, path = setup_store
    if method == "fork":
        # Fork only from a fresh parent without the full suite's native threads.
        child = run_fork_probe(
            """
from pathlib import Path
import json, sys
data = json.load(sys.stdin)
sys.path.insert(0, data['tests'])
from test_token_store_operations import EncryptedSQLiteTokenStore, Keys, fixed_clock, exercise_processes
store = EncryptedSQLiteTokenStore(Path(data['database']), scope='trusted-app-tenant',
                                 key_provider=Keys(), ttl_seconds=60, clock=fixed_clock)
exercise_processes(store, 'fork')
print('workers=4 maps=17 status=passed')
""",
            {"tests": str(Path(__file__).resolve().parent), "database": str(path)},
            timeout=60,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert child.stderr == ""
        assert child.stdout.strip() == "workers=4 maps=17 status=passed"
    else:
        exercise_processes(store, method)


def test_rotation_keeps_reference_expiry_and_retained_old_keys(setup_store):
    store, keys, path = setup_store
    old_ref = store.save(make_map("old"))
    keys.current = "new"
    new_ref = store.save(make_map("new"))
    check_map(store, old_ref, "old")
    check_map(store, new_ref, "new")
    with closing(sqlite3.connect(path)) as conn:
        before = conn.execute("SELECT key_id, expires_at, ciphertext FROM tokens WHERE reference = ?", (old_ref,)).fetchone()
        assert before[:2] == ("old", 160)
        assert conn.execute("SELECT key_id FROM tokens WHERE reference = ?", (new_ref,)).fetchone() == ("new",)
    store._clock = lambda: 120
    barrier = Barrier(4)

    def rotate_or_read(index):
        barrier.wait(timeout=30)
        for _ in range(4):
            if index % 2:
                assert store.rotate(old_ref)
            else:
                check_map(store, old_ref, "old")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(rotate_or_read, range(4)))
    with closing(sqlite3.connect(path)) as conn:
        after = conn.execute("SELECT key_id, expires_at, ciphertext FROM tokens WHERE reference = ?", (old_ref,)).fetchone()
    assert after[:2] == ("new", 160)
    assert after[2] != before[2]
    check_map(store, old_ref, "old")
    store._clock = lambda: 160
    assert store.load(old_ref) is None
    assert store.rotate(old_ref) is False


def test_failed_rotation_preserves_committed_entry(setup_store):
    store, keys, _ = setup_store
    ref = store.save(make_map("retained"))
    keys.current = "unavailable"
    with pytest.raises(TokenStoreError) as error:
        store.rotate(ref)
    assert "synthetic" not in str(error.value)
    check_map(store, ref, "retained")


def test_expiry_purge_is_atomic_and_scope_bound(setup_store):
    store, keys, path = setup_store
    refs = [store.save(make_map(str(i))) for i in range(3)]
    other = EncryptedSQLiteTokenStore(path, scope="other-trusted-tenant", key_provider=keys,
                                     ttl_seconds=60, clock=fixed_clock)
    other_ref = other.save(make_map("other"))
    store._clock = lambda: 159
    live_ref = store.save(make_map("live"))
    store._clock = lambda: 160
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"""CREATE TRIGGER abort_purge BEFORE DELETE ON tokens
                        WHEN OLD.reference = '{refs[1]}' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END""")
    with pytest.raises(TokenStoreError):
        store.purge_expired()
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT count(*) FROM tokens").fetchone() == (5,)
        conn.execute("DROP TRIGGER abort_purge")
    barrier = Barrier(4)

    def purge_or_read(index):
        barrier.wait(timeout=30)
        if index < 2:
            return store.purge_expired()
        for _ in range(4):
            check_map(store, live_ref, "live")
            for ref in refs:
                assert store.load(ref) is None
        return 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(purge_or_read, range(4))) == 3
    assert store.purge_expired() == 0
    check_map(other, other_ref, "other")
    assert store.load(other_ref) is None
    assert store.rotate(other_ref) is False
    store.delete(other_ref)
    check_map(other, other_ref, "other")
    assert store.load("missing") is None
    assert store.rotate("missing") is False


def test_rotation_rejects_tamper_without_values(setup_store):
    store, keys, path = setup_store
    ref = store.save(make_map("secret"))
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE tokens SET ciphertext = ? WHERE reference = ?", (b"synthetic-tamper", ref))
    for operation in (store.load, store.rotate):
        with pytest.raises(TokenStoreError) as error:
            operation(ref)
        assert "synthetic" not in str(error.value)
        assert keys.keys["old"].decode() not in str(error.value)
        assert error.value.__cause__ is None


def test_every_operation_closes_its_connection(tmp_path, monkeypatch):
    connect = sqlite3.connect
    connections = []

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    store = EncryptedSQLiteTokenStore(tmp_path / "private" / "tokens.db", scope="trusted-app-tenant",
                                     key_provider=Keys(), ttl_seconds=60, clock=fixed_clock)
    ref = store.save(make_map("closure"))
    check_map(store, ref, "closure")
    assert store.rotate(ref)
    store.delete(ref)
    assert store.purge_expired() == 0
    assert len(connections) == 6
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_fork_probe_timeout_cleans_its_owned_process_group(tmp_path):
    marker = tmp_path / "grandchild-terminated"
    with pytest.raises(subprocess.TimeoutExpired):
        run_fork_probe(
            """
import json, signal, subprocess, sys, time
from pathlib import Path
data = json.load(sys.stdin)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
code = '''
import signal, sys, time
from pathlib import Path
def stop(*_):
    Path(sys.argv[1]).write_text('terminated')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
time.sleep(60)
'''
grandchild = subprocess.Popen([sys.executable, '-c', code, data['marker']], stdout=subprocess.PIPE, text=True)
assert grandchild.stdout.readline().strip() == 'ready'
time.sleep(60)
""",
            {"marker": str(marker)},
            timeout=1,
        )
    assert marker.read_text() == "terminated"
