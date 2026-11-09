"""Invalid stored maps and failed application callbacks cannot cause silent data loss."""
from contextlib import closing
import json
import sqlite3

import pytest

from sct.token_stores import EncryptedSQLiteTokenStore, TokenStoreError
from sct.utils.anonymization_map import AnonymizationMap


PRIVATE = 'synthetic-private-original'


@pytest.fixture
def store_state(tmp_path):
    fernet = pytest.importorskip('cryptography.fernet').Fernet

    class Keys:
        current = 'old'
        values = {'old': fernet.generate_key(), 'new': fernet.generate_key()}

        def current_key_id(self):
            return self.current

        def fernet_key(self, key_id):
            return self.values[key_id]

    keys, now = Keys(), [100]
    path = tmp_path / 'private' / 'tokens.db'
    options = dict(scope='tenant', key_provider=keys, ttl_seconds=60, clock=lambda: now[0])
    store = EncryptedSQLiteTokenStore(path, **options)
    mapping = AnonymizationMap(session_id='synthetic-session')
    marker = mapping.next_placeholder('PERSON')
    mapping.add(marker, PRIVATE, 'PERSON', 0, len(PRIVATE))
    return store, mapping, marker, keys, now, path, options, fernet


@pytest.mark.parametrize('damage', ['version', 'session', 'reservations', 'entries', 'entry_type', 'entry_range'])
def test_authenticated_malformed_maps_cannot_restore_or_rotate_values(store_state, damage):
    store, mapping, marker, keys, _, path, _, fernet = store_state
    reference = store.save(mapping)
    assert store.load(reference).deanonymize(marker) == PRIVATE
    with closing(sqlite3.connect(path)) as database, database:
        ciphertext = database.execute('SELECT ciphertext FROM tokens').fetchone()[0]
        cipher = fernet(keys.values['old'])
        payload = json.loads(cipher.decrypt(ciphertext))
        if damage == 'version':
            payload['version'] = True
        elif damage == 'session':
            payload['map']['session_id'] = 42
        elif damage == 'reservations':
            payload['map']['reserved'] = [PRIVATE, 42]
        elif damage == 'entries':
            payload['map']['entries'] = {'unexpected': PRIVATE}
        elif damage == 'entry_type':
            payload['map']['entries'][0]['start'] = True
        else:
            payload['map']['entries'][0]['end'] = -1
        database.execute('UPDATE tokens SET ciphertext = ?', (cipher.encrypt(json.dumps(payload).encode()),))
    before = path.read_bytes()
    for operation in (store.load, store.rotate):
        with pytest.raises(TokenStoreError) as error:
            operation(reference)
        assert PRIVATE not in str(error.value)
        assert error.value.__cause__ is None
        assert path.read_bytes() == before


@pytest.mark.parametrize('invalid_clock', [float('nan'), float('inf'), True, PRIVATE])
def test_invalid_clock_cannot_commit_restore_rotate_or_purge_maps(store_state, invalid_clock):
    store, mapping, marker, _, now, path, _, _ = store_state
    reference = store.save(mapping)
    before = path.read_bytes()
    now[0] = invalid_clock
    for operation, args in ((store.save, (mapping,)), (store.load, (reference,)),
                            (store.rotate, (reference,)), (store.purge_expired, ())):
        with pytest.raises(TokenStoreError) as error:
            operation(*args)
        assert PRIVATE not in str(error.value)
        assert path.read_bytes() == before
    now[0] = 100
    assert store.load(reference).deanonymize(marker) == PRIVATE


@pytest.mark.parametrize('damage', ['blank_identifier', 'invalid_identifier_type', 'invalid_key_bytes'])
def test_invalid_key_metadata_cannot_overwrite_a_committed_map(store_state, damage):
    store, mapping, marker, keys, _, path, _, _ = store_state
    reference = store.save(mapping)
    if damage == 'invalid_key_bytes':
        keys.current = 'new'
        keys.values['new'] = b'invalid-key'
    else:
        keys.current = '' if damage == 'blank_identifier' else None
    before = path.read_bytes()
    for operation, value in ((store.save, mapping), (store.rotate, reference)):
        with pytest.raises(TokenStoreError) as error:
            operation(value)
        assert PRIVATE not in str(error.value)
        assert path.read_bytes() == before
    keys.current = 'old'
    assert store.load(reference).deanonymize(marker) == PRIVATE


@pytest.mark.parametrize('operation', ['save', 'rotate', 'delete', 'purge_expired'])
def test_audit_failure_reports_the_committed_outcome_without_rollback(store_state, operation):
    reader, mapping, marker, keys, now, path, options, _ = store_state
    events = []

    def audit(event):
        events.append(event)
        if event.operation == operation:
            raise RuntimeError(PRIVATE)

    store = EncryptedSQLiteTokenStore(path, audit=audit, **options)
    reference = None if operation == 'save' else store.save(mapping)
    if operation == 'rotate':
        keys.current = 'new'
    elif operation == 'purge_expired':
        now[0] = 160
    call, args = (store.save, (mapping,)) if operation == 'save' else (
        (store.purge_expired, ()) if operation == 'purge_expired' else (getattr(store, operation), (reference,)))
    with pytest.raises(TokenStoreError, match='after commit') as error:
        call(*args)
    assert PRIVATE not in str(error.value) + repr(events)
    if operation in ('save', 'rotate'):
        if operation == 'save':
            reference = events[-1].reference
        else:
            del keys.values['old']
        assert reader.load(reference).deanonymize(marker) == PRIVATE
    else:
        now[0] = 100
        assert reader.load(reference) is None
        assert reader.purge_expired() == 0
