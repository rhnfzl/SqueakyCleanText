"""Stream corpus text into private SQLite state, then select verified clusters.

References own sensitive output and shingle state. Close the returned report to
remove that state. Abandoned analysis iterators require caller cleanup of their
output_ref parent directory. Inputs are never overwritten.
"""
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import secrets
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from sct.score import (ONNXQualityScorer, QualityScorer, ScoreConfig, minhash_metadata,
                       minhash_signature, minhash_tokens, score_record)
from .types import CorpusRecord, RecordAnalysis, SelectionConfig, SelectionDecision


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _workspace() -> Path:
    path = Path(tempfile.mkdtemp(prefix='sct-corpus-'))
    path.chmod(0o700)
    return path



def _owned_reference(path: Path, token: str) -> bool:
    """Only generated private references grant directory cleanup ownership."""
    marker = path.parent / '.owner'
    try:
        return bool(token and path.name == 'outputs.sqlite' and not path.is_symlink()
                    and not path.parent.is_symlink() and not marker.is_symlink()
                    and path.parent.name.startswith('sct-corpus-')
                    and path.parent.parent.resolve() == Path(tempfile.gettempdir()).resolve()
                    and secrets.compare_digest(marker.read_text(), token))
    except OSError:
        return False


def _remove_reference(path: Path, token: str) -> None:
    if _owned_reference(path, token):
        shutil.rmtree(path.parent, ignore_errors=True)


def analyse_records(records: Iterable[CorpusRecord], *, cleaner: Any,
                    score_config: ScoreConfig, selection: SelectionConfig,
                    scorer: QualityScorer | None = None) -> Iterator[RecordAnalysis]:
    """Score originals and retain only private disk references in analysis rows.

    No model is downloaded. quality_model must identify a local exported model.
    The caller owns yielded references until select_records transfers ownership.
    """
    if selection.quality_min is not None or score_config.quality:
        if scorer is None:
            if not score_config.quality_model or not Path(score_config.quality_model).is_dir():
                raise ValueError('quality selection needs a scorer or local quality_model')
            scorer = ONNXQualityScorer(score_config.quality_model, revision=score_config.quality_revision)
    root = _workspace()
    token = secrets.token_hex(32)
    marker = root / '.owner'
    marker.touch(mode=0o600)
    marker.write_text(token)
    path = root / 'outputs.sqlite'
    connection = sqlite3.connect(path)
    path.chmod(0o600)
    connection.execute('CREATE TABLE outputs (id TEXT PRIMARY KEY, text TEXT, metadata TEXT, shingles TEXT)')
    identity = {**minhash_metadata(score_config), 'schema_version': 1, 'basis': selection.basis}
    try:
        for record in records:
            result = cleaner.process(record.text)
            quality = None
            if scorer is not None:
                values = scorer([record.text])
                if len(values) != 1 or not math.isfinite(values[0]) or not 0 <= values[0] <= 1:
                    raise ValueError('scorer must return one finite score in [0, 1] per record')
                quality = values[0]
            scores = score_record(result, quality=quality)
            basis = record.text if selection.basis == 'pre_pii' else result.lm_text
            signature = minhash_signature(basis, score_config)
            tokens = minhash_tokens(basis)
            size = score_config.minhash_shingle
            shingles = sorted({_digest('\x1f'.join(tokens[i:i + size]))
                               for i in range(max(len(tokens) - size + 1, 1))}) if signature else []
            try:
                connection.execute('INSERT INTO outputs VALUES (?, ?, ?, ?)',
                                   (record.id, result.lm_text, json.dumps(dict(record.metadata)), json.dumps(shingles)))
            except sqlite3.IntegrityError:
                raise ValueError('duplicate record id') from None
            connection.commit()
            yield RecordAnalysis(record.id, scores, identity, None if signature is None else tuple(signature),
                                 path, _digest(record.text), _digest(basis), token)
    except BaseException:
        connection.close()
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        connection.close()


class SelectionReport:
    """Disk-backed decisions and selected output. Context exit removes private state."""

    def __init__(self, root: Path, sources: dict[Path, str]):
        self.root = root
        self.sources = sources
        self._connections: set[sqlite3.Connection] = set()
        self._closed = False

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise ValueError("selection report is closed")
        connection = sqlite3.connect(self.root / "decisions.sqlite")
        self._connections.add(connection)
        try:
            yield connection
        finally:
            self._connections.discard(connection)
            connection.close()

    def iter_decisions(self) -> Iterator[SelectionDecision]:
        with self._reader() as connection:
            for (payload,) in connection.execute('SELECT payload FROM decisions ORDER BY id'):
                row = json.loads(payload)
                row['reasons'] = tuple(row['reasons'])
                yield SelectionDecision(**row)

    @property
    def kept_ids(self) -> list[str]:
        return [d.id for d in self.iter_decisions() if d.kept]

    @property
    def rejected_ids(self) -> list[str]:
        return [d.id for d in self.iter_decisions() if not d.kept]

    def iter_records(self) -> Iterator[CorpusRecord]:
        with self._reader() as connection:
            for ident, reference in connection.execute('SELECT id, reference FROM decisions WHERE kept=1 ORDER BY id'):
                with closing(sqlite3.connect(reference)) as source:
                    row = source.execute('SELECT text, metadata FROM outputs WHERE id=?', (ident,)).fetchone()
                yield CorpusRecord(ident, row[0], json.loads(row[1]))

    def close(self) -> None:
        self._closed = True
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        for source_path, token in self.sources.items():
            _remove_reference(source_path, token)
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def select_records(analyses: Iterable[RecordAnalysis], *, selection: SelectionConfig) -> SelectionReport:
    """Build transitive duplicate clusters and emit real output decisions.

    SQLite holds candidate bands, verified edges, and cluster membership. MinHash
    candidates are confirmed by exact shingle Jaccard on the declared basis.
    Candidate recall is probabilistic; fixed signatures/config produce stable
    decisions independent of input order. Exact basis matches bypass LSH.
    Shingle hashes and IDs remain identifying, so retain this private state only
    as long as needed and close the report after exporting selected outputs.
    """
    from datasketch import MinHashLSH
    import numpy as np

    root = _workspace()
    sources: dict[Path, str] = {}
    db = sqlite3.connect(root / 'decisions.sqlite')
    (root / 'decisions.sqlite').chmod(0o600)
    db.executescript('''CREATE TABLE records (id TEXT PRIMARY KEY, reference TEXT, shingles TEXT,
        source_digest TEXT, basis_digest TEXT, quality REAL, cluster TEXT);
        CREATE TABLE bands (band INTEGER, hash BLOB, id TEXT);
        CREATE INDEX band_lookup ON bands(band, hash);
        CREATE INDEX basis_lookup ON records(basis_digest);
        CREATE INDEX cluster_lookup ON records(cluster);
        CREATE TABLE edges (a TEXT, b TEXT);
        CREATE INDEX edge_a ON edges(a);
        CREATE INDEX edge_b ON edges(b);
        CREATE TABLE decisions (id TEXT PRIMARY KEY, reference TEXT, kept INTEGER, payload TEXT);''')
    identity = None
    ranges = []
    iterator = None
    try:
        iterator = iter(analyses)
        for analysis in iterator:
            if not _owned_reference(analysis.output_ref, analysis.ownership_token):
                raise ValueError("output_ref must be an SCT-owned private reference")
            sources[analysis.output_ref] = analysis.ownership_token
            metadata = dict(analysis.signature_metadata)
            try:
                signature_config = ScoreConfig(minhash_num_perm=metadata['num_perm'], minhash_seed=metadata['seed'],
                    minhash_shingle=metadata['shingle'], minhash_min_tokens=metadata['min_tokens'])
                expected = {**minhash_metadata(signature_config), 'schema_version': 1, 'basis': selection.basis}
            except (KeyError, TypeError, ValueError):
                raise ValueError('incompatible signature identity') from None
            if metadata != expected or identity is not None and metadata != identity:
                raise ValueError('incompatible signature identity')
            if identity is None:
                identity = metadata
                if metadata['num_perm'] < 2 or selection.similarity_threshold == 1:
                    ranges = [(0, metadata['num_perm'])]
                else:
                    try:
                        ranges = MinHashLSH(threshold=selection.similarity_threshold,
                                            num_perm=metadata['num_perm']).hashranges
                    except ValueError:
                        ranges = [(0, metadata['num_perm'])]
            with closing(sqlite3.connect(analysis.output_ref)) as source:
                row = source.execute('SELECT shingles FROM outputs WHERE id=?', (analysis.id,)).fetchone()
            if row is None:
                raise ValueError('missing private output reference')
            shingles = set(json.loads(row[0]))
            quality = analysis.scores.get('sct_quality')
            if selection.quality_min is not None and (quality is None or not math.isfinite(quality) or not 0 <= quality <= 1):
                raise ValueError('quality selection needs valid scorer results')
            try:
                db.execute('INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (analysis.id, str(analysis.output_ref), row[0], analysis.source_digest,
                            analysis.basis_digest, quality, analysis.id))
            except sqlite3.IntegrityError:
                raise ValueError('duplicate record id') from None
            if analysis.signature is not None:
                if len(analysis.signature) != metadata['num_perm']:
                    raise ValueError('incompatible signature length')
                hashvalues = np.array(analysis.signature, dtype=np.uint64)
                for band, (start, end) in enumerate(ranges):
                    band_hash = hashvalues[start:end].tobytes()
                    for (candidate,) in db.execute('SELECT id FROM bands WHERE band=? AND hash=?', (band, band_hash)):
                        candidate_row = db.execute('SELECT shingles FROM records WHERE id=?', (candidate,)).fetchone()
                        other = set(json.loads(candidate_row[0]))
                        if shingles and len(shingles & other) / len(shingles | other) >= selection.similarity_threshold:
                            db.execute('INSERT INTO edges VALUES (?, ?)', (analysis.id, candidate))
                    db.execute('INSERT INTO bands VALUES (?, ?, ?)', (band, band_hash, analysis.id))
            # Exact nonempty basis matches also work for documents too short for MinHash.
            if analysis.basis_digest != _digest(''):
                db.execute('INSERT INTO edges SELECT ?, id FROM records WHERE basis_digest=? AND id!=?',
                           (analysis.id, analysis.basis_digest, analysis.id))
        # Propagate the smallest stable ID to a fixed point, including transitive edges.
        while True:
            before = db.total_changes
            db.execute('''UPDATE records SET cluster=(SELECT MIN(r.cluster) FROM records r WHERE
                r.id=records.id OR r.id IN (SELECT b FROM edges WHERE a=records.id
                UNION SELECT a FROM edges WHERE b=records.id))
                WHERE cluster>(SELECT MIN(r.cluster) FROM records r WHERE r.id=records.id OR
                r.id IN (SELECT b FROM edges WHERE a=records.id UNION SELECT a FROM edges WHERE b=records.id))''')
            if db.total_changes == before:
                break
        for ident, reference, source_digest, basis_digest, quality, cluster in db.execute(
                'SELECT id, reference, source_digest, basis_digest, quality, cluster FROM records ORDER BY id'):
            # Keep the smallest quality-eligible record, so a rejected record cannot discard a usable duplicate.
            keeper = db.execute('SELECT MIN(id) FROM records WHERE cluster=? AND (? IS NULL OR quality>=?)',
                                (cluster, selection.quality_min, selection.quality_min)).fetchone()[0] or cluster
            quality_pass = selection.quality_min is None or quality >= selection.quality_min
            duplicate = ident != keeper
            collision = bool(db.execute('SELECT 1 FROM records WHERE basis_digest=? AND source_digest!=? LIMIT 1',
                                        (basis_digest, source_digest)).fetchone())
            reasons = tuple(reason for reason, active in [('quality_below_min', not quality_pass),
                                                          ('duplicate', duplicate)] if active)
            kept = selection.dedup_mode == 'report' or quality_pass and not duplicate
            decision = SelectionDecision(ident, keeper, quality_pass, duplicate, collision, kept, reasons)
            db.execute('INSERT INTO decisions VALUES (?, ?, ?, ?)', (ident, reference, kept, json.dumps(asdict(decision))))
        db.commit()
        return SelectionReport(root, sources)
    except BaseException as error:
        try:
            close_iterator = getattr(iterator, "close", None)
            if close_iterator is not None:
                close_iterator()
        except BaseException as cleanup_error:
            error.add_note("input iterator cleanup failed: " + type(cleanup_error).__name__)
        finally:
            for source_path, token in sources.items():
                _remove_reference(source_path, token)
            shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        db.close()
