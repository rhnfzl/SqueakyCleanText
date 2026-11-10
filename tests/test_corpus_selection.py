"""Corpus decisions preserve inputs and use verified, stable clusters."""
from dataclasses import replace
import importlib.util
from types import SimpleNamespace

import pytest

from sct.corpus import CorpusRecord, SelectionConfig, analyse_records, select_records
from sct.score import ScoreConfig


requires_datasketch = pytest.mark.skipif(
    importlib.util.find_spec('datasketch') is None, reason='install the corpus extra for datasketch selection',
)


class Cleaner:
    def process(self, text):
        return SimpleNamespace(lm_text=text.replace('alice@example.com', '<EMAIL>').replace('bob@example.com', '<EMAIL>'),
                               signals=dict(language='en', language_confidence=1., pii_count=0,
                                            pii_share=None, pii_types=[], warnings=[]))


def run(rows, selection=SelectionConfig(), scorer=None):
    return select_records(analyse_records((CorpusRecord(k, t) for k, t in rows), cleaner=Cleaner(),
                          score_config=ScoreConfig(minhash_shingle=1, minhash_min_tokens=2),
                          selection=selection, scorer=scorer), selection=selection)


@requires_datasketch
def test_report_retains_duplicates_and_apply_emits_one_stable_cleaned_record():
    rows = [('z', 'one two three four'), ('a', 'one two three four')]
    with run(rows) as report:
        assert report.kept_ids == ['a', 'z']
        assert [d.keeper_id for d in report.iter_decisions()] == ['a', 'a']
    with run(rows, SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii')) as report:
        assert report.kept_ids == ['a']
        assert [(r.id, r.text) for r in report.iter_records()] == [('a', 'one two three four')]


@requires_datasketch
def test_transitive_near_clusters_and_reordering_choose_smallest_id():
    rows = [('c', 'one two three four'), ('b', 'one two three four five'), ('a', 'two three four five')]
    cfg = SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii', similarity_threshold=.75)
    for ordered in (rows, rows[::-1]):
        with run(ordered, cfg) as report:
            assert report.kept_ids == ['a']


@requires_datasketch
def test_private_values_do_not_appear_in_repr_or_audit_and_collisions_are_flagged():
    rows = [('a', 'write alice@example.com today please'), ('b', 'write bob@example.com today please')]
    assert 'secret' not in repr(CorpusRecord('a', 'secret', {'secret': 'secret'}))
    with run(rows) as report:
        assert all(d.basis_collision for d in report.iter_decisions())
        assert 'example.com' not in repr(list(report.iter_decisions()))
    with run(rows, SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii')) as report:
        assert report.kept_ids == ['a', 'b']


@requires_datasketch
def test_short_signatures_do_not_equate_unrelated_documents():
    with run([('a', 'one'), ('b', 'two')]) as report:
        assert [d.duplicate for d in report.iter_decisions()] == [False, False]


@requires_datasketch
def test_quality_uses_original_and_filters_only_in_apply():
    def scorer(texts):
        return [0.2 if 'alice@example.com' in text else 0.9 for text in texts]
    rows = [('a', 'write alice@example.com today please'), ('b', 'other text four words')]
    cfg = SelectionConfig(quality_min=.5, dedup_mode='apply', dedup_basis='pre_pii')
    with run(rows, cfg, scorer) as report:
        assert report.kept_ids == ['b']
        assert list(report.iter_decisions())[0].reasons == ('quality_below_min',)
    with pytest.raises(ValueError, match='scorer'):
        run(rows, cfg)


@requires_datasketch
def test_duplicate_ids_and_signature_identity_mismatch_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        run([('a', 'one two'), ('a', 'three four')])
    cfg = SelectionConfig()
    analyses = list(analyse_records([CorpusRecord('a', 'one two'), CorpusRecord('b', 'one two')],
                    cleaner=Cleaner(), score_config=ScoreConfig(), selection=cfg))
    analyses[1] = replace(analyses[1], signature_metadata={'seed': 999})
    with pytest.raises(ValueError, match='signature'):
        select_records(analyses, selection=cfg)


@pytest.mark.parametrize('kwargs', [dict(dedup_mode='apply'), dict(similarity_threshold=0),
    dict(similarity_threshold=1.1), dict(quality_min=float('nan')), dict(keeper='first')])
def test_invalid_selection_config_fails_before_processing(kwargs):
    with pytest.raises(ValueError):
        SelectionConfig(**kwargs)


@pytest.mark.parametrize('num_perm,threshold', [(1, .9), (128, 1.)])
@requires_datasketch
def test_supported_signature_sizes_and_exact_threshold_work(num_perm, threshold):
    cfg = SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii', similarity_threshold=threshold)
    analyses = analyse_records([CorpusRecord('b', 'one two three four five'),
                               CorpusRecord('a', 'one two three four five')], cleaner=Cleaner(),
                              score_config=ScoreConfig(minhash_num_perm=num_perm), selection=cfg)
    with select_records(analyses, selection=cfg) as report:
        assert report.kept_ids == ['a']


@requires_datasketch
def test_first_signature_cannot_claim_foreign_tokenizer():
    cfg = SelectionConfig()
    analyses = list(analyse_records([CorpusRecord('a', 'one two three four five')], cleaner=Cleaner(),
                    score_config=ScoreConfig(), selection=cfg))
    analyses[0] = replace(analyses[0], signature_metadata={**analyses[0].signature_metadata, 'tokeniser': 'foreign'})
    with pytest.raises(ValueError, match='signature'):
        select_records(analyses, selection=cfg)


@requires_datasketch
def test_report_references_release_private_text_on_context_exit():
    with run([('a', 'private one two three'), ('b', 'other four five six')]) as report:
        sources = list(report.sources)
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in sources)
        assert all(p.parent.stat().st_mode & 0o777 == 0o700 for p in sources)
        assert not hasattr(next(report.iter_decisions()), 'text')
        root = report.root
    assert not root.exists()
    assert all(not p.exists() for p in sources)


@requires_datasketch
def test_candidate_band_collision_does_not_remove_dissimilar_text(monkeypatch):
    from datasketch import MinHashLSH
    monkeypatch.setattr(MinHashLSH, '__init__', lambda self, **kwargs: setattr(self, 'hashranges', [(0, 0)]))
    cfg = SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii')
    with run([('a', 'one two three four'), ('b', 'five six seven eight')], cfg) as report:
        assert report.kept_ids == ['a', 'b']


@pytest.mark.parametrize('values', [[], [float('nan')], [1.1], [0.5, 0.6]])
@requires_datasketch
def test_invalid_scorer_output_is_rejected_without_partial_selection(values):
    with pytest.raises(ValueError, match='scorer'):
        run([('a', 'one two three four')], SelectionConfig(quality_min=.5), lambda texts: values)


@requires_datasketch
def test_analysis_retains_no_source_records_or_text_in_memory(caplog):
    import weakref
    references = []

    def records():
        for index in range(50):
            record = CorpusRecord(str(index), f'private source value number {index}')
            references.append(weakref.ref(record))
            yield record

    cfg = SelectionConfig()
    analyses = list(analyse_records(records(), cleaner=Cleaner(), score_config=ScoreConfig(), selection=cfg))
    assert all(ref() is None for ref in references)
    assert all('private source' not in repr(row) for row in analyses)
    with select_records(analyses, selection=cfg) as report:
        assert len(report.kept_ids) == 50
    assert 'private source' not in caplog.text


@requires_datasketch
def test_report_quality_decisions_preserve_low_quality_records():
    with run([('a', 'one two three four')], SelectionConfig(quality_min=.5), lambda texts: [.1]) as report:
        assert report.kept_ids == ['a']
        assert list(report.iter_decisions())[0].quality_pass is False


@requires_datasketch
def test_empty_corpus_has_empty_output():
    with run([]) as report:
        assert report.kept_ids == []
        assert list(report.iter_records()) == []


def test_identifying_record_id_is_private_in_repr():
    assert 'private-id' not in repr(CorpusRecord('private-id', 'text'))


@requires_datasketch
def test_private_sqlite_connections_close_on_exhaustion_early_close_and_report_close(monkeypatch):
    import sqlite3
    connections = []
    connect = sqlite3.connect

    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', tracked)
    with run([('a', 'private one two three'), ('b', 'other four five six')]) as report:
        assert len(list(report.iter_records())) == 2
        assert len(list(report.iter_decisions())) == 2
        early = report.iter_records()
        next(early)
        early.close()
        active = report.iter_decisions()
        next(active)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')
    active.close()


@requires_datasketch
def test_failed_selection_closes_its_private_read_connections(monkeypatch):
    import sqlite3
    connections = []
    connect = sqlite3.connect

    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', tracked)
    cfg = SelectionConfig(quality_min=.5)
    analyses = analyse_records([CorpusRecord('a', 'one two three four')], cleaner=Cleaner(),
                              score_config=ScoreConfig(), selection=SelectionConfig())
    with pytest.raises(ValueError, match='scorer'):
        select_records(analyses, selection=cfg)
    analyses.close()
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')


@requires_datasketch
def test_caller_supplied_output_reference_never_owns_its_parent(tmp_path):
    cfg = SelectionConfig()
    analyses = list(analyse_records([CorpusRecord('a', 'one two three four five')], cleaner=Cleaner(),
                                   score_config=ScoreConfig(), selection=cfg))
    external = tmp_path / 'outputs.sqlite'
    external.write_text('caller data')
    with pytest.raises(ValueError, match='owned'):
        select_records([replace(analyses[0], output_ref=external)], selection=cfg)
    assert external.read_text() == 'caller data'
    with select_records(analyses, selection=cfg):
        pass


@requires_datasketch
def test_input_close_failure_preserves_original_error_and_removes_private_state():
    cfg = SelectionConfig()
    analyses = list(analyse_records([CorpusRecord('a', 'one two three four five')], cleaner=Cleaner(),
                                   score_config=ScoreConfig(), selection=cfg))

    class Broken:
        def __iter__(self):
            return self
        def __next__(self):
            if analyses:
                return analyses.pop()
            raise ValueError('original input failure')
        def close(self):
            raise RuntimeError('close failure')

    source = analyses[0].output_ref
    with pytest.raises(ValueError, match='original input failure'):
        select_records(Broken(), selection=cfg)
    assert not source.exists()


@requires_datasketch
def test_iterator_construction_failure_removes_selection_workspace(monkeypatch):
    import sct.corpus.selection as module
    roots = []
    workspace = module._workspace
    def tracked():
        path = workspace()
        roots.append(path)
        return path
    monkeypatch.setattr(module, '_workspace', tracked)

    class Broken:
        def __iter__(self):
            raise ValueError('cannot construct iterator')

    with pytest.raises(ValueError, match='cannot construct iterator'):
        select_records(Broken(), selection=SelectionConfig())
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize('alteration', ['missing_token', 'source_symlink', 'parent_symlink'])
@requires_datasketch
def test_private_reference_capability_rejects_missing_token_and_symlinks(tmp_path, alteration):
    cfg = SelectionConfig()
    analyses = list(analyse_records([CorpusRecord('a', 'one two three four five')], cleaner=Cleaner(),
                                   score_config=ScoreConfig(), selection=cfg))
    original = analyses[0]
    if alteration == 'missing_token':
        altered = replace(original, ownership_token='')
    elif alteration == 'source_symlink':
        link = tmp_path / 'outputs.sqlite'
        link.symlink_to(original.output_ref)
        altered = replace(original, output_ref=link)
    else:
        link = tmp_path / 'sct-corpus-link'
        link.symlink_to(original.output_ref.parent, target_is_directory=True)
        altered = replace(original, output_ref=link / 'outputs.sqlite')
    with pytest.raises(ValueError, match='owned'):
        select_records([altered], selection=cfg)
    assert original.output_ref.exists()
    with select_records([original], selection=cfg) as report:
        assert report.kept_ids == ['a']
