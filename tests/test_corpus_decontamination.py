"""Caller reference matching without raw reference artifacts."""
import importlib.util

import json
import pytest
from sct.corpus import CorpusRecord


def test_reference_matches_and_near_misses(tmp_path):
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig
    text = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
    refs = [CorpusRecord('private-reference', text)]
    with ReferenceIndex.build(tmp_path / 'references.sqlite', refs, config=ReferenceConfig()) as index:
        assert index.match(text).reference_ids == ('private-reference',)
        assert not index.match('different words describing a remote unrelated subject about farming and rivers').reference_ids
        assert not index.match('alpha beta').reference_ids
        assert not index.match('alpha beta gamma delta epsilon other words entirely away from this sentence').reference_ids
        fingerprint = index.identity
    assert text.encode() not in (tmp_path / 'references.sqlite').read_bytes()
    assert fingerprint
    with ReferenceIndex(tmp_path / 'references.sqlite') as index:
        assert index.identity == fingerprint
        assert index.match(text).reference_ids
    with pytest.raises(ValueError):
        ReferenceConfig(mode='apply')


def test_reference_config_changes_identity(tmp_path):
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig
    refs = [CorpusRecord('r', 'one two three four five six seven eight')]
    with ReferenceIndex.build(tmp_path / 'a', refs) as first, ReferenceIndex.build(
        tmp_path / 'b', refs, config=ReferenceConfig(threshold=.5)
    ) as second:
        assert first.identity != second.identity


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in ('datatrove', 'spacy', 'orjson')),
                    reason='install corpus extra for reference recipe execution')
@pytest.mark.parametrize('mode,kept', [('report', 2), ('apply', 1)])
def test_reference_recipe_report_and_apply(tmp_path, mode, kept):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig, curate_resumable, RunIdentityMismatch
    text = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
    other = 'different words describe a remote unrelated subject about farming and rivers under cloudy skies'
    source = tmp_path / 'source'
    source.write_text('\n'.join(json.dumps({'id': key, 'text': value}) for key, value in [('a', text), ('b', other)]))
    with ReferenceIndex.build(tmp_path / 'refs', [CorpusRecord('r', text)],
                              config=ReferenceConfig(mode=mode, basis='pii_normalized')) as refs:
        result = curate_resumable(source, tmp_path / 'run', references=refs,
                                  config=TextCleanerConfig(check_ner_process=False), retention='caller_managed')
        assert result.counts == {'input': 2, 'kept': kept, 'rejected': 2 - kept}
        rows = [json.loads(line) for line in (tmp_path / 'run' / 'reference-decisions.jsonl').read_text().splitlines()]
        assert rows[0]['reference_ids'] == ['r'] and rows[1]['reference_ids'] == []
        assert text not in (tmp_path / 'run' / 'reference-decisions.jsonl').read_text()
        replay = curate_resumable(source, tmp_path / 'run', references=refs,
                                   config=TextCleanerConfig(check_ner_process=False), retention='caller_managed')
        assert replay.counts == result.counts
    with ReferenceIndex.build(tmp_path / 'changed', [CorpusRecord('r', other)]) as changed:
        with pytest.raises(RunIdentityMismatch):
            curate_resumable(source, tmp_path / 'run', references=changed,
                             config=TextCleanerConfig(check_ner_process=False), retention='caller_managed')
    assert source.exists()


def test_reference_identity_is_independent_of_insertion_order(tmp_path):
    from sct.corpus.runner import ReferenceIndex
    refs = [CorpusRecord('a', 'one two three four five six seven eight'),
            CorpusRecord('b', 'another separate reference contains several other distinctive tokens')]
    with ReferenceIndex.build(tmp_path / 'forward', refs) as forward, ReferenceIndex.build(
        tmp_path / 'reverse', reversed(refs)
    ) as reverse:
        assert forward.identity == reverse.identity


@pytest.mark.parametrize('extra,matched', [(3, False), (2, True), (1, True)])
def test_reference_jaccard_below_equal_above_cutoff(tmp_path, extra, matched):
    from fractions import Fraction
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig
    reference_tokens = [f'token{number}' for number in range(18)]
    query_tokens = reference_tokens + [f'additional{number}' for number in range(extra)]
    measured = Fraction(18, 18 + extra)
    cutoff = Fraction(9, 10)
    assert (measured >= cutoff) == matched
    if extra == 2:
        assert measured == cutoff
    config = ReferenceConfig(threshold=.9, shingle=1, min_tokens=1)
    with ReferenceIndex.build(tmp_path / 'references', [CorpusRecord('reference', ' '.join(reference_tokens))],
                              config=config) as references:
        assert bool(references.match(' '.join(query_tokens)).reference_ids) == matched


def test_reference_configuration_and_identity_are_read_only(tmp_path):
    from dataclasses import replace
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig
    with ReferenceIndex.build(tmp_path / 'references', [CorpusRecord('ref', 'one two three four five six')],
                              config=ReferenceConfig(basis='pii_normalized')) as references:
        original = references.identity
        with pytest.raises(AttributeError):
            references.config = replace(references.config, mode='apply')
        with pytest.raises(AttributeError):
            references.identity = 'replacement'
        assert references.identity == original and references.config.mode == 'report'


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in ('datatrove', 'spacy', 'orjson')),
                    reason='install corpus extra for reference config resume guard')
def test_changed_reference_config_refuses_completed_reuse(tmp_path):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import ReferenceIndex, ReferenceConfig, curate_resumable, RunIdentityMismatch
    text = 'one two three four five six seven eight nine ten eleven twelve'
    source = tmp_path / 'source'
    source.write_text(json.dumps({'id': 'record', 'text': text}))
    kwargs = {'config': TextCleanerConfig(check_ner_process=False), 'retention': 'caller_managed'}
    refs = [CorpusRecord('reference', text)]
    with ReferenceIndex.build(tmp_path / 'report', refs,
                              config=ReferenceConfig(basis='pii_normalized')) as report:
        assert curate_resumable(source, tmp_path / 'run', references=report, **kwargs).counts['kept'] == 1
    with ReferenceIndex.build(tmp_path / 'apply', refs,
                              config=ReferenceConfig(mode='apply', basis='pii_normalized')) as apply:
        with pytest.raises(RunIdentityMismatch):
            curate_resumable(source, tmp_path / 'run', references=apply, **kwargs)
        assert curate_resumable(source, tmp_path / 'fresh', references=apply, **kwargs).counts['kept'] == 0
