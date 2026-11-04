"""User term lists (decision 19): custom_deny_terms, custom_allow_terms, custom_patterns.

SCT ships no list; every term here is test data a user would pass. No model download needed.
"""

import pickle

import pytest

import sct.config as config_module
from sct import TextCleaner
from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor
from sct.utils.constants import terms_regex
from tests.helpers import FakeRunDetector, word_ner

NO_NER = dict(check_ner_process=False)


def _cleaner(mode='placeholder', **fields):
    return TextCleaner(TextCleanerConfig(replacement_mode=mode, **NO_NER, **fields))


# --- defaults and validation -------------------------------------------------------------------------

def test_defaults_are_empty_and_change_nothing():
    cfg = TextCleanerConfig()
    assert cfg.custom_deny_terms is None and cfg.custom_allow_terms == () and cfg.custom_patterns is None
    assert cfg.custom_terms_case_sensitive is False
    text = "Blue Sky ships to bob@x.com"
    assert _cleaner().process(text).lm_text == "Blue Sky ships to <EMAIL>"


def test_legacy_globals(monkeypatch):
    monkeypatch.setattr(config_module, 'CUSTOM_DENY_TERMS', {'PROJECT': ('Blue Sky',)})
    monkeypatch.setattr(config_module, 'CUSTOM_ALLOW_TERMS', ('Acme',))
    monkeypatch.setattr(config_module, 'CUSTOM_PATTERNS', {'TICKET': r'TCK-\d+'})
    monkeypatch.setattr(config_module, 'CUSTOM_TERMS_CASE_SENSITIVE', True)
    cfg = config_module._config_from_module_globals()
    assert dict(cfg.custom_deny_terms) == {'PROJECT': ('Blue Sky',)}
    assert cfg.custom_allow_terms == ('Acme',)
    assert dict(cfg.custom_patterns) == {'TICKET': r'TCK-\d+'}
    assert cfg.custom_terms_case_sensitive is True


@pytest.mark.parametrize("fields, message", [
    ({'custom_deny_terms': {'project': ('x',)}}, "upper case"),
    ({'custom_deny_terms': {'PROJECT': 'Blue Sky'}}, "tuple of strings"),
    ({'custom_deny_terms': {'PROJECT': ('',)}}, "non-empty"),
    ({'custom_deny_terms': {'A': ('acme',), 'B': ('ACME',)}}, "both"),
    ({'custom_deny_terms': ['x']}, "mapping"),
    ({'custom_allow_terms': 'Acme'}, "tuple of strings"),
    ({'custom_allow_terms': ('  ',)}, "non-empty"),
    ({'custom_patterns': {'TICKET': '('}}, "not a valid regex"),
    ({'custom_patterns': {'TICKET': r'\d*'}}, "empty string"),
    ({'custom_patterns': {'TICKET': r'\b'}}, "empty string"),
    ({'custom_patterns': {'TICKET': ''}}, "non-empty regex"),
    ({'custom_patterns': {'<TICKET>': 'x'}}, "upper case"),
    ({'custom_terms_case_sensitive': 'yes'}, "bool"),
])
def test_validation_errors_are_clear(fields, message):
    with pytest.raises(ValueError, match=message):
        TextCleanerConfig(**fields)


def test_same_term_differing_in_case_is_fine_when_case_sensitive():
    cfg = TextCleanerConfig(custom_deny_terms={'A': ('acme',), 'B': ('ACME',)}, custom_terms_case_sensitive=True)
    assert set(cfg.custom_deny_terms) == {'A', 'B'}


def test_config_is_frozen_and_pickles():
    cfg = TextCleanerConfig(custom_deny_terms={'PROJECT': ['Blue Sky']}, custom_allow_terms=['Acme'],
                            custom_patterns={'TICKET': r'TCK-\d+'})
    assert cfg.custom_deny_terms['PROJECT'] == ('Blue Sky',) and cfg.custom_allow_terms == ('Acme',)
    with pytest.raises(TypeError):
        cfg.custom_deny_terms['X'] = ('y',)
    restored = pickle.loads(pickle.dumps(cfg))  # noqa: S301 - trusted round trip
    assert dict(restored.custom_deny_terms) == dict(cfg.custom_deny_terms)
    assert dict(restored.custom_patterns) == dict(cfg.custom_patterns)


# --- term matching -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, found", [
    ("Acme ships", ["Acme"]),
    ("xAcme, Acmes, Acme1, Acmé", []),        # glued to a letter, digit or mark of a spaced script
    ("(Acme) acme-ACME", ["Acme", "acme", "ACME"]),
    ("金额ACME元", ["ACME"]),                  # CJK letters do not glue
    ("我住在東京都", ["東京"]),                  # a CJK term edge is a word edge
    ("ไปสวัสดีครับ", ["สวัสดี"]),               # Thai, no spaces
])
def test_terms_are_unicode_aware_whole_words(text, found):
    assert [m.group() for m in terms_regex(["Acme", "東京", "สวัสดี"]).finditer(text)] == found


def test_case_sensitive_option():
    cleaner = _cleaner(custom_deny_terms={'ORG': ('Acme',)}, custom_terms_case_sensitive=True)
    assert cleaner.process("Acme and ACME").lm_text == "<ORG> and ACME"
    cleaner = _cleaner(custom_deny_terms={'ORG': ('Acme',)})
    assert cleaner.process("Acme and ACME").lm_text == "<ORG> and <ORG>"


def test_longest_term_wins_an_overlap():
    cleaner = _cleaner(custom_deny_terms={'ORG': ('Acme',), 'PRODUCT': ('Acme Cloud',)})
    assert cleaner.process("Acme Cloud by Acme").lm_text == "<PRODUCT> by <ORG>"


def test_cjk_and_thai_deny_terms_in_running_text():
    cleaner = _cleaner(custom_deny_terms={'PROJECT': ('蓝天计划', 'โครงการฟ้า')})
    assert cleaner.process("我们的蓝天计划很好").lm_text == "我们的<PROJECT>很好"
    assert cleaner.process("เริ่มโครงการฟ้าวันนี้").lm_text == "เริ่ม<PROJECT>วันนี้"


# --- interaction with the built-in steps -------------------------------------------------------------

def test_user_rules_run_before_built_in_regex():
    # 0612345678 alone is a phone; the user pattern takes the whole code first.
    cleaner = _cleaner(custom_patterns={'EMPLOYEE_ID': r'EMP-\d{10}'})
    assert cleaner.process("EMP-0612345678 called 0612345678").lm_text == "<EMPLOYEE_ID> called <PHONE>"


def test_allow_terms_keep_regex_detections():
    cleaner = _cleaner(custom_allow_terms=('support@acme.com', 'Acme 2024'))
    out = cleaner.process("Mail support@acme.com or bob@x.com about the Acme 2024 report").lm_text
    assert out == "Mail support@acme.com or <EMAIL> about the Acme 2024 report"


def test_allowed_value_seen_outside_the_term_is_masked_everywhere():
    cleaner = _cleaner(custom_allow_terms=('Acme 2024',))
    out = cleaner.process("Acme 2024 report, founded 2024").lm_text
    assert out == "Acme <YEAR> report, founded <YEAR>"  # no leak: 2024 also stands alone


def test_allow_terms_win_over_deny_terms_and_patterns():
    cleaner = _cleaner(custom_deny_terms={'ORG': ('Acme',)}, custom_patterns={'TICKET': r'TCK-\d+'},
                       custom_allow_terms=('Acme', 'TCK-1'))
    assert cleaner.process("Acme TCK-1 TCK-22").lm_text == "Acme TCK-1 <TICKET>"


def test_allow_terms_keep_ner_detections():
    ner = word_ner(0, FakeRunDetector())
    ner._allow_regex = terms_regex(["Ann"])
    result = ner.ner_process_detailed("Ann met Bob today", positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == "Ann met <PERSON> today"
    assert [f.entity.text for f in result.findings] == ["Bob"]


# --- replacement modes -------------------------------------------------------------------------------

def test_reversible_mode_restores_user_terms():
    cleaner = _cleaner('reversible', custom_deny_terms={'PROJECT': ('Blue Sky',)},
                       custom_patterns={'TICKET': r'TCK-\d+'})
    text = "Blue Sky is TCK-42, see blue sky and bob@x.com"
    result = cleaner.process(text)
    assert result.lm_text == "<PROJECT_0> is <TICKET_0>, see <PROJECT_1> and <EMAIL_0>"
    assert result.metadata['anon_map'].deanonymize(result.lm_text) == text


def test_synthetic_mode_fakes_known_labels_and_tags_the_rest():
    cleaner = _cleaner('synthetic', custom_deny_terms={'PERSON': ('Ann Smith',), 'PROJECT': ('Blue Sky',)})
    out = cleaner.process("Ann Smith runs Blue Sky").lm_text
    assert "Ann Smith" not in out and out.endswith(" runs <PROJECT>")
    assert "<PERSON>" not in out


def test_process_batch_and_counts_are_value_free():
    cleaner = _cleaner(custom_deny_terms={'PROJECT': ('Blue Sky',)}, custom_patterns={'TICKET': r'TCK-\d+'})
    first, second = cleaner.process_batch(["Blue Sky TCK-1", "nothing here"])
    assert first.lm_text == "<PROJECT> <TICKET>" and second.lm_text == "nothing here"
    assert first.metadata['regex_counts'] == {'PROJECT': {'count': 1, 'chars': 8}, 'TICKET': {'count': 1, 'chars': 5}}
    signals = first.signals
    assert signals['pii_types'] == {'PROJECT': 1, 'TICKET': 1} and signals['sensitive']
    assert "Blue" not in repr(signals) and "TCK" not in repr(signals)


def test_process_json():
    cleaner = _cleaner(custom_deny_terms={'PROJECT': ('Blue Sky',)})
    result = process_json(cleaner, {"note": "Blue Sky", "Blue Sky": ["ok", "blue sky"]})
    assert result.value == {"note": "<PROJECT>", "<PROJECT>": ["ok", "<PROJECT>"]}
    assert result.signals['pii_types'] == {'PROJECT': 3}


def test_streaming_does_not_split_a_deny_term():
    cleaner = _cleaner(custom_deny_terms={'PROJECT': ('Blue Sky Initiative',)})
    text = "word " * 10 + "Blue Sky Initiative " + "tail " * 20
    windowed = WindowedStreamingRedactor(cleaner, max_buffer_chars=64, overlap_chars=8)
    out = "".join(windowed.feed(text[i:i + 7]) for i in range(0, len(text), 7)) + windowed.finish()
    assert "<PROJECT>" in out and "Sky" not in out
    buffered = BufferedStreamingRedactor(cleaner)
    buffered.feed(text)
    assert "<PROJECT>" in buffered.finish()


# --- review fixes ------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_allow_term_inside_an_email_does_not_unmask_the_email(mode):
    cleaner = _cleaner(mode, custom_deny_terms={'ORG': ('Acme',)}, custom_allow_terms=('Acme',))
    out = cleaner.process("mail bob@Acme.com today").lm_text
    assert "bob@" not in out and "Acme.com" not in out


def test_windowed_streaming_masks_a_deny_phrase_fed_one_char_at_a_time():
    cleaner = TextCleaner(TextCleanerConfig.from_preset(
        'training', **NO_NER, custom_deny_terms={'SECRET': ('Blue Sky Initiative',)}))
    text = "word " * 20 + "the Blue Sky Initiative is here and more words follow here " * 3
    windowed = WindowedStreamingRedactor(cleaner, max_buffer_chars=64, overlap_chars=8)
    out = "".join(windowed.feed(ch) for ch in text) + windowed.finish()
    assert out == cleaner.process(text).lm_text and "Sky" not in out


def test_windowed_streaming_keeps_a_custom_pattern_with_its_lookbehind():
    cleaner = _cleaner(custom_patterns={'ACCOUNT': r'(?<=account )\d{6}'})
    text = "word " * 9 + "account 123456 tail "
    windowed = WindowedStreamingRedactor(cleaner, max_buffer_chars=64, overlap_chars=8)
    out = "".join(windowed.feed(text[i:i + 7]) for i in range(0, len(text), 7)) + windowed.finish()
    assert "123456" not in out and "<ACCOUNT>" in out


def test_deny_term_matching_a_unicode_case_variant():
    assert _cleaner(custom_deny_terms={'SECRET': ('i',)}).process("go İ now").lm_text == "go <SECRET> now"


def test_empty_deny_term_collection_is_rejected():
    with pytest.raises(ValueError, match="has no terms"):
        TextCleanerConfig(custom_deny_terms={'SECRET': ()})


@pytest.mark.parametrize("term, text", [
    ("Jose\u0301", "ask Jos\u00e9 now"),  # NFD term, NFC text
    ("\uff21\uff23\uff2d\uff25", "ask \uff21\uff23\uff2d\uff25 now"),  # fullwidth, folded in the text
])
def test_terms_are_normalised_like_the_text(term, text):
    cleaner = TextCleaner(TextCleanerConfig.from_preset('training', **NO_NER, custom_deny_terms={'ORG': (term,)}))
    assert cleaner.process(text).lm_text == "ask <ORG> now"
    allowed = TextCleaner(TextCleanerConfig.from_preset(
        'training', **NO_NER, custom_patterns={'ORG': r'\w{4}'}, custom_allow_terms=(term,)))
    assert "<ORG>" not in allowed.process(text).lm_text


def test_overlapping_custom_rules_round_trip_in_reversible_mode():
    cleaner = _cleaner('reversible', custom_deny_terms={'PERSON': ('Ann',)}, custom_patterns={'SECRET': r'.+ secret'})
    result = cleaner.process("Ann secret")
    assert result.lm_text == "<SECRET_0>"
    assert result.metadata['anon_map'].deanonymize(result.lm_text) == "Ann secret"


def test_allow_terms_scale_linearly():
    import time
    cleaner = _cleaner(custom_allow_terms=('Acme 2024',))
    started = time.perf_counter()
    out = cleaner.process("Acme 2024 " * 800).lm_text
    assert time.perf_counter() - started < 2 and "<NUMBER>" not in out


# --- custom_patterns_timeout (ReDoS guard) -------------------------------------------------------------

_SLOW = {'SLOW': r'(a|aa)+$'}  # catastrophic backtracking on 'a' * 40 + '!'
_SLOW_TEXT = 'mail bob@x.com ' + 'a' * 40 + '!'


def _timed(call, limit):
    import time
    started = time.perf_counter()
    try:
        return call()
    finally:
        assert time.perf_counter() - started < limit + 1


def test_timeout_defaults_validation_and_legacy_globals(monkeypatch):
    from sct import CustomPatternTimeout
    cfg = TextCleanerConfig()
    assert (cfg.custom_patterns_timeout, cfg.custom_patterns_on_timeout) == (2.0, 'raise')
    assert TextCleanerConfig(custom_patterns_timeout=None).custom_patterns_timeout is None
    for bad in (0, -1, True, 'x', float('inf'), float('nan')):
        with pytest.raises(ValueError, match='custom_patterns_timeout'):
            TextCleanerConfig(custom_patterns_timeout=bad)
    with pytest.raises(ValueError, match='custom_patterns_on_timeout'):
        TextCleanerConfig(custom_patterns_on_timeout='skip')
    monkeypatch.setattr(config_module, 'CUSTOM_PATTERNS_TIMEOUT', 0.5)
    monkeypatch.setattr(config_module, 'CUSTOM_PATTERNS_ON_TIMEOUT', 'warn')
    cfg = config_module._config_from_module_globals()
    assert (cfg.custom_patterns_timeout, cfg.custom_patterns_on_timeout) == (0.5, 'warn')
    error = pickle.loads(pickle.dumps(CustomPatternTimeout('SLOW', 0.5)))  # noqa: S301 - trusted round trip
    assert isinstance(error, TimeoutError) and error.label == 'SLOW' and "'SLOW'" in str(error)


def test_timeout_raise_names_the_label_not_the_text():
    from sct import CustomPatternTimeout
    cleaner = _cleaner(custom_patterns=_SLOW, custom_patterns_timeout=0.5)
    with pytest.raises(CustomPatternTimeout) as caught:
        _timed(lambda: cleaner.process(_SLOW_TEXT), 0.5)
    assert caught.value.label == 'SLOW' and 'SLOW' in str(caught.value) and 'aaaa' not in str(caught.value)
    with pytest.raises(CustomPatternTimeout):
        _timed(lambda: cleaner.process_batch(['ok', _SLOW_TEXT]), 0.5)
    with pytest.raises(CustomPatternTimeout):
        _timed(lambda: process_json(cleaner, {'k': _SLOW_TEXT}), 0.5)
    with pytest.raises(CustomPatternTimeout):
        _timed(lambda: WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=10).feed(_SLOW_TEXT), 0.5)


@pytest.mark.parametrize('mode', ['placeholder', 'reversible'])
def test_timeout_warn_skips_the_pattern_for_that_document(mode, caplog):
    cleaner = _cleaner(mode, custom_patterns={**_SLOW, 'TICKET': r'TCK-\d+'}, custom_patterns_timeout=0.5,
                       custom_patterns_on_timeout='warn')
    with caplog.at_level('WARNING', logger='sct.sct'):
        result = _timed(lambda: cleaner.process(_SLOW_TEXT + ' TCK-12'), 0.5)
    assert result.lm_text.startswith('mail <EMAIL') and 'a' * 40 + '!' in result.lm_text
    assert '<TICKET' in result.lm_text  # the other patterns still run
    assert 'custom_pattern_timeout' in result.metadata['warnings']
    assert 'custom_pattern_timeout' in result.signals['warnings']
    assert "'SLOW'" in caplog.text and 'aaaa' not in caplog.text
    if mode == 'reversible':  # nothing from the timed-out pattern in the map
        assert {e.entity_type for e in result.metadata['anon_map'].entries} == {'EMAIL', 'TICKET'}
    fast = cleaner.process('only aaa')  # the same pattern on a short document still applies
    assert 'custom_pattern_timeout' not in fast.metadata['warnings'] and 'SLOW' in fast.lm_text
    batch = _timed(lambda: cleaner.process_batch([_SLOW_TEXT, 'aa']), 1.0)
    assert 'custom_pattern_timeout' in batch[0].metadata['warnings']
    assert 'custom_pattern_timeout' not in batch[1].metadata['warnings']


def test_timeout_warn_in_json_and_streaming():
    cleaner = _cleaner(custom_patterns=_SLOW, custom_patterns_timeout=0.5, custom_patterns_on_timeout='warn')
    result = _timed(lambda: process_json(cleaner, {'k': _SLOW_TEXT}), 0.5)
    assert 'custom_pattern_timeout' in result.signals['warnings'] and 'a' * 40 in result.value['k']
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=80, overlap_chars=20)
    out = _timed(lambda: stream.feed(_SLOW_TEXT + ' tail words here ' * 4) + stream.finish(), 3.0)
    assert '<EMAIL>' in out and 'bob@x.com' not in out


def test_normal_patterns_unaffected_by_the_timeout():
    text = 'ticket TCK-123 and TCK-9'
    for timeout in (2.0, None):
        cleaner = _cleaner(custom_patterns={'TICKET': r'TCK-\d+'}, custom_patterns_timeout=timeout)
        result = cleaner.process(text)
        assert result.lm_text == 'ticket <TICKET> and <TICKET>' and 'custom_pattern_timeout' not in result.metadata['warnings']
