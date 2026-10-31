"""0.8.0 full test round 2 (Linux run A, Mac round 2) bug fixes."""

import types

import pytest

from sct import TextCleaner, TextCleanerConfig


# --- Linux 2: Lingua confidence is rounded so workers and in-process agree ---

def test_language_confidence_is_rounded():
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False, check_detect_language=True))
    lang = types.SimpleNamespace(name="ENGLISH")
    noisy = [types.SimpleNamespace(language=lang, value=0.5735166974819533)]
    cleaner._detector = types.SimpleNamespace(compute_language_confidence_values=lambda t: noisy)
    assert cleaner._detect_language("x")[1] == 0.573517


# --- Linux 5: importing sct does not pull presidio (and through it torch/GLiNER) ---

def test_import_sct_does_not_load_presidio():
    import subprocess
    import sys
    code = "import sct, sys; print('presidio_anonymizer' in sys.modules)"
    out = subprocess.run(  # noqa: S603 - fixed argv, own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- Mac C: isolated-symbol removal keeps brackets and punctuation glued to a word or placeholder ---

@pytest.mark.parametrize("text, expected", [
    ("He was (born 1980) here.", "He was (born <YEAR>) here."),
    ("He was (born 1980).", "He was (born <YEAR>)."),
    ("He was (born in Paris). Ok", "He was (born in Paris). Ok"),
    ("Voor vragen, bel +31 20 123 4567.", "Voor vragen, bel <PHONE>."),
    ("end , start", "end start"),
])
def test_isolated_symbols_need_space_on_both_sides(text, expected):
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    assert cleaner.process(text).lm_text == expected
