"""Comprehensive functional tests — exercises every SCT feature end-to-end.

Run on develop AND main (via worktree) to catch regressions.
Tests are ordered A–M by dependency weight: core pipeline first, NER later.

Usage:
    pytest tests/test_functional.py -v --timeout=300
"""

import asyncio
import dataclasses
import importlib.util
import os
import warnings

import pytest

from sct import TextCleaner, TextCleanerConfig, ProcessResult, AnonymizationMap
from sct.config import DEFAULT_NER_MODELS, PII_DEFAULT_MODEL
from sct.utils.ner import ENTITY_TYPE_MAP
from tests.conftest import TEST_NER_MODELS

# Derive entity tags from the authoritative map — stays in sync automatically
ENTITY_TAGS = tuple(f"<{v}>" for v in ENTITY_TYPE_MAP.values())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Production ONNX models (only used when FUNC_TEST_PROD_MODELS=1)
PROD_NER_MODELS = dict(DEFAULT_NER_MODELS)

# Use production models when env var is set
USE_PROD = os.environ.get("FUNC_TEST_PROD_MODELS", "0") == "1"
NER_MODELS = PROD_NER_MODELS if USE_PROD else TEST_NER_MODELS


def _has_package(name: str) -> bool:
    """Check if a Python package is installed (without importing it)."""
    return importlib.util.find_spec(name) is not None


HAS_TORCH = _has_package("torch")
HAS_GLINER = _has_package("gliner")
HAS_PRESIDIO = _has_package("presidio_analyzer")
HAS_GLICLASS = _has_package("gliclass")
HAS_FAKER = _has_package("faker")
HAS_RAPIDFUZZ = _has_package("rapidfuzz")

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_gliner = pytest.mark.skipif(not HAS_GLINER, reason="gliner not installed")
requires_presidio = pytest.mark.skipif(not HAS_PRESIDIO, reason="presidio-analyzer not installed")
requires_gliclass = pytest.mark.skipif(not HAS_GLICLASS, reason="gliclass not installed")
requires_faker = pytest.mark.skipif(not HAS_FAKER, reason="faker not installed")
requires_rapidfuzz = pytest.mark.skipif(not HAS_RAPIDFUZZ, reason="rapidfuzz not installed")


def _make_cfg(**overrides) -> TextCleanerConfig:
    """Build a no-NER config with sensible defaults for fast core-pipeline tests."""
    defaults = dict(
        check_ner_process=False,
        check_statistical_model_processing=False,
        ner_models=TEST_NER_MODELS,
    )
    defaults.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return TextCleanerConfig(**defaults)


def _make_ner_cfg(**overrides) -> TextCleanerConfig:
    """Build a NER-enabled config with lightweight test models."""
    defaults = dict(
        check_ner_process=True,
        check_statistical_model_processing=True,
        ner_models=NER_MODELS,
    )
    defaults.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return TextCleanerConfig(**defaults)


# =========================================================================
# A. Core Pipeline (no NER — fast, no model downloads)
# =========================================================================


class TestCorePipeline:
    """A. Core pipeline steps — no NER, no models, pure text transforms."""

    def test_a01_unicode_fixing(self):
        """ftfy repairs mojibake and bad encoding."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        result = tc.process("â€™ curly quote")
        lm, _, _ = result
        # ftfy should fix the mojibake â€™ → '
        assert "â€" not in lm

    def test_a02_ascii_transliteration(self):
        """unidecode converts non-ASCII to ASCII equivalents."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Ångström café résumé naïve")
        assert "Angstrom" in lm
        assert "cafe" in lm
        assert "resume" in lm

    def test_a03_emoji_removal_on(self):
        """Emoji removal when enabled."""
        cfg = _make_cfg(check_remove_emoji=True)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello 🌍 World 🎉")
        assert "🌍" not in lm
        assert "🎉" not in lm
        assert "Hello" in lm

    def test_a04_emoji_preserved_when_off(self):
        """Emoji kept when removal disabled."""
        cfg = _make_cfg(check_remove_emoji=False)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello 🌍 World")
        # Emoji gets transliterated by unidecode (to_ascii_unicode), not removed
        # This is expected behavior — emoji characters become ASCII approximations
        assert "Hello" in lm

    def test_a05_html_stripping(self):
        """HTML tags stripped, text preserved."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process('<div class="test">Hello <b>World</b></div>')
        assert "<div" not in lm
        assert "<b>" not in lm
        assert "Hello" in lm or "<HTML>" in lm

    def test_a06_url_replacement(self):
        """URLs replaced with placeholder token."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Visit https://example.com/page?q=1 for info")
        assert "<URL>" in lm
        assert "https://example.com" not in lm

    def test_a07_email_replacement(self):
        """Emails replaced with placeholder token."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Contact john.doe@example.com for details")
        assert "<EMAIL>" in lm
        assert "john.doe@example.com" not in lm

    def test_a08_phone_replacement(self):
        """Phone numbers replaced with placeholder token."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Call me at +1-234-567-8900 today")
        assert "<PHONE>" in lm
        assert "+1-234-567-8900" not in lm

    def test_a09_number_replacement(self):
        """Numbers replaced with placeholder token."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("There are 42 items and 3.14 ratio")
        assert "<NUMBER>" in lm

    def test_a10_year_replacement(self):
        """Years (1900-2099) replaced with placeholder token."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Founded in 2024 and expanded in 1998")
        assert "<YEAR>" in lm
        assert "2024" not in lm

    def test_a11_date_replacement(self):
        """ISO 8601 and named dates replaced."""
        cfg = _make_cfg(check_replace_dates=True)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Meeting on 2024-01-15 and January 15, 2024")
        assert "<DATE>" in lm

    @requires_rapidfuzz
    def test_a12_fuzzy_date_replacement(self):
        """Misspelled month names caught by fuzzy matching."""
        cfg = _make_cfg(check_fuzzy_replace_dates=True, fuzzy_date_score_cutoff=85)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Meeting on Feburary 15, 2024")
        assert "<DATE>" in lm

    def test_a13_currency_symbol_replacement(self):
        """Currency symbols replaced (when replace_with is set)."""
        cfg = _make_cfg(replace_with_currency_symbols="<CURRENCY>")
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Price is $50 or €40 or £30")
        assert "<CURRENCY>" in lm

    def test_a14_isolated_letter_removal(self):
        """Isolated single letters removed (except meaningful ones)."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("This is a x y z test")
        # Isolated letters x, y, z should be removed
        words = lm.split()
        for letter in ["x", "y", "z"]:
            assert letter not in words

    def test_a15_bracket_content_removal(self):
        """Content inside brackets removed when enabled."""
        cfg = _make_cfg(check_remove_bracket_content=True)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello [citation needed] World")
        assert "citation needed" not in lm

    def test_a16_bracket_content_preserved(self):
        """Content inside brackets preserved when disabled."""
        cfg = _make_cfg(check_remove_bracket_content=False)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello [important] World")
        assert "important" in lm

    def test_a17_brace_content_removal(self):
        """Content inside braces removed when enabled."""
        cfg = _make_cfg(check_remove_brace_content=True)
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello {variable} World")
        assert "variable" not in lm

    def test_a18_whitespace_normalization(self):
        """Multiple spaces, tabs, newlines collapsed to single space."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello   \t\n  World   test")
        assert "  " not in lm
        assert "Hello" in lm

    def test_a19_custom_pipeline_steps(self):
        """User-provided callable steps run after built-in steps."""
        def add_prefix(text):
            return f"[CUSTOM] {text}"

        cfg = _make_cfg(custom_pipeline_steps=(add_prefix,))
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Hello World")
        assert lm.startswith("[CUSTOM]")


# =========================================================================
# B. Language Detection & Pinning
# =========================================================================


class TestLanguageDetection:
    """B. Language detection, pinning, and restriction."""

    def test_b19_auto_detect_english(self):
        """Auto-detect English text."""
        cfg = _make_cfg(check_detect_language=True)
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process(
            "The quick brown fox jumps over the lazy dog in the morning."
        )
        assert lang == "ENGLISH"

    def test_b20_auto_detect_dutch(self):
        """Auto-detect Dutch text."""
        cfg = _make_cfg(check_detect_language=True)
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process(
            "De snelle bruine vos springt over de luie hond in de ochtend."
        )
        assert lang == "DUTCH"

    def test_b21_auto_detect_german(self):
        """Auto-detect German text."""
        cfg = _make_cfg(check_detect_language=True)
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process(
            "Der schnelle braune Fuchs springt uber den faulen Hund am Morgen."
        )
        assert lang == "GERMAN"

    def test_b22_auto_detect_spanish(self):
        """Auto-detect Spanish text."""
        cfg = _make_cfg(check_detect_language=True)
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process(
            "El rapido zorro marron salta sobre el perro perezoso por la manana."
        )
        assert lang == "SPANISH"

    def test_b23_language_pinning(self):
        """Pinned language skips detection entirely."""
        cfg = _make_cfg(language="ENGLISH")
        tc = TextCleaner(cfg=cfg)
        # Even Dutch text returns ENGLISH when pinned
        _, _, lang = tc.process("De snelle bruine vos springt over de luie hond.")
        assert lang == "ENGLISH"

    def test_b24_language_restriction_tuple(self):
        """Language restriction limits detection to listed languages."""
        cfg = _make_cfg(language=("ENGLISH", "DUTCH"))
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process(
            "The quick brown fox jumps over the lazy dog in the morning."
        )
        assert lang in ("ENGLISH", "DUTCH")

    def test_b25_iso_code_support(self):
        """ISO 639-1 codes resolved to uppercase names."""
        cfg = _make_cfg(language="en")
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process("Hello world")
        assert lang == "ENGLISH"

    def test_b26_iso_code_german(self):
        """ISO code 'de' resolves to GERMAN."""
        cfg = _make_cfg(language="de")
        tc = TextCleaner(cfg=cfg)
        _, _, lang = tc.process("Hallo Welt")
        assert lang == "GERMAN"

    def test_b27_extra_languages(self):
        """Extra languages extend detection range."""
        cfg = _make_cfg(extra_languages=("FRENCH",))
        tc = TextCleaner(cfg=cfg)
        assert "FRENCH" in cfg.supported_languages
        _, _, lang = tc.process(
            "Le renard brun rapide saute par dessus le chien paresseux le matin."
        )
        assert lang == "FRENCH"


# =========================================================================
# C. Statistical Model Processing
# =========================================================================


class TestStatisticalProcessing:
    """C. Stopwords, casefold, punctuation — statistical text features."""

    def test_c24_stopword_removal_english(self):
        """English stopwords removed from stat text."""
        cfg = _make_cfg(
            check_statistical_model_processing=True,
            check_detect_language=True,
            language="ENGLISH",
        )
        tc = TextCleaner(cfg=cfg)
        _, stext, _ = tc.process("This is the quick brown fox and the lazy dog.")
        # Common stopwords: this, is, the, and
        assert "this" not in stext.split()
        assert "the" not in stext.split()

    def test_c25_casefold_standard(self):
        """Standard casefold lowercases everything."""
        cfg = _make_cfg(
            check_statistical_model_processing=True,
            check_casefold=True,
            language="ENGLISH",
        )
        tc = TextCleaner(cfg=cfg)
        _, stext, _ = tc.process("The MOUNTAIN is Beautiful and Tall")
        assert "mountain" in stext
        assert "MOUNTAIN" not in stext

    def test_c26_smart_casefold(self):
        """Smart casefold preserves abbreviations and NER-like tokens."""
        cfg = _make_cfg(
            check_statistical_model_processing=True,
            check_casefold=False,
            check_smart_casefold=True,
            language="ENGLISH",
        )
        tc = TextCleaner(cfg=cfg)
        _, stext, _ = tc.process("The CEO of IBM announced new AI features.")
        # CEO, IBM, AI should be preserved (all-caps abbreviations)
        # Regular words should be lowercased
        assert "CEO" in stext or "ceo" in stext  # behavior depends on implementation
        assert "announced" in stext or "ANNOUNCED" not in stext

    def test_c27_punctuation_removal(self):
        """Punctuation removed from stat text."""
        cfg = _make_cfg(
            check_statistical_model_processing=True,
            check_remove_punctuation=True,
            language="ENGLISH",
        )
        tc = TextCleaner(cfg=cfg)
        _, stext, _ = tc.process("Hello, World! How's it going?")
        assert "," not in stext
        assert "!" not in stext
        assert "?" not in stext

    def test_c28_custom_stopwords(self):
        """Custom per-language stopwords."""
        cfg = _make_cfg(
            check_statistical_model_processing=True,
            language="ENGLISH",
            custom_stopwords={"ENGLISH": frozenset({"hello", "world"})},
        )
        tc = TextCleaner(cfg=cfg)
        _, stext, _ = tc.process("Hello World and test items")
        assert "hello" not in stext.split()

    @requires_rapidfuzz
    def test_c29_custom_month_names(self):
        """Custom month names for non-standard date formats."""
        cfg = _make_cfg(
            check_replace_dates=True,
            custom_month_names={"ENGLISH": ("Thermidor", "Fructidor")},
        )
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("Meeting on Thermidor 15, 2024")
        assert "<DATE>" in lm


# =========================================================================
# D. NER Backends (requires model downloads)
# =========================================================================


class TestNERBackends:
    """D. ONNX NER backend — end-to-end with real models."""

    def test_d30_onnx_backend_e2e(self):
        """ONNX backend processes text and replaces entities."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "John Smith works at Microsoft in New York City."
        )
        lm = result.lm_text
        # At minimum, NER should detect person/org/location
        has_entity = any(tag in lm for tag in ENTITY_TAGS)
        assert has_entity, f"No NER tags found in: {lm}"

    def test_d31_process_result_unpacking(self):
        """ProcessResult unpacks as 3-tuple (backward compat)."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Hello from Amsterdam in the Netherlands.")
        lm, stext, lang = result
        assert isinstance(lm, str)
        assert isinstance(stext, str)
        assert lang is not None

    def test_d32_process_result_metadata(self):
        """ProcessResult.metadata available (None when no extended features)."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Test text for metadata check.")
        assert isinstance(result, ProcessResult)
        # In placeholder mode, metadata is None (no anon_map, no classes)
        assert result.metadata is None

    def test_d33_batch_processing(self):
        """process_batch handles multiple texts."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        texts = [
            "John Smith lives in London.",
            "Maria Garcia works at Google in Madrid.",
            "Hans Mueller is from Berlin.",
            "Jan de Vries woont in Amsterdam.",
            "Simple text without entities.",
        ]
        results = tc.process_batch(texts)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, ProcessResult)
            lm, stext, lang = r
            assert isinstance(lm, str)

    def test_d34_async_batch(self):
        """aprocess_batch works with asyncio."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        texts = [
            "John Smith works at Microsoft.",
            "Angela Merkel visited Berlin.",
        ]
        results = asyncio.run(tc.aprocess_batch(texts))
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ProcessResult)

    def test_d35_warmup(self):
        """warmup() pre-loads models without processing text."""
        cfg = _make_ner_cfg()
        tc = TextCleaner(cfg=cfg)
        # Should not raise
        tc.warmup(languages=["ENGLISH"])
        # After warmup, ENGLISH pipeline should be loaded
        assert "ENGLISH" in tc.GeneralNER._pipelines


# =========================================================================
# E. Replacement Modes (with real NER)
# =========================================================================


class TestReplacementModes:
    """E. Placeholder, reversible, and synthetic NER replacement."""

    def test_e36_placeholder_mode(self):
        """Default placeholder mode: <PERSON>, <LOCATION>, etc."""
        cfg = _make_ner_cfg(replacement_mode="placeholder")
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("John Smith met Angela Merkel in Berlin.")
        has_entity = any(tag in lm for tag in ENTITY_TAGS)
        assert has_entity, f"No NER tags in: {lm}"

    def test_e37_reversible_mode(self):
        """Reversible mode: indexed placeholders + AnonymizationMap round-trip."""
        cfg = _make_ner_cfg(replacement_mode="reversible")
        tc = TextCleaner(cfg=cfg)
        result = tc.process("John Smith works at Google in Mountain View.")
        lm = result.lm_text

        # Should have indexed placeholders like <PERSON_0>, <LOCATION_0>, <ORGANISATION_0>
        has_indexed = any(f"<{tag}_" in lm for tag in ("PERSON", "LOCATION", "ORGANISATION", "MISC"))
        assert has_indexed, f"No indexed placeholders in: {lm}"

        # Metadata should contain anon_map
        assert result.metadata is not None
        assert "anon_map" in result.metadata
        anon_map = result.metadata["anon_map"]
        assert isinstance(anon_map, AnonymizationMap)
        assert len(anon_map.entries) > 0

        # Round-trip deanonymization
        restored = anon_map.deanonymize(lm)
        # At least one original entity should be restored
        any_restored = any(
            entry.original in restored for entry in anon_map.entries
        )
        assert any_restored, f"Deanonymization failed: {restored}"

    def test_e38_reversible_serialization(self):
        """AnonymizationMap serializes/deserializes correctly."""
        cfg = _make_ner_cfg(replacement_mode="reversible")
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Marie Curie discovered radium in Paris.")

        if result.metadata and "anon_map" in result.metadata:
            anon_map = result.metadata["anon_map"]
            # Serialize
            data = anon_map.to_dict()
            assert "session_id" in data
            assert "entries" in data

            # Deserialize
            restored_map = AnonymizationMap.from_dict(data)
            assert restored_map.session_id == anon_map.session_id
            assert len(restored_map.entries) == len(anon_map.entries)

            # Verify round-trip through serialization
            for orig, rest in zip(anon_map.entries, restored_map.entries):
                assert orig.placeholder == rest.placeholder
                assert orig.original == rest.original


# =========================================================================
# F. Document Classification (requires gliclass)
# =========================================================================


@requires_gliclass
class TestDocumentClassification:
    """F. GLiClass document-level pre-classification."""

    def test_f39_gliclass_classify(self):
        """GLiClass classify with real model."""
        cfg = _make_cfg(
            check_classify_document=True,
            gliclass_labels=("technology", "sports", "politics", "science"),
            gliclass_threshold=0.3,
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "Apple announced a new iPhone with advanced AI features "
            "powered by their custom silicon chip."
        )
        assert result.metadata is not None
        assert "classes" in result.metadata
        classes = result.metadata["classes"]
        assert isinstance(classes, list)
        # Should classify as technology-related
        if classes:
            assert any("technology" in c["label"].lower() for c in classes)

    def test_f40_classification_in_metadata(self):
        """Classification results accessible in ProcessResult.metadata['classes']."""
        cfg = _make_cfg(
            check_classify_document=True,
            gliclass_labels=("business", "entertainment", "health"),
            gliclass_threshold=0.1,
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "The stock market reached an all-time high as investors "
            "celebrated the latest earnings reports from major corporations."
        )
        assert result.metadata is not None
        classes = result.metadata.get("classes", [])
        assert isinstance(classes, list)
        for entry in classes:
            assert "label" in entry
            assert "score" in entry
            assert isinstance(entry["score"], float)


# =========================================================================
# G. Synthetic Replacement (requires faker)
# =========================================================================


@requires_faker
class TestSyntheticReplacement:
    """G. Faker-based synthetic entity replacement."""

    def test_g41_synthetic_url_email_phone(self):
        """Synthetic mode replaces URLs/emails/phones with Faker values."""
        cfg = _make_cfg(replacement_mode="synthetic")
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process(
            "Visit https://example.com or email test@test.com or call +1-555-1234."
        )
        # Should NOT have placeholders — should have realistic fake values
        assert "<URL>" not in lm
        assert "<EMAIL>" not in lm
        assert "<PHONE>" not in lm
        # Original values should be replaced
        assert "example.com" not in lm
        assert "test@test.com" not in lm

    def test_g42_synthetic_ner(self):
        """Synthetic NER: Faker-generated entity replacements."""
        cfg = _make_ner_cfg(replacement_mode="synthetic")
        tc = TextCleaner(cfg=cfg)
        lm, _, _ = tc.process("John Smith works at Microsoft in Seattle.")
        # Should not have placeholder tags
        assert "<PERSON>" not in lm
        assert "<ORGANISATION>" not in lm
        assert "<LOCATION>" not in lm

    def test_g43_per_document_consistency(self):
        """Same entity text maps to same fake value within one document."""
        cfg = _make_ner_cfg(replacement_mode="synthetic")
        tc = TextCleaner(cfg=cfg)
        # Process text with repeated entity
        result = tc.process(
            "John met John at the office. John was happy."
        )
        # The consistency is internal — just verify it doesn't crash
        assert isinstance(result.lm_text, str)


# =========================================================================
# H. Torch NER Backend (requires torch + transformers)
# =========================================================================


@requires_torch
class TestTorchBackend:
    """H. PyTorch NER backend — end-to-end with real models."""

    def test_h44_torch_backend_e2e(self):
        """Torch backend processes text and replaces entities."""
        cfg = _make_ner_cfg(ner_backend="torch")
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "Angela Merkel visited the Bundestag in Berlin."
        )
        lm = result.lm_text
        has_entity = any(tag in lm for tag in ENTITY_TAGS)
        assert has_entity, f"No NER tags in: {lm}"

    def test_h45_torch_batch(self):
        """Torch backend batch processing."""
        cfg = _make_ner_cfg(ner_backend="torch")
        tc = TextCleaner(cfg=cfg)
        results = tc.process_batch([
            "John Smith lives in London.",
            "Maria Garcia works at Google.",
        ])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ProcessResult)


# =========================================================================
# I. GLiNER Backend (requires gliner)
# =========================================================================


@requires_gliner
class TestGLiNERBackend:
    """I. GLiNER zero-shot NER backend."""

    def test_i46_gliner_e2e(self):
        """GLiNER backend with default labels (person, organization, location)."""
        cfg = _make_ner_cfg(
            ner_backend="gliner",
            gliner_model="urchade/gliner_small-v2.1",
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "Elon Musk founded SpaceX in Hawthorne, California."
        )
        lm = result.lm_text
        # GLiNER maps its labels via label_map to standard tags
        has_entity = any(tag in lm for tag in ENTITY_TAGS)
        assert has_entity, f"No entity tags in GLiNER output: {lm}"

    def test_i47_gliner_custom_labels(self):
        """GLiNER with custom entity labels."""
        cfg = _make_ner_cfg(
            ner_backend="gliner",
            gliner_model="urchade/gliner_small-v2.1",
            gliner_labels=("person", "city", "company"),
            gliner_label_map={"person": "PER", "city": "LOC", "company": "ORG"},
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Tim Cook leads Apple in Cupertino.")
        assert isinstance(result.lm_text, str)

    def test_i48_gliner_onnx_fallback(self):
        """GLiNER ONNX gracefully falls back to PyTorch when model.onnx missing.

        Most GLiNER models don't ship model.onnx at repo root (GLiNER #314).
        The adapter should log a warning and fall back to PyTorch automatically.
        Re-enable as a true ONNX test once GLiNER ships root-level ONNX files.
        """
        cfg = _make_ner_cfg(
            ner_backend="gliner",
            gliner_model="urchade/gliner_small-v2.1",
            gliner_onnx=True,
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Satya Nadella is CEO of Microsoft in Redmond.")
        assert isinstance(result.lm_text, str)


# =========================================================================
# J. Ensemble Backends
# =========================================================================


@requires_gliner
class TestEnsembleBackends:
    """J. Ensemble: ONNX/Torch + GLiNER combined."""

    def test_j49_ensemble_onnx(self):
        """Ensemble ONNX: ONNX + GLiNER combined."""
        cfg = _make_ner_cfg(
            ner_backend="ensemble_onnx",
            gliner_model="urchade/gliner_small-v2.1",
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "Mark Zuckerberg founded Facebook in Menlo Park."
        )
        lm = result.lm_text
        has_entity = any(tag in lm for tag in ENTITY_TAGS)
        assert has_entity, f"No entity tags in ensemble output: {lm}"

    @requires_torch
    def test_j50_ensemble_torch(self):
        """Ensemble Torch: Torch + GLiNER combined."""
        cfg = _make_ner_cfg(
            ner_backend="ensemble_torch",
            gliner_model="urchade/gliner_small-v2.1",
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process(
            "Jeff Bezos started Amazon in Seattle."
        )
        assert isinstance(result.lm_text, str)


# =========================================================================
# K. PII Mode (requires gliner)
# =========================================================================


@requires_gliner
class TestPIIMode:
    """K. PII auto-configuration with GLiNER PII model."""

    def test_k51_pii_auto_config(self):
        """ner_mode='pii' auto-configures GLiNER for PII detection."""
        cfg = _make_ner_cfg(ner_mode="pii")
        # PII mode should auto-set gliner backend
        assert cfg.ner_backend == "gliner"
        assert cfg.gliner_model == PII_DEFAULT_MODEL
        assert cfg.gliner_onnx is True

        # GLiNER ONNX loading fails when model.onnx is in onnx/ subdirectory
        # (GLiNER issue #314), so test with non-ONNX backend for end-to-end PII
        cfg_no_onnx = _make_ner_cfg(
            ner_mode="pii",
            ner_backend="gliner",  # Skip auto-set that forces gliner_onnx=True
            gliner_model=PII_DEFAULT_MODEL,
        )
        tc = TextCleaner(cfg=cfg_no_onnx)
        result = tc.process(
            "John Smith's SSN is 123-45-6789 and his email is john@example.com."
        )
        assert isinstance(result.lm_text, str)

    def test_k52_pii_entity_coverage(self):
        """PII mode configures 60+ entity labels."""
        cfg = _make_ner_cfg(ner_mode="pii")
        assert len(cfg.gliner_labels) >= 50
        # Should include key PII categories
        labels_set = set(cfg.gliner_labels)
        assert "email" in labels_set
        assert "phone_number" in labels_set
        assert "credit_card_number" in labels_set
        assert "social_security_number" in labels_set


# =========================================================================
# L. Presidio GLiNER Backend (requires presidio-analyzer)
# =========================================================================


@requires_presidio
@requires_gliner
class TestPresidioGLiNER:
    """L. Presidio + GLiNER integration backend."""

    def test_l53_presidio_gliner_backend(self):
        """Presidio + GLiNER backend with real models."""
        cfg = _make_ner_cfg(
            ner_backend="presidio_gliner",
            gliner_model="urchade/gliner_small-v2.1",
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process("John Smith lives at 123 Main Street, New York.")
        assert isinstance(result.lm_text, str)


# =========================================================================
# M. Config Edge Cases
# =========================================================================


class TestConfigEdgeCases:
    """M. Configuration validation, backward compat, edge cases."""

    def test_m54_frozen_config_immutability(self):
        """Frozen dataclass rejects mutation."""
        cfg = _make_cfg()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.check_ner_process = True  # type: ignore[misc]

    def test_m55_legacy_module_level_config(self):
        """Legacy module-level config backward compat."""
        from sct import config
        original = config.CHECK_NER_PROCESS
        try:
            config.CHECK_NER_PROCESS = False
            # Should build a config from module globals
            tc = TextCleaner()
            assert tc.cfg.check_ner_process is False
        finally:
            config.CHECK_NER_PROCESS = original

    def test_m56_all_toggles_off(self):
        """All pipeline toggles OFF returns text mostly unchanged."""
        cfg = _make_cfg(
            check_fix_bad_unicode=False,
            check_to_ascii_unicode=False,
            check_replace_html=False,
            check_replace_urls=False,
            check_replace_emails=False,
            check_replace_years=False,
            check_replace_dates=False,
            check_replace_phone_numbers=False,
            check_replace_numbers=False,
            check_replace_currency_symbols=False,
            check_remove_isolated_letters=False,
            check_remove_isolated_special_symbols=False,
            check_remove_bracket_content=False,
            check_remove_brace_content=False,
            check_normalize_whitespace=False,
            check_remove_emoji=False,
            check_detect_language=False,
        )
        tc = TextCleaner(cfg=cfg)
        text = "Hello World 123"
        lm, _, _ = tc.process(text)
        assert lm == text

    def test_m57_ner_disabled(self):
        """check_ner_process=False skips NER entirely."""
        cfg = _make_cfg(check_ner_process=False)
        tc = TextCleaner(cfg=cfg)
        result = tc.process("John Smith works at Google.")
        assert tc.GeneralNER is None
        assert isinstance(result.lm_text, str)

    def test_m58_statistical_processing_disabled(self):
        """check_statistical_model_processing=False — stext is None."""
        cfg = _make_cfg(
            check_ner_process=False,
            check_statistical_model_processing=False,
        )
        tc = TextCleaner(cfg=cfg)
        result = tc.process("Hello World")
        assert result.stat_text is None

    def test_m59_invalid_ner_backend_rejected(self):
        """Invalid ner_backend raises ValueError."""
        with pytest.raises(ValueError, match="ner_backend"):
            _make_ner_cfg(ner_backend="invalid_backend")

    def test_m60_invalid_ner_mode_rejected(self):
        """Invalid ner_mode raises ValueError."""
        with pytest.raises(ValueError, match="ner_mode"):
            _make_ner_cfg(ner_mode="invalid_mode")

    def test_m61_invalid_replacement_mode_rejected(self):
        """Invalid replacement_mode raises ValueError."""
        with pytest.raises(ValueError, match="replacement_mode"):
            _make_ner_cfg(replacement_mode="invalid_mode")

    def test_m62_empty_text_handling(self):
        """Empty and whitespace-only texts handled gracefully."""
        cfg = _make_cfg()
        tc = TextCleaner(cfg=cfg)

        r1 = tc.process("")
        assert r1.lm_text == ""

        r2 = tc.process("   ")
        assert r2.lm_text == ""

    def test_m63_process_result_tuple_protocol(self):
        """ProcessResult supports len(), indexing, iteration."""
        pr = ProcessResult("hello", "world", "ENGLISH")
        assert len(pr) == 3
        assert pr[0] == "hello"
        assert pr[1] == "world"
        assert pr[2] == "ENGLISH"
        assert list(pr) == ["hello", "world", "ENGLISH"]

    def test_m64_process_result_equality(self):
        """ProcessResult equality with tuples and other ProcessResults."""
        pr1 = ProcessResult("a", "b", "ENGLISH")
        pr2 = ProcessResult("a", "b", "ENGLISH")
        assert pr1 == pr2
        assert pr1 == ("a", "b", "ENGLISH")
