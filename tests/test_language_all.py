"""All-75 language detection, confidence, pins and bring-your-own NER model (0.8.0)."""

import threading

import pytest
from lingua import Language, LanguageDetectorBuilder

import sct.config as config_module
from sct import TextCleaner, TextCleanerConfig
from sct.utils.ner import GeneralNER
from tests.conftest import TEST_NER_MODELS

# EN/NL/DE/ES labels must not change when detection widens to 75 languages (decision 4).
CORE_SENTENCES = [
    "Hello world", "Hello World and test items", "This is English text", "John Smith works at Google.",
    "This is the quick brown fox and the lazy dog.", "Satya Nadella is CEO of Microsoft in Redmond.",
    "Hello from Amsterdam in the Netherlands.", "This is clearly English text about the weather",
    "Please send the invoice to our office before Friday.",
    "The meeting was moved to next week because of the holiday.",
    "We are looking for an experienced software engineer to join our team in London.",
    "De snelle bruine vos springt over de luie hond.", "Hallo, hoe gaat het met je vandaag?",
    "Ik woon in Amsterdam en werk bij een bank.",
    "Wij zoeken een ervaren ontwikkelaar voor ons team in Utrecht.",
    "Het weer is vandaag erg mooi.", "De vergadering is verplaatst naar volgende week dinsdag.",
    "Bedankt voor je bericht, ik neem zo snel mogelijk contact op.", "Goedemorgen allemaal", "Dank je wel",
    "Hallo Welt", "Guten Morgen, wie geht es Ihnen heute?", "Ich wohne in Berlin und arbeite bei einer Bank.",
    "Wir suchen einen erfahrenen Entwickler für unser Team in München.", "Das Wetter ist heute sehr schön.",
    "Die Besprechung wurde auf nächsten Dienstag verschoben.", "Vielen Dank für Ihre Nachricht.",
    "Bitte senden Sie die Rechnung bis Freitag.", "Danke schön",
    "Hola, ¿cómo estás hoy?", "Vivo en Madrid y trabajo en un banco.",
    "Buscamos un desarrollador con experiencia para nuestro equipo en Barcelona.",
    "El tiempo hoy es muy bueno.", "La reunión se ha aplazado hasta el próximo martes.",
    "Gracias por tu mensaje, te responderé pronto.", "Por favor envíe la factura antes del viernes.",
    "Buenos días a todos", "Muchas gracias",
    "Angela Merkel war Bundeskanzlerin von Deutschland.", "Mark Rutte was premier van Nederland.",
    "Pedro Sánchez es el presidente del Gobierno de España.",
]

OTHER_SCRIPTS = {
    "HINDI": "नमस्ते, आप कैसे हैं? मैं दिल्ली में रहता हूँ और एक बैंक में काम करता हूँ।",
    "THAI": "สวัสดีครับ ผมอาศัยอยู่ในกรุงเทพและทำงานที่ธนาคาร",
    "ARABIC": "مرحبا، كيف حالك؟ أنا أعيش في القاهرة وأعمل في بنك.",
    "CHINESE": "你好，我住在北京，在一家银行工作。",
    "JAPANESE": "こんにちは、私は東京に住んでいて、銀行で働いています。",
    "BENGALI": "নমস্কার, আমি ঢাকায় থাকি এবং একটি ব্যাংকে কাজ করি।",
    "TAMIL": "வணக்கம், நான் சென்னையில் வசிக்கிறேன், ஒரு வங்கியில் வேலை செய்கிறேன்.",
}


def _cfg(**overrides):
    return TextCleanerConfig(check_ner_process=False, **overrides)


def _four_language_labels():
    four = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.DUTCH, Language.GERMAN, Language.SPANISH,
    ).build()
    labels = []
    for text in CORE_SENTENCES:
        detected = four.detect_language_of(text)
        labels.append(detected.name if detected else None)
    return labels


def test_core_labels_unchanged_for_bare_config():
    cleaner = TextCleaner(cfg=_cfg())
    assert [cleaner.process(t).language for t in CORE_SENTENCES] == _four_language_labels()


def test_core_labels_unchanged_for_module_globals(monkeypatch):
    monkeypatch.setattr(config_module, "CHECK_NER_PROCESS", False)
    cleaner = TextCleaner()
    assert cleaner.cfg.detect_all_languages is True
    assert [cleaner.process(t).language for t in CORE_SENTENCES] == _four_language_labels()


@pytest.mark.parametrize("expected,text", sorted(OTHER_SCRIPTS.items()))
def test_other_scripts_detected(expected, text):
    result = TextCleaner(cfg=_cfg()).process(text)
    assert result.language == expected
    assert result.metadata["language_source"] == "detected"
    assert 0.0 < result.metadata["language_confidence"] <= 1.0


def test_detect_all_languages_false_keeps_old_labels():
    cleaner = TextCleaner(cfg=_cfg(detect_all_languages=False))
    assert [cleaner.process(t).language for t in CORE_SENTENCES] == _four_language_labels()
    assert cleaner.process(OTHER_SCRIPTS["HINDI"]).language != "HINDI"


def test_module_global_turns_all_languages_off(monkeypatch):
    monkeypatch.setattr(config_module, "CHECK_NER_PROCESS", False)
    monkeypatch.setattr(config_module, "DETECT_ALL_LANGUAGES", False)
    monkeypatch.setattr(config_module, "LANGUAGE_CONFIDENCE_WARNING", 0.9)
    cfg = config_module._config_from_module_globals()
    assert cfg.detect_all_languages is False
    assert cfg.language_confidence_warning == 0.9


def test_any_lingua_language_can_be_pinned():
    result = TextCleaner(cfg=_cfg(language="hi")).process("Hello world")
    assert result.language == "HINDI"
    assert result.metadata["language_confidence"] == 1.0
    assert result.metadata["language_source"] == "pinned"
    assert result.metadata["warnings"] == []


def test_tuple_with_non_default_language():
    cleaner = TextCleaner(cfg=_cfg(language=("en", "th")))
    assert cleaner.process(OTHER_SCRIPTS["THAI"]).language == "THAI"
    fallback = cleaner.process(OTHER_SCRIPTS["HINDI"])
    assert fallback.language == "ENGLISH"
    assert fallback.metadata["language_source"] == "fallback"
    assert fallback.metadata["warnings"] == ["language_fallback"]
    assert 0.0 <= fallback.metadata["language_confidence"] <= 1.0


def test_low_confidence_warning_threshold():
    text = "Hello world"
    low = TextCleaner(cfg=_cfg()).process(text)
    assert low.metadata["language_confidence"] < 0.5
    assert low.metadata["warnings"] == ["language_low_confidence"]
    off = TextCleaner(cfg=_cfg(language_confidence_warning=0.0)).process(text)
    assert off.language == low.language
    assert off.metadata["warnings"] == []


def test_unknown_language_warning():
    result = TextCleaner(cfg=_cfg()).process("12345 !!!")
    assert result.language is None
    assert result.metadata["language_confidence"] is None
    assert result.metadata["language_source"] == "none"
    assert result.metadata["warnings"] == ["language_unknown"]


def test_no_detection_no_warning():
    cleaner = TextCleaner(cfg=_cfg(check_detect_language=False, check_remove_stopwords=False))
    result = cleaner.process("Hello world")
    assert result.language is None
    assert result.metadata["language_source"] == "none"
    assert result.metadata["warnings"] == []


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_confidence_warning_validated(value):
    with pytest.raises(ValueError, match="language_confidence_warning"):
        _cfg(language_confidence_warning=value)


def test_bring_your_own_model_routes_hindi(monkeypatch):
    model = TEST_NER_MODELS["ENGLISH"]
    cfg = TextCleanerConfig(
        check_ner_process=True,
        ner_models={**TEST_NER_MODELS, "HINDI": model},
        ner_model_revisions={},
    )
    cleaner = TextCleaner(cfg=cfg)
    ner = cleaner.GeneralNER
    assert ner._get_ensemble_keys("HINDI") == ("HINDI",)
    assert ner._get_ensemble_keys("FRENCH") == ner._ensemble_default_keys
    loaded = []
    original = ner._ensure_loaded
    monkeypatch.setattr(ner, "_ensure_loaded", lambda key: (loaded.append(key), original(key))[1])
    result = cleaner.process(OTHER_SCRIPTS["HINDI"])
    assert result.language == "HINDI"
    assert "HINDI" in ner._pipelines
    assert "HINDI" in loaded and "MULTILINGUAL" not in loaded
    assert "ner_fallback_models" not in result.metadata["warnings"]
    french = cleaner.process("Bonjour, je m'appelle Marie et j'habite à Paris depuis dix ans.")
    assert french.language == "FRENCH"
    assert "ner_fallback_models" in french.metadata["warnings"]


def test_builtin_fallback_keys_keep_default_route():
    ner = object.__new__(GeneralNER)
    ner._ner_backend = "onnx"
    ner._ensemble_models = config_module.DEFAULT_NER_ENSEMBLE
    ner._ensemble_default_keys = config_module.NER_ENSEMBLE_DEFAULT_KEYS
    ner._model_names = dict(config_module.DEFAULT_NER_MODELS)
    for language in ("FRENCH", "PORTUGUESE", "ITALIAN", "MULTILINGUAL", "HINDI"):
        assert ner._get_ensemble_keys(language) == config_module.NER_ENSEMBLE_DEFAULT_KEYS
        assert ner.uses_fallback_models(language)
    assert ner._get_ensemble_keys("DUTCH") == ("DUTCH", "ENGLISH", "MULTILINGUAL")
    assert not ner.uses_fallback_models("DUTCH")
    ner._model_names["FRENCH"] = "my/french-ner"
    assert ner._get_ensemble_keys("FRENCH") == ("FRENCH",)


class _FakeGLiNER:
    max_context_length = 512
    labels = ["person"]
    label_map = {"person": "PER"}

    def __call__(self, texts, batch_size=1):
        return [[] for _ in texts]


def test_gliner_hindi_pin_runs(monkeypatch):
    def fake_ner(**kwargs):
        ner = object.__new__(GeneralNER)
        ner._ner_backend = "gliner"
        ner._ner_batch_size = 1
        ner._ner_chunk_overlap = 0
        ner._gliner_pipe = _FakeGLiNER()
        ner._replacement_mode = "placeholder"
        ner._synthetic_replacer = None
        ner._inference_locks = {}
        ner._locks_lock = threading.Lock()
        ner._ensemble_models = {}
        ner._ensemble_default_keys = ()
        ner._model_names = {}
        ner._provenance = {}
        ner.tokenizer = None
        return ner

    monkeypatch.setattr("sct.sct.ner.GeneralNER", fake_ner)
    cfg = TextCleanerConfig(
        check_ner_process=True, ner_backend="gliner", gliner_model="fake/gliner", language="hi",
    )
    result = TextCleaner(cfg=cfg).process(OTHER_SCRIPTS["HINDI"])
    assert result.language == "HINDI"
    assert result.metadata["warnings"] == []
