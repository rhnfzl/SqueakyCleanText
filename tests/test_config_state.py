import copy
import dataclasses
import pickle
import warnings
from types import MappingProxyType

import pytest

from sct import TextCleaner, TextCleanerConfig


def _legacy_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return TextCleanerConfig(ner_models_list=(
            "en-model", "nl-model", "de-model", "es-model", "multi-model",
        ))


CONFIGS = {
    "default": TextCleanerConfig,
    "legacy": _legacy_config,
    "custom": lambda: TextCleanerConfig(
        ner_mode="pii",
        ner_models={"ENGLISH": "en-model", "MULTILINGUAL": "multi-model"},
        gliner_label_descriptions={"person": "a person's name"},
        custom_stopwords={"en": frozenset({"foo"})},
        custom_month_names={"en": ["jan"]},
        language=("en", "nl"),
        check_ner_process=False,
    ),
}


def _fields(cfg):
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("copier", [
    lambda c: pickle.loads(pickle.dumps(c)),  # noqa: S301 - trusted round trip
    copy.deepcopy,
    dataclasses.replace,
])
def test_config_copies_keep_exact_state(name, copier):
    cfg = CONFIGS[name]()

    restored = copier(cfg)

    assert restored == cfg
    assert _fields(restored) == _fields(cfg)
    for field, value in _fields(cfg).items():
        assert type(getattr(restored, field)) is type(value), field
    with pytest.raises(TypeError):
        restored.ner_models["FRENCH"] = "x"


def test_default_config_round_trip_does_not_grow_models():
    cfg = pickle.loads(pickle.dumps(TextCleanerConfig()))  # noqa: S301 - trusted round trip
    replaced = dataclasses.replace(TextCleanerConfig())

    for c in (cfg, replaced):
        assert len(c.ner_models) == 5
        assert len(c.supported_languages) == 4
        assert isinstance(c.ner_models, MappingProxyType)


def test_partial_ner_models_dict_still_fills_all_defaults():
    cfg = TextCleanerConfig(ner_models={
        "ENGLISH": "en-model", "MULTILINGUAL": "multi-model",
    })

    assert len(cfg.ner_models) == 8
    assert dataclasses.replace(cfg) == cfg


def test_unpickled_config_gives_identical_output():
    cfg = TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    )
    text = "Mail maria@example.com on 2023-01-01 or call +1-202-555-0147"

    restored = pickle.loads(pickle.dumps(cfg))  # noqa: S301 - trusted round trip

    assert TextCleaner(cfg=restored).process(text) == (
        TextCleaner(cfg=cfg).process(text)
    )


def test_new_ip_fields_do_not_shift_positional_arguments():
    # 0.7.0 order: detect_language, fix_bad_unicode, to_ascii_unicode, html, urls,
    # emails, years. The 7th positional argument must still be check_replace_years.
    cfg = TextCleanerConfig(True, True, True, True, True, True, False)

    assert cfg.check_replace_years is False
    assert cfg.check_replace_ips is True


def test_partial_ner_models_same_for_any_mapping_type():
    models = {"ENGLISH": "en-model", "MULTILINGUAL": "multi-model"}

    as_dict = TextCleanerConfig(ner_models=models)
    as_proxy = TextCleanerConfig(ner_models=MappingProxyType(models))

    assert as_proxy.ner_models == as_dict.ner_models
    assert as_proxy.supported_languages == as_dict.supported_languages
