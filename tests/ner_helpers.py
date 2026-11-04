"""A GeneralNER built without loading models."""

from dataclasses import replace

from sct.config import GLINER_BACKENDS, TextCleanerConfig
from sct.utils.ner import GeneralNER


def bare_ner():
    """GeneralNER skipping __init__, with a default config and no collaborators."""
    ner = object.__new__(GeneralNER)
    ner.cfg = TextCleanerConfig()
    ner._allow_regex = None
    ner._synthetic_replacer = None
    ner._regex_markers = ()
    ner._privacy_policy = None
    ner._engine = None
    ner._tokenizers = {}
    return ner


def set_cfg(ner, **fields):
    """Change config fields of *ner*; a GLiNER backend gets the model name the config requires."""
    if fields.get("ner_backend") in GLINER_BACKENDS:
        fields.setdefault("gliner_model", "fake/gliner")
    ner.cfg = replace(ner.cfg, **fields)
