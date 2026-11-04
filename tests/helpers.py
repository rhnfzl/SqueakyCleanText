"""Helpers shared by test modules: fake NER detectors and tokenizers, and a GeneralNER built without models."""

import asyncio
import re
import threading
from dataclasses import replace
from types import SimpleNamespace

from presidio_anonymizer import AnonymizerEngine

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


class FakeGLiNER:
    """Finds every occurrence of NAMES in a chunk, like a perfect detector."""

    max_context_length = 100  # -> 90 tokens -> 180-char chunks
    labels = ["person"]
    label_map = {"person": "PER"}
    names = ("王小明", "राहुल", "ரவி")

    def __call__(self, texts, batch_size=1):
        out = []
        for text in texts:
            found = []
            for name in self.names:
                start = text.find(name)
                while start != -1:
                    found.append({"entity_group": "PER", "score": 0.99, "word": name,
                                  "start": start, "end": start + len(name)})
                    start = text.find(name, start + 1)
            out.append(found)
        return out


def gliner_ner(mode):
    ner = bare_ner()
    set_cfg(ner, ner_backend="gliner")
    ner._gliner_pipe = FakeGLiNER()
    set_cfg(ner, replacement_mode=mode)
    ner._synthetic_replacer = None
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner._ensemble_models = {}
    ner._ensemble_default_keys = ()
    ner.tokenizer = None
    return ner


def word_tokenizer(text, **kwargs):
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    return SimpleNamespace(input_ids=spans, offset_mapping=spans)


class FakeRunDetector(FakeGLiNER):
    """Tags capitalised words as PER; with min_words=2 only runs of two or more."""

    def __init__(self, min_words=1):
        self.pattern = re.compile(rf"[A-Z][a-z]+(?: [A-Z][a-z]+){{{min_words - 1},}}")

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99, "word": m.group(),
              "start": m.start(), "end": m.end()} for m in self.pattern.finditer(text)]
            for text in texts
        ]


def word_ner(overlap, detector, budget=12, mode="placeholder"):
    ner = gliner_ner(mode)
    ner._gliner_pipe = detector
    ner.tokenizer = word_tokenizer
    ner.min_token_length = budget
    set_cfg(ner, ner_chunk_overlap=overlap)
    ner._engine = AnonymizerEngine()
    return ner


def numbered_words(n, names):
    words = [f"w{i}" for i in range(n)]
    for index, name in names.items():
        words[index] = name
    return " ".join(words)


class WordCountTokenizer:
    """Word tokenizer counting *per_word* tokens per word, like a heavier vocabulary."""

    max_len_single_sentence = 1000

    def __init__(self, per_word):
        self.per_word = per_word

    def __call__(self, text, **kwargs):
        spans = word_tokenizer(text).input_ids
        return SimpleNamespace(input_ids=spans * self.per_word, offset_mapping=spans)


def both_paths(ner, text, privacy_policy=None, **kwargs):
    """Run *text* through the single and the batch path; one result per path run."""
    kwargs = {"positional_tags": ("PER", "LOC"), "ner_confidence_threshold": 0.85, **kwargs}
    ner._privacy_policy = privacy_policy
    single = ner.ner_process_detailed(text, **kwargs)
    batch = ner.ner_process_batch_detailed([text, text], [None, None], **kwargs)
    return [single] + batch


def finalize_spans(text, spans, **attrs):
    # One window over the whole text; spans are (start, end, group, score) from one model.

    ner = bare_ner()
    set_cfg(ner, replacement_mode=attrs.pop('mode', 'placeholder'), **attrs)
    items = [{'entity_group': g, 'score': s, 'start': a, 'end': b, 'model': 'm'} for a, b, g, s in spans]
    windows = [(text, 0, list(range(len(text) + 1)), 0, len(text))]
    return ner, windows, [items]


def client_call(server, coro):
    from fastmcp import Client

    async def run():
        async with Client(server) as client:
            return await coro(client)

    return asyncio.run(run())
