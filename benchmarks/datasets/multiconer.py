"""MultiCoNER v2: hi, bn, zh NER test sets (CoNLL BIO), fine classes grouped to PER/LOC/ORG.

zh tokens are characters, so zh text joins them with no separator. Id is the sentence uuid.
"""

from benchmarks.datasets import bio_to_spans, hf_file, sample_per_lang

NAME = "multiconer"
HF_ID = "MultiCoNER/multiconer_v2"
REVISION = "4be2d62c912977ee26ed14d2553a4fe17ca3d980"
LICENCE = "CC-BY-4.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"
FILES = {"hi": "HI-Hindi/hi_test.conll", "bn": "BN-Bangla/bn_test.conll", "zh": "ZH-Chinese/zh_test.conll"}


def _records():
    for lang, name in FILES.items():
        sep = "" if lang == "zh" else " "
        sentence_id, tokens, tags = None, [], []
        for line in [*hf_file(HF_ID, name, REVISION).read_text(encoding="utf-8").splitlines(), ""]:
            if line.startswith("# id "):
                sentence_id = line.split()[2]
            elif line.strip():
                parts = line.split()
                tokens.append(parts[0])
                tags.append(parts[-1])
            elif tokens:
                text, spans = bio_to_spans(tokens, tags, sep=sep)
                yield {"id": sentence_id, "lang": lang, "text": text, "spans": spans}
                tokens, tags = [], []


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(), n, seed)
