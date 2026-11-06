"""DialogPII (Zenodo 20863452): synthetic dialogs (Gemini 2.5 Pro, human-curated) in 11 languages, no split.

A record is one dialog, turns joined as ``SPEAKER: text`` lines. The TTS + Whisper transcript version is
the language code plus ``-asr`` (``de-asr``), so it is scored apart from the curated text. The languages are
parallel localisations of the same 147 storylines: ``group`` is ``scenario:chat_number``, so every language and
both versions of one storyline land on the same side of the dev/test split. Cite Roller et al. 2026.
"""

import json
import zipfile
from pathlib import PurePosixPath

from benchmarks.datasets import md5_checked, sample_per_lang, url_file, valid_spans

NAME = "dialogpii"
HF_ID = "zenodo:20863452"
REVISION = "6b034ea96364a3f638133eefee3ee481"  # md5 of DialogPII.zip
LICENCE = "CC-BY-4.0 (cite Roller et al. 2026, doi 10.5281/zenodo.20863452)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"
URL = "https://zenodo.org/records/20863452/files/DialogPII.zip?download=1"
LANGS = {"SP": "es"}  # the set's own code for Spanish; the other ten are ISO codes in upper case


def dialog_record(turns: list[dict]) -> tuple[str, list[dict]]:
    """Join turns as ``SPEAKER: text`` lines; shift each turn's spans by its offset in the joined text."""
    parts, spans, offset = [], [], 0
    for turn in turns:
        prefix = f"{turn['speaker']}: "
        start = offset + len(prefix)
        parts.append(prefix + turn["text"])
        spans += [{"start": start + a["start"], "end": start + a["end"], "label": a["type"],
                   "form": a.get("text")}
                  for a in turn.get("annotations") or ()]
        offset = start + len(turn["text"]) + 1
    text = "\n".join(parts)
    return text, valid_spans(text, spans)


def _records(path):
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            file = PurePosixPath(name)
            # scenario from the file name (<LANG>_<scenario>[_transcript].json): the same key in every language
            scenario = file.stem.split("_", 1)[1].removesuffix("_transcript")
            code = LANGS.get(data["language"].upper(), data["language"].lower())
            asr = "transcripts" in file.parts
            lang = f"{code}-asr" if asr else code
            for dialog in data["dialogs"]:
                text, spans = dialog_record(dialog["dialog"]["turns"])
                yield {"id": f"{lang}:{scenario}:{dialog['chat_number']}", "lang": lang, "text": text,
                       "group": f"{scenario}:{dialog['chat_number']}", "spans": spans}


def load(n: int | None, seed: int) -> list[dict]:
    path = md5_checked(url_file(URL, f"dialogpii/{REVISION}/DialogPII.zip"), REVISION)
    return sample_per_lang(_records(path), n, seed)
