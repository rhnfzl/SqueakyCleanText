"""Test adapter (entry ``dummy``): no model, standard library only.

By default it marks e-mail addresses (``EMAIL``, score 0.9) and digit runs (``NUMBER``,
score 0.6). Config for tests, all keyed by doc id: ``spans`` (fixed predictions),
``sleep`` (seconds), ``raise`` (ids that raise), ``crash`` (ids that kill the process),
``crash_once`` with ``state_dir`` (crash the first time only), ``bad`` (ids that return an
out-of-range span).
"""

import os
import re
import time
from pathlib import Path

from _protocol import serve

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
DIGITS = re.compile(r"\d+")


def load(config: dict):
    def one(doc: dict) -> dict:
        doc_id = doc["id"]
        time.sleep(config.get("sleep", {}).get(doc_id, 0))
        if doc_id in config.get("crash", ()):
            os._exit(3)
        if doc_id in config.get("crash_once", ()):
            marker = Path(config["state_dir"]) / f"crashed-{doc_id}"
            if not marker.exists():
                marker.touch()
                os._exit(3)
        if doc_id in config.get("raise", ()):
            raise RuntimeError("dummy failure")
        if doc_id in config.get("bad", ()):
            return {"spans": [{"start": 0, "end": len(doc["text"]) + 5, "label": "NUMBER"}]}
        if doc_id in config.get("spans", {}):
            return {"spans": config["spans"][doc_id]}
        spans = [{"start": m.start(), "end": m.end(), "label": "EMAIL", "score": 0.9} for m in EMAIL.finditer(doc["text"])]
        spans += [{"start": m.start(), "end": m.end(), "label": "NUMBER", "score": 0.6} for m in DIGITS.finditer(doc["text"])
                  if not any(s["start"] <= m.start() < s["end"] for s in spans)]
        return {"spans": spans}

    return lambda docs: [one(doc) for doc in docs]


if __name__ == "__main__":
    serve(load)
