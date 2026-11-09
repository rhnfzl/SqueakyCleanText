"""knowledgator/gliner-pii-base-v1.0 adapter (entry ``gliner-knowledgator``) on the ``gliner`` package.

The same checkpoint as ``sct-gliner``, run standalone, the model card's quick start: ``GLiNER.from_pretrained``
at the pinned revision and ``threshold=0.3`` (every card example), through ``gliner_nvidia_adapter.load``
(same library calls: ``inference``, flat NER, fp16 on cuda, fp32 on cpu, the torch memory cap). Prompts are
the fixed ``GLINER_PROMPTS`` list; the card has no label list of its own, so there is no ``card`` mode.
Long docs, the vendor's way: gliner truncates at ``max_len`` = 2048 words (``gliner_config.json``).

Config: as ``gliner_nvidia_adapter`` (``threshold``, ``device``, ``threads``, ``dtype``, ``revision``).
"""

from _metadata import GLINER_KNOWLEDGATOR_MODEL as MODEL, GLINER_KNOWLEDGATOR_REVISION as REVISION
from _protocol import serve
from gliner_nvidia_adapter import load as gliner_load

THRESHOLD = 0.3


def load(config: dict):
    if config.get("labels", "common") != "common":
        raise ValueError("gliner-knowledgator has no card label list: labels must be 'common'")
    return gliner_load({"threshold": THRESHOLD, **config, "model": MODEL, "revision": config.get("revision", REVISION)})


if __name__ == "__main__":
    serve(load)
