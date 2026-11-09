"""Piiranha v1 adapter: entry ``piiranha`` (iiiorg/piiranha-v1-detect-personal-information, mdeberta-v3-base).

Licence CC-BY-NC-ND-4.0: benchmark use only; never shipped, bundled or redistributed (no weights in git).
The card's path: transformers token classification, 17 types in an IO scheme (``I-`` tags only), so
consecutive same-type tokens form one span (``_hf_token``). The card's context is 256 tokens ("if your
text is longer than that, just split it up"): the pipeline windows each doc at 256 tokens with a 64-token
overlap. No vendor masking output (the runner applies the common mask). ``lang`` is ignored.

Config: ``threshold`` (drop spans below it; default 0 = vendor), ``device``, ``threads``, ``max_len``,
``stride``, ``gpu_mem_gb``.
"""

import _hf_token
from _metadata import PIIRANHA as LABELS
from _metadata import PIIRANHA_MODEL as MODEL, PIIRANHA_REVISION as REVISION
from _protocol import serve

MAX_LEN, STRIDE = 256, 64

OUT_OF_SCOPE = ("PASSWORD", "USERNAME")


def load(config: dict):
    return _hf_token.load(MODEL, config.get("revision", REVISION), set(LABELS) | set(OUT_OF_SCOPE), config,
                          MAX_LEN, STRIDE)


if __name__ == "__main__":
    serve(load)
