"""ai4privacy multilingual categorical anonymiser: entry ``ai4privacy``
(ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii, ModernBERT-base, MIT).

transformers token classification, 20 OpenPII types in BIO tags, merged by the pipeline (``_hf_token``).
The card gives no context limit or chunking advice; ModernBERT takes 8192 positions, and windows of 512
tokens with a 128-token overlap keep the encoder inside the length it was fine-tuned on (OpenPII rows are
short). No vendor masking output. ``lang`` is ignored (the card takes none).

Config: ``threshold`` (default 0 = vendor), ``device``, ``threads``, ``max_len``, ``stride``, ``gpu_mem_gb``.
"""

import _hf_token
from _metadata import AI4PRIVACY as LABELS
from _metadata import AI4PRIVACY_MODEL as MODEL, AI4PRIVACY_REVISION as REVISION
from _protocol import serve

MAX_LEN, STRIDE = 512, 128

OUT_OF_SCOPE = ("AGE", "GENDER", "SEX", "TIME", "TITLE")


def load(config: dict):
    return _hf_token.load(MODEL, config.get("revision", REVISION), set(LABELS) | set(OUT_OF_SCOPE), config,
                          MAX_LEN, STRIDE)


if __name__ == "__main__":
    serve(load)
