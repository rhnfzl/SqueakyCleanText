"""ai4privacy multilingual categorical anonymiser: entry ``ai4privacy``
(ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii, ModernBERT-base, MIT).

transformers token classification, 20 OpenPII types in BIO tags, merged by the pipeline (``_hf_token``).
The card gives no context limit or chunking advice; ModernBERT takes 8192 positions, and windows of 512
tokens with a 128-token overlap keep the encoder inside the length it was fine-tuned on (OpenPII rows are
short). No vendor masking output. ``lang`` is ignored (the card takes none).

Config: ``threshold`` (default 0 = vendor), ``device``, ``threads``, ``max_len``, ``stride``, ``gpu_mem_gb``.
"""

import _hf_token
from _protocol import serve

MODEL = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
REVISION = "83ef30d5e7c9d113ad80ce745b564cdd2320c5d5"
MAX_LEN, STRIDE = 512, 128

# Same rules as piiranha (OpenPII names): every one of the 20 types is in LABELS or OUT_OF_SCOPE.
LABELS = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON",
    "CITY": "LOCATION", "STREET": "LOCATION", "BUILDINGNUM": "LOCATION",
    "EMAIL": "EMAIL", "TELEPHONENUM": "PHONE", "DATE": "DATE",
    **dict.fromkeys(("CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM", "PASSPORTNUM", "SOCIALNUM", "TAXNUM",
                     "ZIPCODE"), "NUMBER"),
}
OUT_OF_SCOPE = ("AGE", "GENDER", "SEX", "TIME", "TITLE")


def load(config: dict):
    return _hf_token.load(MODEL, config.get("revision", REVISION), set(LABELS) | set(OUT_OF_SCOPE), config,
                          MAX_LEN, STRIDE)


if __name__ == "__main__":
    serve(load)
