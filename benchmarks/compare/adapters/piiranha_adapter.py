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
from _protocol import serve

MODEL = "iiiorg/piiranha-v1-detect-personal-information"
REVISION = "255acde67a2f34cf452eb42e365b24d2957352fc"
MAX_LEN, STRIDE = 256, 64

# Tool label -> SCT type (PROTOCOL section 5); every one of the 17 types is in LABELS or OUT_OF_SCOPE.
LABELS = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON",
    "CITY": "LOCATION", "STREET": "LOCATION", "BUILDINGNUM": "LOCATION",
    "EMAIL": "EMAIL", "TELEPHONENUM": "PHONE", "DATEOFBIRTH": "DATE",
    **dict.fromkeys(("ACCOUNTNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM", "SOCIALNUM", "TAXNUM",
                     "ZIPCODE"), "NUMBER"),
}
OUT_OF_SCOPE = ("PASSWORD", "USERNAME")


def load(config: dict):
    return _hf_token.load(MODEL, config.get("revision", REVISION), set(LABELS) | set(OUT_OF_SCOPE), config,
                          MAX_LEN, STRIDE)


if __name__ == "__main__":
    serve(load)
