"""CARMEN-I (PhysioNet, doi 10.13026/01e6-yq19, v1.0.2): real Spanish/Catalan clinical records. Gated.

Needs PhysioNet credentialing, CITI training, the signed DUA and the authors' per-project approval. Licence
PhysioNet Credentialed Health Data 1.5.0: aggregate scores only, local tools only (no hosted APIs), no samples
or per-example outputs. The loader (``replaced/`` text + ``ann/replaced/anon/`` spans) and its label map are
written once access is granted, from the release's annotation.conf.
"""

NAME = "carmen"
HF_ID = "physionet:carmen-i"
REVISION = "1.0.2"
LICENCE = "PhysioNet Contributor Review Health Data License 1.5.0 + DUA (aggregate scores only)"
PUBLISH = "scores-only"
DELETE_PREDICTIONS = True  # licence: no prediction caches kept after scoring (runner)
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    raise RuntimeError("CARMEN-I needs PhysioNet access (credentialing, CITI training, signed DUA and the authors' "
                       "approval); the loader is written once the data is available")
