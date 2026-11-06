"""Presidio FULL adapter: entry ``presidio-full`` (separate from the quick-start ``presidio`` entry).

Everything Presidio ships, configured through ``presidio_adapter.load`` (``entries.RUNS`` holds the config):
one spaCy engine per language (``entries.PRESIDIO_MODELS``: the ``*_core_news_lg`` pack where one exists,
else ``xx_ent_wiki_sm``) routed by the gold language code, ``PhoneRecognizer`` with one region per scored
language (``entries.PHONE_REGIONS``), every predefined pattern recognizer of the installed release in every
language (``enable: "all"``, including the off-by-default country ones), and spaCy ORGANIZATION kept
(``organizations: true``; the vendor's ``default.yaml`` lists it in ``labels_to_ignore``). No GLiNER,
transformers or Stanza recognizer: those would make it another model's entry (PROTOCOL section 3).
CPU only, like ``presidio``.
"""

from _protocol import serve
from presidio_adapter import load

if __name__ == "__main__":
    serve(load)
