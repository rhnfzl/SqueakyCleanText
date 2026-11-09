"""OpenMed/privacy-filter-multilingual-v2 adapter: entry ``openmed`` (an openai/privacy-filter fine-tune).

Same architecture and BIOES head as OPF (54 types x B/I/E/S + O), so it runs through
``opf_adapter.make_predict``: ``AutoModelForTokenClassification`` in bf16, log-softmax, the constrained
Viterbi decoder, whitespace-trimmed char spans, 4096-token windows, ``flex_attention`` on cuda. The card
recommends the ``openmed`` package, which "runs the model's own BIOES Viterbi decoder"; the repo ships no
``viterbi_calibration.json``, so every transition bias is 0 (the same decoder at its neutral operating point).
``lang`` is ignored.

``LABELS`` is the frozen type -> SCT map (PROTOCOL section 5): name parts -> PERSON, address parts and GPS
-> LOCATION, postcodes and IDs / accounts / cards / device and vehicle IDs -> NUMBER (round-1 rules);
crypto addresses, money, attributes, job, credentials and TIME out of scope. ``load`` refuses a checkpoint
whose types differ. ``score`` and ``masked_text`` as in ``opf_adapter``.

Config: ``device``, ``threads``, ``n_ctx``, ``attn``, ``revision``, ``gpu_mem_gb``; ``biases`` (OPF's bias names).
"""

from _metadata import OPENMED as LABELS
from _metadata import OPENMED_MODEL as MODEL, OPENMED_REVISION as REVISION
from _protocol import serve
from opf_adapter import BIAS_KEYS, make_predict, merge_biases


OUT_OF_SCOPE = ("ACCOUNTNAME", "AGE", "AMOUNT", "BITCOINADDRESS", "CREDITCARDISSUER", "CURRENCY", "CURRENCYCODE",
                "CURRENCYNAME", "CURRENCYSYMBOL", "ETHEREUMADDRESS", "EYECOLOR", "GENDER", "HEIGHT", "JOBDEPARTMENT",
                "JOBTITLE", "LITECOINADDRESS", "OCCUPATION", "ORDINALDIRECTION", "PASSWORD", "PREFIX", "SEX", "TIME",
                "USERAGENT", "USERNAME")


def check_labels(bioes: list[str]) -> None:
    found = {label[2:] for label in bioes if label != "O"}
    expected = set(LABELS) | set(OUT_OF_SCOPE)
    if found != expected:
        raise RuntimeError(f"OpenMed labels changed: {sorted(found ^ expected)}")


def load(config: dict):
    predict, labels = make_predict(MODEL, config.get("revision", REVISION),
                                   merge_biases(dict.fromkeys(BIAS_KEYS, 0.0), config), config)
    check_labels(labels)
    return predict


if __name__ == "__main__":
    serve(load)
