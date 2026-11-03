"""bardsai-v3 adapter: bardsai/eu-pii-anonimization-multilang-v3 via its bundled ``pii_detector.py``.

Runs the vendor quick start for ONNX/CPU: ``PIIDetector(repo, onnx="onnx/model_quantized.onnx")``
and ``predict(text)``, from a snapshot pinned at REVISION (the quick start itself does not pin).
Long docs are the detector's own business: 512-token windows with a 128-token overlap (native
track; the model allows 8192 positions but the decoder never uses more than 512). The decoder
has no batch API, so docs run one by one; ``lang`` is ignored (the quick start takes none).

Spans keep the tool's labels; ``LABELS`` below is the frozen label -> SCT type map for
``entries.py``. ``score`` is the decoder's own pooled probability of the span's type (mean
of its sub-token probabilities, B-/I- pooled, as the decoder ranks words); regex e-mails get
None (always kept). ``masked_text`` is ``anonymize()``'s format (``[LABEL]``), built from the
same spans so the model runs once.

Config: ``threshold`` (tuned-track knob: drop spans scoring below it; default 0 = vendor),
``onnx`` (default ``onnx/model_quantized.onnx``; ``onnx/model.onnx`` for FP32), ``threads``
(onnxruntime intra-op and torch threads). ``device`` is ignored: this entry is CPU only.
"""

import importlib.util
from pathlib import Path

from _protocol import serve

REPO = "bardsai/eu-pii-anonimization-multilang-v3"
REVISION = "091984b5fbce4c161abc4776ed0f4e3206e1b374"
CONTEXT = {"max_len": 512, "stride": 128}  # the detector's defaults, recorded for PROTOCOL section 11

# Tool label -> SCT type (PROTOCOL section 5 rules); every one of the 37 labels is in LABELS or OUT_OF_SCOPE.
LABELS = {
    "PERSON_NAME": "PERSON", "PERSON_ALIAS": "PERSON",
    "LOCATION": "LOCATION", "POSTAL_ADDRESS": "LOCATION", "GEO_LOCATION": "LOCATION",
    "ORGANIZATION_NAME": "ORGANISATION",
    "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "IP_ADDRESS": "IP",
    "URL": "URL", "IDENTIFYING_LINK": "URL",
    "DATE_TIME": "DATE", "DATE_OF_BIRTH": "DATE",
    **{t: "NUMBER" for t in ("PERSON_IDENTIFIER", "ORGANIZATION_IDENTIFIER", "BANK_ACCOUNT_IDENTIFIER",
                             "ACCOUNT_IDENTIFIER", "PAYMENT_CARD", "PAYMENT_CARD_SECURITY",
                             "DOCUMENT_IDENTIFIER", "VEHICLE_IDENTIFIER", "DEVICE_IDENTIFIER")},
}
OUT_OF_SCOPE = ("AUTH_SECRET", "CONTACT_HANDLE", "DOCUMENT_REFERENCE", "FINANCIAL_AMOUNT", "PERSON_ATTRIBUTE",
                "PERSON_ROLE_OR_TITLE", "PROPER_NAME",
                # GDPR art. 9
                "BIOMETRIC_DATA", "CRIMINAL_OFFENCE_DATA", "ETHNIC_ORIGIN", "HEALTH_DATA", "POLITICAL_OPINION",
                "RELIGION_OR_BELIEF", "SEXUAL_ORIENTATION", "TRADE_UNION_MEMBERSHIP")
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "pii_detector.py"]


def load(config: dict):
    from huggingface_hub import snapshot_download

    onnx = config.get("onnx", "onnx/model_quantized.onnx")
    path = snapshot_download(REPO, revision=REVISION, allow_patterns=[*FILES, onnx])
    if config.get("threads"):
        import functools

        import onnxruntime
        import torch

        torch.set_num_threads(int(config["threads"]))
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = int(config["threads"])
        # the detector builds its own session with no options; pass ours through
        onnxruntime.InferenceSession = functools.partial(onnxruntime.InferenceSession, sess_options=opts)
    spec = importlib.util.spec_from_file_location("pii_detector", Path(path) / "pii_detector.py")
    pii_detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pii_detector)
    det = pii_detector.PIIDetector(path, onnx=onnx, **CONTEXT)
    if set(det.types) != set(LABELS) | set(OUT_OF_SCOPE):
        raise RuntimeError(f"bardsai labels changed: {sorted(set(det.types) ^ (set(LABELS) | set(OUT_OF_SCOPE)))}")
    threshold = float(config.get("threshold") or 0.0)

    probs = {}
    char_probs = det._char_probs

    def keep_probs(text):  # reuse the decoder's own probabilities for span scores (no second pass)
        probs["last"] = char_probs(text)
        return probs["last"]

    det._char_probs = keep_probs
    emails = pii_detector._EMAIL

    def one(text: str) -> dict:
        probs.clear()
        found = det.predict(text)
        spans = []
        regex_emails = {(m.start(), m.end()) for m in emails.finditer(text)}
        for s in found:
            score = None
            if not (s["label"] == "EMAIL_ADDRESS" and (s["start"], s["end"]) in regex_emails):
                acc, cnt, _ = probs["last"]
                sel = cnt[s["start"]:s["end"]] > 0  # token-start chars, as the decoder pools words
                k = det.types.index(s["label"])
                score = float(acc[s["start"]:s["end"]][sel][:, k].sum() / cnt[s["start"]:s["end"]][sel].sum())
                if score < threshold:
                    continue
            spans.append({"start": s["start"], "end": s["end"], "label": s["label"], "score": score})
        masked, last = [], 0
        for s in spans:
            masked += [text[last:s["start"]], f"[{s['label']}]"]
            last = s["end"]
        return {"spans": spans, "masked_text": "".join([*masked, text[last:]])}

    return lambda docs: [one(d["text"]) for d in docs]


if __name__ == "__main__":
    serve(load)
