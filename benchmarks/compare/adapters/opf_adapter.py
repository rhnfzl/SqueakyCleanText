"""OPF adapter: entry ``opf`` (openai/privacy-filter, pinned revision, on ``transformers``).

The model runs through ``AutoModelForTokenClassification``; its BIOES token logits are
decoded the vendor's way, ported from the ``opf`` package (github.com/openai/privacy-filter,
``opf/_core/runtime.py`` + ``decoding.py``): log-softmax, constrained Viterbi with the six
transition biases of the checkpoint's ``viterbi_calibration.json`` default operating point
(all 0.0 at the pinned revision), whitespace-trimmed char spans, bf16 weights. The plain HF
``pipeline(aggregation_strategy="simple")`` is not used: it only strips ``B-``/``I-``, so
``E-``/``S-`` tokens become separate spans with prefixed labels.

Long docs: the vendor splits the token sequence into non-overlapping windows of ``n_ctx``
tokens (4096 on CPU, the checkpoint's ``default_n_ctx`` 128000 on GPU). HF's eager attention
builds a full T x T mask, so 128k windows cannot fit a 5-8 GB GPU; this adapter uses the
vendor's 4096 on every device. Attention is banded (+-128 tokens), so only tokens within 128 of a
window edge lose context. On cuda the model runs ``flex_attention`` (only the band is computed): eager
attention's full T x T scores do not fit a 4096-token window under the 5 GB ``gpu_mem_gb`` cap (smoke run,
2026-11-03: 6 of 20 RedactionBench dev docs hit CUDA OOM; peak 2.8 GB with flex at 4096 tokens). Flex and
eager agree on 99.99% of token argmaxes (bf16 rounding); CPU keeps eager.

Config (default track = ``{}``): ``device`` (cpu|cuda, default cuda when available),
``threads``, ``n_ctx`` (window tokens, default 4096), ``attn`` (attention implementation).
Tuned-track knob (PROTOCOL section 3,
"OPF's documented decoder-bias preset"): ``biases`` = {bias name: float}, merged over the
calibration file; keys as in ``BIAS_KEYS``. ``revision`` pins the checkpoint.

``score`` = exp(mean token log-prob of the decoded labels): OPF itself gives no span score;
it is reported for threshold curves only and filters nothing. ``masked_text`` uses the
vendor's ``<LABEL>`` placeholders.
"""

import json
import math

from _protocol import cap_gpu, serve

MODEL = "openai/privacy-filter"
REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
BIAS_KEYS = ("transition_bias_background_stay", "transition_bias_background_to_start",
             "transition_bias_inside_to_continue", "transition_bias_inside_to_end",
             "transition_bias_end_to_background", "transition_bias_end_to_start")
N_CTX = 4096
NEG_INF = -1e9


def tag_of(label: str) -> tuple[str, str | None]:
    """``B-private_person`` -> (``B``, ``private_person``); ``O`` -> (``O``, None)."""
    return ("O", None) if label == "O" else (label[0], label[2:])


def transition_scores(labels: list[str], biases: dict):
    """Start, end and transition scores of the vendor's ViterbiCRFDecoder (decoding.py)."""
    import torch

    n = len(labels)
    tags = [tag_of(label) for label in labels]
    start, end = torch.full((n,), NEG_INF), torch.full((n,), NEG_INF)
    trans = torch.full((n, n), NEG_INF)
    bias = {k.removeprefix("transition_bias_"): v for k, v in biases.items()}
    for i, (tag, span) in enumerate(tags):
        if tag in "OBS":
            start[i] = 0.0
        if tag in "OES":
            end[i] = 0.0
        for j, (next_tag, next_span) in enumerate(tags):
            if tag in "OES":  # background or a closed span: stay out, or open a new span
                if next_tag == "O":
                    trans[i, j] = bias["background_stay" if tag == "O" else "end_to_background"]
                elif next_tag in "BS":
                    trans[i, j] = bias["background_to_start" if tag == "O" else "end_to_start"]
            elif next_span == span and next_tag in "IE":  # inside a span: continue or close the same label
                trans[i, j] = bias["inside_to_continue" if next_tag == "I" else "inside_to_end"]
    return start, end, trans


def viterbi(logp, start, end, trans) -> list[int]:
    """Best label path for one ``[T, labels]`` log-prob tensor (the vendor's ``decode``)."""
    import torch

    scores = logp[0] + start
    back = []
    for t in range(1, logp.shape[0]):
        best, path = (scores.unsqueeze(1) + trans).max(dim=0)
        scores = best + logp[t]
        back.append(path)
    if not torch.isfinite(scores).any():
        return logp.argmax(dim=1).tolist()
    label = int((scores + end).argmax())
    out = [label]
    for path in reversed(back):
        label = int(path[label])
        out.append(label)
    return out[::-1]


def token_spans(path: list[int], labels: list[str]) -> list[tuple[str, int, int]]:
    """(label, first token, end token) for each B..E / S run of a valid BIOES path."""
    spans, begin = [], None
    for t, label_id in enumerate(path):
        tag, span = tag_of(labels[label_id])
        if tag in "BS":
            begin = t
        if tag in "ES" and begin is not None:
            spans.append((span, begin, t + 1))
            begin = None
    return spans


def mask(text: str, spans: list[dict]) -> str:
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        text = text[:span["start"]] + f"<{span['label'].upper()}>" + text[span["end"]:]
    return text


def merge_biases(biases: dict, config: dict) -> dict:
    unknown = set(config.get("biases", {})) - set(BIAS_KEYS)
    if unknown:
        raise ValueError(f"unknown OPF biases: {sorted(unknown)}")
    return {**biases, **{k: float(v) for k, v in config.get("biases", {}).items()}}


def load(config: dict):
    from huggingface_hub import hf_hub_download

    revision = config.get("revision", REVISION)
    with open(hf_hub_download(MODEL, "viterbi_calibration.json", revision=revision), encoding="utf-8") as f:
        biases = dict(json.load(f)["operating_points"]["default"]["biases"])
    return make_predict(MODEL, revision, merge_biases(biases, config), config)[0]


def make_predict(name: str, revision: str, biases: dict, config: dict):
    """(predict, BIOES labels) for an OPF-architecture checkpoint (openai/privacy-filter and its fine-tunes)."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    n_ctx = int(config.get("n_ctx") or N_CTX)
    cap_gpu(config)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    attn = config.get("attn") or ("flex_attention" if device == "cuda" else "eager")
    model = AutoModelForTokenClassification.from_pretrained(name, revision=revision, dtype=torch.bfloat16,
                                                            attn_implementation=attn).to(device).eval()
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    start, end, trans = transition_scores(labels, biases)
    pad = model.config.pad_token_id

    @torch.inference_mode()
    def predict(docs: list[dict]) -> list[dict]:
        enc = tokenizer([d["text"] for d in docs], add_special_tokens=False, return_offsets_mapping=True)
        windows = [(i, ids[s:s + n_ctx]) for i, ids in enumerate(enc["input_ids"]) for s in range(0, len(ids), n_ctx)]
        logps: list[list] = [[] for _ in docs]
        for g in range(0, len(windows), len(docs)):  # at most len(docs) windows per forward pass
            group = windows[g:g + len(docs)]
            width = max(len(ids) for _, ids in group)
            ids = torch.tensor([w + [pad] * (width - len(w)) for _, w in group], device=device)
            att = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for _, w in group], device=device)
            logits = model(input_ids=ids, attention_mask=att).logits.float()
            for row, (i, w) in enumerate(group):
                logps[i].append(torch.log_softmax(logits[row, :len(w)], dim=-1).cpu())
        out = []
        for doc, parts, offsets in zip(docs, logps, enc["offset_mapping"], strict=True):
            text, spans = doc["text"], []
            if parts:
                logp = torch.cat(parts)
                path = viterbi(logp, start, end, trans)
                for label, a, b in token_spans(path, labels):
                    s, e = offsets[a][0], offsets[b - 1][1]
                    while s < e and text[s].isspace():
                        s += 1
                    while e > s and text[e - 1].isspace():
                        e -= 1
                    if e > s:
                        mean = sum(float(logp[t, path[t]]) for t in range(a, b)) / (b - a)
                        spans.append({"start": s, "end": e, "label": label, "score": round(math.exp(mean), 6)})
            out.append({"spans": spans, "masked_text": mask(text, spans)})
        return out

    return predict, labels


if __name__ == "__main__":
    serve(load)
