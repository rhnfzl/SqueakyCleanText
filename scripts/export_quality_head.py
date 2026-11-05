"""Export the FineWeb2-HQ-Plus quality classifier to one ONNX graph.

Graph: input_ids, attention_mask -> xlm-roberta-base mean-pooled embedding
-> MLP head -> sigmoid score in [0, 1]. Writes ``model.onnx``,
``tokenizer.json`` and ``sct_quality.json`` (sources and revisions) into
``--out`` for ``sct.score.ONNXQualityScorer`` / ``sct-report --quality-model``.

Needs torch and transformers (export time only; SCT never loads the pickle
at runtime). Licences checked 2026-10-31: head epfml/FineWeb2-HQ-Plus-Classifier
Apache-2.0, backbone FacebookAI/xlm-roberta-base MIT.

    python scripts/export_quality_head.py --out models/fw2hq-plus
"""

import argparse
import json
from pathlib import Path

import huggingface_hub
import torch
from transformers import AutoModel, AutoTokenizer

from sct.utils.atomic_file import write_text_atomic

HEAD_REPO = "epfml/FineWeb2-HQ-Plus-Classifier"
BACKBONE_REPO = "FacebookAI/xlm-roberta-base"


class QualityModel(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        # Masked mean = the model card's mean(1) for unpadded single texts.
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.sigmoid(self.head(pooled)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--head-revision", default="main")
    parser.add_argument("--backbone-revision", default="main")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    head_dir = huggingface_hub.snapshot_download(HEAD_REPO, revision=args.head_revision)
    head = torch.nn.Sequential(
        torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 1),
    )
    state = torch.load(f"{head_dir}/model.pt", weights_only=True, map_location="cpu")
    head.load_state_dict({k.removeprefix("classifier."): v for k, v in state.items()})
    backbone = AutoModel.from_pretrained(BACKBONE_REPO, revision=args.backbone_revision)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_REPO, revision=args.backbone_revision)

    model = QualityModel(backbone, head).eval()
    sample = tokenizer(["export sample", "a second, longer export sample"],
                       return_tensors="pt", padding=True)
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["score"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "score": {0: "batch"}},
        opset_version=args.opset,
    )
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    info = huggingface_hub.HfApi()
    write_text_atomic(out / "sct_quality.json", json.dumps({
        "head": HEAD_REPO,
        "head_revision": info.model_info(HEAD_REPO, revision=args.head_revision).sha,
        "backbone": BACKBONE_REPO,
        "backbone_revision": info.model_info(BACKBONE_REPO, revision=args.backbone_revision).sha,
        "output": "sigmoid(head(masked mean of last_hidden_state)), computed on original text",
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
