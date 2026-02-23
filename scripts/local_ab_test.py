#!/usr/bin/env python3
"""Local A/B comparison: HuggingFace torch pipeline vs ONNX pipeline.

Runs both backends on the same inputs in a single process and compares
entity labels, spans, and confidence scores.

Requires: torch, transformers, onnxruntime (all in current env)
"""

import numpy as np

# ── Torch (HuggingFace) pipeline ──
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
import torch

# ── ONNX pipeline (our implementation) ──
from sct.utils.onnx_pipeline import load_onnx_ner_model


def format_entities(entities):
    """Format entity list for display."""
    parts = []
    for e in entities:
        score = e['score']
        parts.append(f"({e['entity_group']}, '{e['word']}', {score:.4f})")
    return "[" + ", ".join(parts) + "]"


def main():
    torch_model_name = "dslim/bert-base-NER"
    onnx_model_name = "protectai/bert-base-NER-onnx"

    print("Loading torch pipeline...")
    tok = AutoTokenizer.from_pretrained(torch_model_name)
    model = AutoModelForTokenClassification.from_pretrained(torch_model_name)
    torch_pipe = hf_pipeline(
        "ner", model=model, tokenizer=tok,
        aggregation_strategy="simple", device="cpu"
    )

    print("Loading ONNX pipeline...")
    onnx_pipe, _ = load_onnx_ner_model(onnx_model_name, device="cpu")

    # Test texts
    texts = [
        "John Smith works at Microsoft in New York.",
        "Angela Merkel besuchte Berlin.",
        "The quick brown fox jumps over the lazy dog.",
        "Dr. Sarah Wilson met Pablo Garcia in Madrid.",
        "Arnold Schwarzenegger visited Liechtenstein.",
        "Jose Maria Gonzalez lives in Sao Paulo.",
        "John.",
        "Microsoft Corporation and Google Inc. are in Silicon Valley.",
        "Willem woont in Amsterdam.",
        "John Smith works at Microsoft in New York. " * 3,
    ]

    print("=" * 80)
    print("A/B COMPARISON: HuggingFace torch vs ONNX pipeline")
    print("=" * 80)

    exact_match = 0
    entity_match = 0
    total = 0
    score_diffs = []

    for i, text in enumerate(texts):
        display = text[:80] + ("..." if len(text) > 80 else "")
        print(f'\n--- Test {i + 1}: "{display}" ---')

        # Torch
        with torch.no_grad():
            torch_results = torch_pipe(text)
        torch_entities = [
            {
                "entity_group": r["entity_group"],
                "word": r["word"],
                "start": r["start"],
                "end": r["end"],
                "score": float(r["score"]),
            }
            for r in torch_results
        ]

        # ONNX
        onnx_results = onnx_pipe([text])[0]
        onnx_entities = [
            {
                "entity_group": r["entity_group"],
                "word": r["word"],
                "start": r["start"],
                "end": r["end"],
                "score": float(r["score"]),
            }
            for r in onnx_results
        ]

        total += 1

        # Compare entity labels + spans (ignoring scores)
        torch_labels = [(e["entity_group"], e["start"], e["end"]) for e in torch_entities]
        onnx_labels = [(e["entity_group"], e["start"], e["end"]) for e in onnx_entities]

        labels_match = torch_labels == onnx_labels
        if labels_match:
            entity_match += 1

        # Compare scores
        scores_close = True
        if len(torch_entities) == len(onnx_entities):
            for te, oe in zip(torch_entities, onnx_entities):
                diff = abs(te["score"] - oe["score"])
                score_diffs.append(diff)
                if diff > 0.01:
                    scores_close = False

        exact = labels_match and scores_close
        if exact:
            exact_match += 1

        status = "EXACT" if exact else ("LABELS_MATCH" if labels_match else "MISMATCH")
        print(f"  Status: {status}")
        print(f"  Torch entities ({len(torch_entities)}): {format_entities(torch_entities)}")
        print(f"  ONNX  entities ({len(onnx_entities)}): {format_entities(onnx_entities)}")

        if not labels_match:
            torch_only = set(torch_labels) - set(onnx_labels)
            onnx_only = set(onnx_labels) - set(torch_labels)
            if torch_only:
                print(f"  Torch-only: {torch_only}")
            if onnx_only:
                print(f"  ONNX-only:  {onnx_only}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests:          {total}")
    print(f"Exact match:          {exact_match}/{total} ({100 * exact_match / total:.1f}%)")
    print(f"Entity label match:   {entity_match}/{total} ({100 * entity_match / total:.1f}%)")
    if score_diffs:
        print(f"Score diff (mean):    {np.mean(score_diffs):.6f}")
        print(f"Score diff (max):     {np.max(score_diffs):.6f}")
        print(f"Score diff (median):  {np.median(score_diffs):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
