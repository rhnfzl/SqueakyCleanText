#!/usr/bin/env python3
"""One-time ONNX model export script.

Converts PyTorch NER models to ONNX format and uploads to HuggingFace Hub.
Requires a temporary environment with: torch, optimum[onnxruntime], transformers, huggingface_hub.

Usage:
    pip install torch transformers optimum[onnxruntime] huggingface_hub
    huggingface-cli login
    python scripts/export_onnx_models.py
    python scripts/export_onnx_models.py --models dslim/bert-base-NER  # single model
    python scripts/export_onnx_models.py --dry-run  # preview without uploading
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Source → ONNX repo mapping
#
# Naming convention for target repos: rhnfzl/xlm-roberta-large-ner-{language}-onnx
# or rhnfzl/xlm-roberta-large-conll0{2,3}-{language}-onnx for CoNLL-trained models.
#
# To add a new language:
#   1. Find a suitable source model on HuggingFace (prefer XLM-RoBERTa-based, CoNLL-style labels)
#   2. Add an entry below: "source-org/model-name": "rhnfzl/target-onnx-repo-name"
#   3. Run: python scripts/export_onnx_models.py --models source-org/model-name
#   4. After upload, update DEFAULT_NER_MODELS in sct/config.py with the new ONNX repo ID
#
# CoNLL-02/03 coverage: English, German, Dutch, Spanish (FacebookAI series below).
# Other languages: search HuggingFace for XLM-RoBERTa NER models fine-tuned on
# WikiNER, Evalita (Italian), HAREM (Portuguese), QUAERO (French), or similar corpora.
#
# Pending — add entries below when suitable source models are identified:
#   "source-org/xlm-roberta-large-french-ner":     "rhnfzl/xlm-roberta-large-ner-french-onnx",
#   "source-org/xlm-roberta-large-portuguese-ner": "rhnfzl/xlm-roberta-large-ner-portuguese-onnx",
#   "source-org/xlm-roberta-large-italian-ner":    "rhnfzl/xlm-roberta-large-ner-italian-onnx",
MODEL_MAP = {
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": "rhnfzl/xlm-roberta-large-conll03-english-onnx",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": "rhnfzl/xlm-roberta-large-conll02-dutch-onnx",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german": "rhnfzl/xlm-roberta-large-conll03-german-onnx",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": "rhnfzl/xlm-roberta-large-conll02-spanish-onnx",
    "Babelscape/wikineural-multilingual-ner": "rhnfzl/wikineural-multilingual-ner-onnx",
    "dslim/bert-base-NER": "rhnfzl/bert-base-NER-onnx",
    # ModernBERT NER models (export with --device cpu to avoid FlashAttention issues).
    # Optimization (--optimize) is not yet supported for ModernBERT architecture.
    # English-only, 8192 token context — optional alternative to XLM-RoBERTa defaults.
    "MatteoFasulo/ModernBERT-base-NER": "rhnfzl/modernbert-base-ner-conll03-english-onnx",
}

# Models requiring special export flags (overrides default optimum-cli invocation)
MODEL_EXPORT_FLAGS: dict[str, list[str]] = {
    # ModernBERT: must use CPU to avoid FlashAttention Triton errors
    "MatteoFasulo/ModernBERT-base-NER": ["--device", "cpu"],
}

# Files to include in the ONNX repo (model.onnx_data added for large models
# where optimum splits weights into an external data file)
REQUIRED_FILES = [
    "model.onnx",
    "model.onnx_data",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def export_model(source_model: str, output_dir: Path) -> None:
    """Export a model to ONNX using optimum-cli."""
    print(f"\n{'='*60}")
    print(f"Exporting: {source_model}")
    print(f"Output:    {output_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", source_model,
        "--task", "token-classification",
    ]
    # Apply model-specific export flags (e.g. --device cpu for ModernBERT)
    extra_flags = MODEL_EXPORT_FLAGS.get(source_model, [])
    if extra_flags:
        cmd.extend(extra_flags)
        print(f"  Using extra flags: {' '.join(extra_flags)}")
    cmd.append(str(output_dir))
    subprocess.run(cmd, check=True)  # noqa: S603

    # Verify required files exist
    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found in export output")
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {fname}: {size_mb:.1f} MB")


def upload_model(output_dir: Path, repo_id: str, dry_run: bool = False) -> None:
    """Upload ONNX model to HuggingFace Hub."""
    if dry_run:
        print(f"  DRY RUN: Would upload {output_dir} → {repo_id}")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload each required file
    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"  Uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export NER models to ONNX format")
    parser.add_argument(
        "--models", nargs="+",
        help="Specific source models to export (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Export but don't upload to Hub",
    )
    parser.add_argument(
        "--output-base", type=Path, default=Path("onnx_exports"),
        help="Base directory for exports (default: ./onnx_exports)",
    )
    parser.add_argument(
        "--keep-exports", action="store_true",
        help="Don't clean up export directories after upload",
    )
    args = parser.parse_args()

    # Determine which models to export
    if args.models:
        models = {m: MODEL_MAP[m] for m in args.models if m in MODEL_MAP}
        unknown = [m for m in args.models if m not in MODEL_MAP]
        if unknown:
            print(f"Unknown models (not in MODEL_MAP): {unknown}")
            sys.exit(1)
    else:
        models = MODEL_MAP

    print(f"Will export {len(models)} model(s):")
    for src, dst in models.items():
        print(f"  {src} → {dst}")

    args.output_base.mkdir(parents=True, exist_ok=True)

    for source_model, onnx_repo in models.items():
        safe_name = source_model.replace("/", "__")
        output_dir = args.output_base / safe_name

        try:
            export_model(source_model, output_dir)
            upload_model(output_dir, onnx_repo, dry_run=args.dry_run)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        finally:
            if not args.keep_exports and output_dir.exists():
                shutil.rmtree(output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
