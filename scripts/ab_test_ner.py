#!/usr/bin/env python3
"""A/B comparison script for torch vs ONNX NER output.

Usage:
    # On main branch (torch venv):
    python scripts/ab_test_ner.py --backend torch --output results_torch.json

    # On develop branch (onnx venv):
    python scripts/ab_test_ner.py --backend onnx --output results_onnx.json

    # Compare results:
    python scripts/ab_test_ner.py --compare results_torch.json results_onnx.json
"""

import argparse
import json
import sys

# Test inputs covering all languages, entity types, edge cases
TEST_INPUTS = [
    # English — single entity types
    {"id": "en_person", "text": "John Smith works at the office.", "lang": "ENGLISH"},
    {"id": "en_org", "text": "Microsoft Corporation announced quarterly results.", "lang": "ENGLISH"},
    {"id": "en_location", "text": "The conference was held in New York.", "lang": "ENGLISH"},

    # English — multi-entity
    {"id": "en_multi", "text": "John Smith works at Microsoft in New York.", "lang": "ENGLISH"},
    {"id": "en_adjacent", "text": "Dr. Sarah Wilson met Angela Merkel in Berlin.", "lang": "ENGLISH"},

    # English — no entities
    {"id": "en_none", "text": "The quick brown fox jumps over the lazy dog.", "lang": "ENGLISH"},

    # German
    {"id": "de_mixed", "text": "Angela Merkel besuchte die Technische Universität Berlin.", "lang": "GERMAN"},

    # Dutch
    {"id": "nl_mixed", "text": "Willem-Alexander woont in Den Haag.", "lang": "DUTCH"},

    # Spanish
    {"id": "es_mixed", "text": "Pablo García trabaja en Madrid para Google.", "lang": "SPANISH"},

    # Unicode / accented names
    {"id": "unicode", "text": "José María González lives in São Paulo.", "lang": "ENGLISH"},

    # Short text
    {"id": "short", "text": "John.", "lang": "ENGLISH"},

    # Subword challenges
    {"id": "subword", "text": "Arnold Schwarzenegger visited Liechtenstein.", "lang": "ENGLISH"},

    # All-entity text
    {"id": "all_entity", "text": "John Smith Sarah Johnson Microsoft Google New York London.", "lang": "ENGLISH"},

    # Long text requiring chunking
    {
        "id": "long_chunking",
        "text": "John Smith works at Microsoft in New York. " * 50,
        "lang": "ENGLISH",
    },
]


def run_backend(backend: str, output_path: str):
    """Run NER on all test inputs and save results."""
    results = []

    if backend == "torch":
        _run_torch(results)
    elif backend == "onnx":
        _run_onnx(results)
    else:
        print(f"Unknown backend: {backend}")
        sys.exit(1)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} results to {output_path}")


def _run_torch(results):
    """Run with PyTorch/transformers backend (main branch)."""
    from sct.utils.ner import GeneralNER

    ner = GeneralNER(device='cpu')
    _process_all(ner, results)


def _run_onnx(results):
    """Run with ONNX Runtime backend (develop branch)."""
    from sct.utils.ner import GeneralNER

    ner = GeneralNER(device='cpu')
    _process_all(ner, results)


def _process_all(ner, results):
    """Process all test inputs through the NER pipeline."""
    for test in TEST_INPUTS:
        print(f"Processing: {test['id']}...")
        try:
            processed = ner.ner_process(
                test['text'],
                positional_tags=['PER', 'LOC', 'ORG', 'MISC'],
                ner_confidence_threshold=0.85,
                language=test.get('lang'),
            )
            results.append({
                'id': test['id'],
                'input': test['text'][:200],  # truncate long inputs
                'output': processed[:500],
                'lang': test.get('lang'),
            })
        except Exception as e:
            results.append({
                'id': test['id'],
                'error': str(e),
            })


def compare_results(file_a: str, file_b: str):
    """Compare two result files and report differences."""
    with open(file_a) as f:
        results_a = {r['id']: r for r in json.load(f)}
    with open(file_b) as f:
        results_b = {r['id']: r for r in json.load(f)}

    all_ids = sorted(set(results_a.keys()) | set(results_b.keys()))

    exact_match = 0
    total = 0
    mismatches = []

    print(f"\n{'='*70}")
    print(f"A/B Comparison: {file_a} vs {file_b}")
    print(f"{'='*70}\n")

    for test_id in all_ids:
        a = results_a.get(test_id, {})
        b = results_b.get(test_id, {})

        if 'error' in a or 'error' in b:
            print(f"  {test_id}: ERROR in {'A' if 'error' in a else 'B'}")
            print(f"    A: {a.get('error', a.get('output', 'N/A')[:100])}")
            print(f"    B: {b.get('error', b.get('output', 'N/A')[:100])}")
            continue

        total += 1
        out_a = a.get('output', '')
        out_b = b.get('output', '')

        if out_a == out_b:
            exact_match += 1
            print(f"  {test_id}: EXACT MATCH")
        else:
            mismatches.append(test_id)
            print(f"  {test_id}: MISMATCH")
            print(f"    A: {out_a[:150]}")
            print(f"    B: {out_b[:150]}")

            # Check if entity tokens are the same
            a_tokens = set(t for t in out_a.split() if t.startswith('<') and t.endswith('>'))
            b_tokens = set(t for t in out_b.split() if t.startswith('<') and t.endswith('>'))
            if a_tokens == b_tokens:
                print("    NOTE: Same entity tokens detected, text differs only in non-entity spans")

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total comparisons: {total}")
    print(f"  Exact matches:     {exact_match}/{total} ({100*exact_match/total:.1f}%)" if total else "  No comparisons")
    print(f"  Mismatches:        {len(mismatches)}")
    if mismatches:
        print(f"  Mismatched IDs:    {', '.join(mismatches)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="A/B test NER backends")
    parser.add_argument("--backend", choices=["torch", "onnx"],
                        help="Backend to run")
    parser.add_argument("--output", type=str,
                        help="Output JSON file path")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.backend and args.output:
        run_backend(args.backend, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
