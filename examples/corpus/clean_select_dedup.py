"""Write a new cleaned JSONL corpus, keeping all records unless apply is explicit."""
from argparse import ArgumentParser
from pathlib import Path

from sct.config import TextCleanerConfig
from sct.corpus import SelectionConfig
from sct.corpus.datatrove import curate_datatrove


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--basis', choices=('pre_pii', 'pii_normalized'))
    args = parser.parse_args()
    manifest = curate_datatrove(
        args.source, args.output, retention='caller_managed',
        config=TextCleanerConfig(check_ner_process=False),
        selection=SelectionConfig(dedup_mode='apply' if args.apply else 'report', dedup_basis=args.basis),
    )
    print(manifest.counts)


if __name__ == '__main__':
    main()
