"""Write a new cleaned JSONL corpus, keeping all records unless apply is explicit."""
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path

from sct.config import TextCleanerConfig
from sct.corpus import SelectionConfig, ReferenceIndex, curate_resumable
from sct.corpus.datatrove import curate_datatrove


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', action='store_true', help='resume only verified identical private checkpoints')
    parser.add_argument('--references', type=Path, help='caller-built private SCT ReferenceIndex database')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--basis', choices=('pre_pii', 'pii_normalized'))
    args = parser.parse_args()
    with ExitStack() as stack:
        references = stack.enter_context(ReferenceIndex(args.references)) if args.references else None
        recipe = curate_resumable if args.resume or references is not None else curate_datatrove
        options = {'references': references} if recipe is curate_resumable else {}
        manifest = recipe(
            args.source, args.output, retention='caller_managed',
            config=TextCleanerConfig(check_ner_process=False),
            selection=SelectionConfig(dedup_mode='apply' if args.apply else 'report', dedup_basis=args.basis),
            **options,
        )
    print(manifest.counts)


if __name__ == '__main__':
    main()
