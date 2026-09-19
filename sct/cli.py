"""Command-line interface for text and JSON cleaning."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sct-clean")
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input file, or - for standard input",
    )
    parser.add_argument("--no-ner", action="store_true")
    parser.add_argument("--pii", action="store_true")
    parser.add_argument("--json-input", action="store_true")
    parser.add_argument(
        "--replacement-mode",
        choices=("placeholder", "reversible", "synthetic"),
        default="placeholder",
    )
    parser.add_argument(
        "--token-map",
        help="Restricted JSON output for reversible token mappings",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> int:
    """Run the command-line cleaner."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.replacement_mode == "reversible" and not args.token_map:
        parser.error("--token-map is required for reversible mode")
    if args.json_input and args.replacement_mode == "reversible":
        parser.error("reversible mode currently supports text input only")
    if args.input == "-":
        source = stdin.read()
    else:
        source = Path(args.input).read_text(encoding="utf-8")

    try:
        cleaner = TextCleaner(cfg=TextCleanerConfig(
            check_ner_process=not args.no_ner,
            ner_mode="pii" if args.pii else "standard",
            replacement_mode=args.replacement_mode,
        ))
    except ImportError as exc:
        parser.error(str(exc))
    if args.json_input:
        cleaned = process_json(cleaner, json.loads(source)).value
        stdout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    else:
        result = cleaner.process(source)
        stdout.write(result.lm_text + "\n")
        if args.replacement_mode == "reversible":
            anon_map = result.metadata["anon_map"]
            token_path = Path(args.token_map)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            if token_path.is_symlink():
                raise ValueError("--token-map must not be a symbolic link")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(
                token_path,
                flags,
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
            except Exception:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
                json.dump(
                    anon_map.to_dict(),
                    token_file,
                    indent=2,
                    sort_keys=True,
                )
                token_file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
