"""Command-line interface for text and JSON cleaning."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO
from collections.abc import Sequence

from sct.adapters import process_json
from sct.presets import PRESETS, preset_config
from sct.sct import TextCleaner
from sct.utils.ner import ModelLoadError


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
        "--preset",
        choices=sorted(PRESETS),
        default="sct",
        help="Config preset: sct (library default) or training (PII only, text kept as written)",
    )
    parser.add_argument(
        "--to-ascii",
        action="store_true",
        help="Transliterate to ASCII (0.7 default: 日本語 -> Ri Ben Yu); off keeps every script",
    )
    parser.add_argument(
        "--replacement-mode",
        choices=("placeholder", "reversible", "synthetic"),
        default="placeholder",
    )
    parser.add_argument(
        "--token-map",
        help="JSON output for reversible token mappings (mode 0600 on POSIX)",
    )
    return parser


def _write_token_map(token_path: Path, anon_map) -> None:
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


def _check_reversible_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.token_map:
        parser.error("--token-map is required for reversible mode")
    if args.json_input:
        parser.error("reversible mode currently supports text input only")
    # shortcut: Windows on Python 3.13+ has os.fchmod, but it only sets the read-only flag; refuse there too
    # if the map must be restricted from other accounts.
    if not hasattr(os, "fchmod"):
        parser.error("reversible mode needs os.fchmod to restrict the token map, which this platform lacks")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> int:
    """Run the command-line cleaner."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.replacement_mode == "reversible":
        _check_reversible_args(parser, args)
    try:
        source = stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:  # missing file or not UTF-8
        parser.error(f"cannot read input: {exc}")

    try:
        cleaner = TextCleaner(cfg=preset_config(
            args.preset,
            check_ner_process=not args.no_ner,
            ner_mode="pii" if args.pii else "standard",
            replacement_mode=args.replacement_mode,
            **({"check_to_ascii_unicode": True} if args.to_ascii else {}),
        ))
        if args.json_input:
            output = json.dumps(process_json(cleaner, json.loads(source)).value, ensure_ascii=False)
        else:
            result = cleaner.process(source)
            output = result.lm_text
            if args.replacement_mode == "reversible":
                # Before stdout: masked text with no map could never be reversed.
                _write_token_map(Path(args.token_map), result.metadata["anon_map"])
    # ValueError: bad JSON, two keys cleaned to the same text, or a symlinked --token-map.
    except (ImportError, ModelLoadError, OSError, ValueError) as exc:
        parser.error(str(exc))
    stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
