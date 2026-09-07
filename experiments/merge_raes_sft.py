#!/usr/bin/env python3
"""Merge multiple RAES SFT export directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.training_export import merge_sft_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dirs", nargs="+", help="Export directories containing SFT JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Merged output directory.")
    parser.add_argument("--name", default="merged", help="Dataset name for the merge report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = merge_sft_exports(
        export_dirs=[Path(path) for path in args.export_dirs],
        output_dir=args.output_dir,
        name=args.name,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
