#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def print_tree(value: Any, name: str = "$", depth: int = 0, max_depth: int = 4) -> None:
    indent = "  " * depth

    if isinstance(value, dict):
        print(f"{indent}{name}: dict ({len(value)} keys)")

        if depth >= max_depth:
            return

        for key, child in value.items():
            print_tree(child, str(key), depth + 1, max_depth)

    elif isinstance(value, list):
        print(f"{indent}{name}: list ({len(value)} items)")

        if depth >= max_depth or not value:
            return

        print_tree(value[0], "[0]", depth + 1, max_depth)

    else:
        representation = repr(value)

        if len(representation) > 160:
            representation = representation[:157] + "..."

        print(f"{indent}{name}: {type(value).__name__} = {representation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    for path in args.files:
        print()
        print("=" * 100)
        print(path)
        print("=" * 100)

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        print_tree(data, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
