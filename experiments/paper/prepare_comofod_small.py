#!/usr/bin/env python3
"""Validate and copy the complete 512x512 base sets from CoMoFoD.

Run without options for a dry run.  Add --copy only after reviewing the
report.  The source is never modified and image bytes are copied verbatim.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "benchmark" / "CoMoFoD"
DESTINATION = ROOT / "experiments" / "paper" / "data"
BASE_NAME = re.compile(r"^(?P<id>\d{3})_(?P<kind>[FOBM])\.png$")
KINDS = ("F", "O", "B", "M")
TARGETS = {
    "F": Path("images/forged"),
    "O": Path("images/original"),
    "B": Path("masks"),
    "M": Path("masks"),
}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def discover() -> dict[str, dict[str, Path]]:
    sets: dict[str, dict[str, Path]] = {}
    for path in SOURCE.iterdir():
        if not path.is_file():
            continue
        match = BASE_NAME.fullmatch(path.name)
        if match:
            sets.setdefault(match["id"], {})[match["kind"]] = path
    return sets


def validate(sets: dict[str, dict[str, Path]]) -> tuple[list[str], Counter[str]]:
    complete: list[str] = []
    skipped: Counter[str] = Counter()
    for set_id in sorted(sets):
        files = sets[set_id]
        missing = [kind for kind in KINDS if kind not in files]
        if missing:
            skipped[f"missing {','.join(missing)}"] += 1
            continue

        sizes = {kind: image_size(files[kind]) for kind in KINDS}
        if sizes["F"] != (512, 512) or sizes["O"] != (512, 512):
            skipped[
                f"F/O not 512x512 (F={sizes['F']}, O={sizes['O']})"
            ] += 1
            continue
        if sizes["B"] != sizes["F"] or sizes["M"] != sizes["F"]:
            skipped[
                f"mask size differs (F={sizes['F']}, B={sizes['B']}, M={sizes['M']})"
            ] += 1
            continue
        complete.append(set_id)
    return complete, skipped


def copy_sets(sets: dict[str, dict[str, Path]], complete: list[str]) -> None:
    planned = [
        (sets[set_id][kind], DESTINATION / TARGETS[kind] / sets[set_id][kind].name)
        for set_id in complete
        for kind in KINDS
    ]
    conflicts = [
        destination
        for source, destination in planned
        if destination.exists() and not filecmp.cmp(source, destination, shallow=False)
    ]
    if conflicts:
        rendered = "\n".join(str(path) for path in conflicts)
        raise RuntimeError(f"Refusing to overwrite non-identical files:\n{rendered}")

    copied = 0
    already_present = 0
    for source, destination in planned:
        if destination.exists():
            already_present += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    print(f"Copied: {copied}; already identical: {already_present}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", help="copy after validation")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise RuntimeError(f"Source directory does not exist: {SOURCE}")
    sets = discover()
    complete, skipped = validate(sets)
    print(f"Base set IDs found: {len(sets)}")
    print(f"Complete 512x512 sets: {len(complete)}")
    print(f"Skipped sets: {sum(skipped.values())}")
    for reason, count in sorted(skipped.items()):
        print(f"  {count}: {reason}")
    print("Examples to copy:", ", ".join(complete[:10]))

    if args.copy:
        copy_sets(sets, complete)
    else:
        print("Dry run only. No files were copied.")


if __name__ == "__main__":
    main()
