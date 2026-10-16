#!/usr/bin/env python3
"""Check a built release data tree before uploading it.

Confirms every split the loader and the scripts expect is present, that each
file carries the keys the readers require, and that the Stage 2 splits really
do carry their modality inputs.

    python scripts/verify_release_data.py /path/to/release_data

Exits non-zero if anything is wrong.
"""

import json
import os
import sys

EXPECTED = {
    "stage1":   ["train.jsonl", "valid.jsonl", "test.jsonl"],
    "ligand":   ["train.jsonl", "valid.jsonl", "test_ligand.jsonl", "test_ligand_unseen.jsonl"],
    "antibody": ["train.jsonl", "valid.jsonl", "final_antibody_test.jsonl"],
}

# Modality input each Stage 2 split must carry, per fairseq/data/indexed_dataset.py
MODALITY = {"ligand": "ligand", "antibody": "antigen"}

# Placeholder keys that should have been stripped from the Stage 2 splits.
VESTIGIAL = ["antibody", "ligand"]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: verify_release_data.py <release_data_dir>")
    root = sys.argv[1]

    problems, notes = [], []
    print(f"{'file':44s} {'rows':>9s} {'size':>9s}  keys")
    print("-" * 115)

    for d, files in EXPECTED.items():
        for fn in files:
            p = os.path.join(root, d, fn)
            if not os.path.exists(p):
                problems.append(f"MISSING {d}/{fn}")
                print(f"{d + '/' + fn:44s} {'-':>9s} {'-':>9s}  MISSING")
                continue

            rows, first = 0, None
            bad_json = 0
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows += 1
                    if first is None:
                        try:
                            first = json.loads(line)
                        except json.JSONDecodeError:
                            bad_json += 1

            size = os.path.getsize(p)
            keys = sorted(first) if first else []
            print(f"{d + '/' + fn:44s} {rows:9,d} {size/1e6:8.1f}M  {','.join(keys)}")

            if bad_json:
                problems.append(f"{d}/{fn}: first line is not valid JSON")
                continue
            if not first:
                problems.append(f"{d}/{fn} is empty")
                continue
            if "tokenized_instruction" not in first:
                problems.append(f"{d}/{fn}: no tokenized_instruction (the loader requires it)")
            if not ("target_protein" in first or "output" in first):
                problems.append(f"{d}/{fn}: neither target_protein nor output")

            mod = MODALITY.get(d)
            if mod and not first.get(mod):
                problems.append(f"{d}/{fn}: first row has empty or missing '{mod}'")
            if mod:
                leftover = [k for k in VESTIGIAL if k != mod and k in first]
                if leftover:
                    notes.append(f"{d}/{fn}: still carries placeholder key(s) {leftover}")

    print()
    for n in notes:
        print("NOTE:", n)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("All files present and carry the keys the loader requires.")


if __name__ == "__main__":
    main()
