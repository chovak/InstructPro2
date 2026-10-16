#!/usr/bin/env python3
"""Build and verify targets.csv for the InstructPro2 ground-truth structures.

This builds the manifest from the tree itself and
verifies it against the released JSONL.

Two kinds of entry:

  row      one target PDB per released row, matched by the row index encoded
           in the filename (test_ligand, test_ligand_unseen,
           final_antibody_test). Verified by comparing the residues in the
           PDB against the row's target_protein.

  cluster  one cluster folder per released row, joined on metadata.cluster
           (stage1/test). Every PDB in the folder is a scoring target, so
           there is nothing to verify per-file; what matters is that the
           folder exists and is non-empty.

Writes <gt>/targets.csv and prints a coverage report. Read-only apart from
that one file (and the renames under --rename-antibody).

usage:
  python3 -I build_targets_manifest.py \
      --gt   ~/instructpro2/data/eval_structures \
      --data ~/instructpro2/data
"""

import argparse
import csv
import difflib
import json
import os
import re
import sys
from collections import Counter

# split name -> (data subdir, jsonl file, ground-truth dir)
ROW_SPLITS = {
    "test_ligand":         ("ligand",   "test_ligand.jsonl",        "test_ligand"),
    "test_ligand_unseen":  ("ligand",   "test_ligand_unseen.jsonl", "test_ligand_unseen"),
    "final_antibody_test": ("antibody", "final_antibody_test.jsonl", "final_antibody_test"),
}
CLUSTER_SPLIT = ("stage1", "test.jsonl")

ID_IN_NAME = re.compile(r"(\d+)\.pdb$")
TAG = re.compile(r"<[^>]*>")
MASK = re.compile(r"<M([1-9]|[1-2][0-9]|30)>")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "MSE": "M",
}


def pdb_sequence(path, chain=None):
    """One-letter sequence from a PDB's CA atoms, first chain encountered."""
    seq = []
    want = chain
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch = line[21]
            if want is None:
                want = ch
            elif ch != want:
                continue
            seq.append(THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
    return "".join(seq)


def released_target(row):
    """The row's full target sequence as plain residues.

    An infilling target is stored as

        scaffold-with-<M1>..<Mn><SEP><M1>span1<M2>span2...

    so the spans have to be substituted back into the scaffold to recover
    the real chain. Stripping the tags and concatenating instead would give
    the right residues in the wrong order -- same length, scrambled middle.
    """
    raw = row.get("target_protein") or row.get("output") or ""
    if "<SEP>" not in raw:
        return re.sub(r"[^A-Z]", "", TAG.sub("", raw))

    head, _, tail = raw.partition("<SEP>")
    found = list(MASK.finditer(tail))
    spans = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(tail)
        spans[m.group()] = re.sub(r"[^A-Z]", "", tail[m.end():end])

    rebuilt = head
    for tag, span in spans.items():
        rebuilt = rebuilt.replace(tag, span)
    return re.sub(r"[^A-Z]", "", MASK.sub("", rebuilt))


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def similarity(a, b):
    """Gap-tolerant sequence similarity.

    Position-by-position comparison is useless here: a crystal structure
    routinely has unresolved residues, and a single missing one shifts
    every position after it. difflib finds the matching blocks instead, so
    an internal gap costs only the residues actually absent.
    """
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def do_row_split(split, data_root, gt_root, args, writer, report):
    subdir, basename, gt_name = ROW_SPLITS[split]
    jsonl = os.path.join(data_root, subdir, basename)
    gt_dir = os.path.join(gt_root, gt_name)

    print("-" * 74)
    print(f"{split}")

    if not os.path.exists(jsonl):
        print(f"  JSONL missing: {jsonl}")
        return
    if not os.path.isdir(gt_dir):
        print(f"  ground-truth dir missing: {gt_dir}")
        return

    # row index -> filename, taken from the trailing number in each name
    by_id = {}
    collisions = 0
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith(".pdb"):
            continue
        m = ID_IN_NAME.search(name)
        if not m:
            continue
        rid = int(m.group(1))
        if rid in by_id:
            collisions += 1
        by_id[rid] = name

    rows = load_jsonl(jsonl)
    print(f"  released rows   : {len(rows):,}")
    print(f"  target PDBs     : {len(by_id):,}")
    if collisions:
        print(f"  WARNING: {collisions} filenames share a row index")

    verdicts = Counter()
    worst = []

    for idx, row in enumerate(rows):
        name = by_id.get(idx)
        if name is None:
            verdicts["no_target"] += 1
            writer.writerow({
                "split": split, "kind": "row", "orig_id": idx,
                "cluster_id": "", "target": "", "n_targets": 0,
                "seq_match": "", "status": "no_target",
            })
            continue

        expected = released_target(row)
        observed = pdb_sequence(os.path.join(gt_dir, name))
        score = similarity(expected, observed)

        if score >= 0.99:
            status = "exact"
        elif score >= args.min_similarity:
            status = "close"
        else:
            status = "MISMATCH"
            if len(worst) < 5:
                worst.append((idx, name, score, len(expected), len(observed)))
        verdicts[status] += 1

        writer.writerow({
            "split": split, "kind": "row", "orig_id": idx,
            "cluster_id": "",
            "target": os.path.join(gt_name, name),
            "n_targets": 1,
            "seq_match": f"{score:.3f}",
            "status": status,
        })

    total = len(rows)
    scored = total - verdicts["no_target"]
    print(f"  exact match     : {verdicts['exact']:,}")
    print(f"  close match     : {verdicts['close']:,}  "
          f"(>= {args.min_similarity:.0%}, differing termini or modelled gaps)")
    print(f"  MISMATCH        : {verdicts['MISMATCH']:,}")
    print(f"  no target       : {verdicts['no_target']:,}  "
          f"({verdicts['no_target'] / total:.1%} of rows unscorable)")

    if verdicts["MISMATCH"]:
        print("  examples:")
        for idx, name, score, le, lo in worst:
            print(f"    row {idx:<5} {name:<44} match {score:.2f} "
                  f"(released {le} aa, pdb {lo} aa)")
        report.append(f"{split}: {verdicts['MISMATCH']} rows point at the wrong structure")
    elif scored:
        report.append(f"{split}: {scored:,} targets verified, "
                      f"{verdicts['no_target']:,} rows have none")


def do_cluster_split(data_root, gt_root, writer, report):
    subdir, basename = CLUSTER_SPLIT
    jsonl = os.path.join(data_root, subdir, basename)

    print("-" * 74)
    print("stage1/test (cluster-based)")

    if not os.path.exists(jsonl):
        print(f"  JSONL missing: {jsonl}")
        return

    rows = load_jsonl(jsonl)
    print(f"  released rows   : {len(rows):,}")

    counts = Counter()
    members_total = 0
    sizes = []

    for idx, row in enumerate(rows):
        cluster = (row.get("metadata") or {}).get("cluster")
        if cluster is None:
            counts["no_cluster_field"] += 1
            writer.writerow({
                "split": "stage1_test", "kind": "cluster", "orig_id": idx,
                "cluster_id": "", "target": "", "n_targets": 0,
                "seq_match": "", "status": "no_cluster_field",
            })
            continue

        cid = f"cluster_{cluster}"
        cdir = os.path.join(gt_root, cid)
        if not os.path.isdir(cdir):
            counts["missing_folder"] += 1
            status, n = "missing_folder", 0
        else:
            n = len([f for f in os.listdir(cdir) if f.endswith(".pdb")])
            if n == 0:
                counts["empty_folder"] += 1
                status = "empty_folder"
            else:
                counts["ok"] += 1
                status = "ok"
                members_total += n
                sizes.append(n)

        writer.writerow({
            "split": "stage1_test", "kind": "cluster", "orig_id": idx,
            "cluster_id": cid, "target": cid if n else "",
            "n_targets": n, "seq_match": "", "status": status,
        })

    print(f"  scorable rows   : {counts['ok']:,}")
    print(f"  empty cluster   : {counts['empty_folder']:,}")
    print(f"  missing folder  : {counts['missing_folder']:,}")
    print(f"  no cluster field: {counts['no_cluster_field']:,}")
    if sizes:
        sizes.sort()
        print(f"  members/cluster : min {sizes[0]}  median {sizes[len(sizes)//2]}  "
              f"max {sizes[-1]}  total {members_total:,}")

    unscorable = counts["empty_folder"] + counts["missing_folder"] + counts["no_cluster_field"]
    if unscorable:
        report.append(f"stage1_test: {counts['ok']:,} rows scorable, "
                      f"{unscorable:,} have no usable cluster")
    else:
        report.append(f"stage1_test: all {counts['ok']:,} rows scorable")


def rename_antibody(gt_root):
    """Rename target_test_antibody_v2_*.pdb to match the released split name."""
    d = os.path.join(gt_root, "final_antibody_test")
    if not os.path.isdir(d):
        print("final_antibody_test/ not found; skipping rename")
        return
    n = 0
    for name in sorted(os.listdir(d)):
        if name.startswith("target_test_antibody_v2_") and name.endswith(".pdb"):
            new = name.replace("target_test_antibody_v2_",
                               "target_final_antibody_test_")
            os.rename(os.path.join(d, name), os.path.join(d, new))
            n += 1
    print(f"renamed {n} antibody target files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="eval_structures directory")
    ap.add_argument("--data", required=True, help="released data root")
    ap.add_argument("--out", default=None, help="default: <gt>/targets.csv")
    ap.add_argument("--min-similarity", type=float, default=0.90,
                    help="below this a row is reported as a mismatch (default 0.90)")
    ap.add_argument("--rename-antibody", action="store_true",
                    help="rename target_test_antibody_v2_* to match the split name")
    args = ap.parse_args()

    if not os.path.isdir(args.gt):
        sys.exit(f"not a directory: {args.gt}")
    out = args.out or os.path.join(args.gt, "targets.csv")

    if args.rename_antibody:
        rename_antibody(args.gt)

    report = []
    fields = ["split", "kind", "orig_id", "cluster_id", "target",
              "n_targets", "seq_match", "status"]

    print("=" * 74)
    print("BUILDING targets.csv")
    print("=" * 74)

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for split in ROW_SPLITS:
            do_row_split(split, args.data, args.gt, args, writer, report)
        do_cluster_split(args.data, args.gt, writer, report)

    print("=" * 74)
    print(f"wrote {out}")
    for line in report:
        print(f"  {line}")
    print("=" * 74)
    print("\nPaths in `target` are relative to the ground-truth root, so the")
    print("tree can be published and relocated without rewriting anything.")


if __name__ == "__main__":
    main()