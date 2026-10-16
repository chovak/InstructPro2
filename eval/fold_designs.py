#!/usr/bin/env python3
"""Fold InstructPro2 generations with ESMFold.

Replaces two scripts that had drifted apart:
    eval/fold_instructpro2_downstream_output.py   (Stage 2: ligand, antibody)
    fold_stage1_eval.py                           (Stage 1: AR + infilling)

Both parsed design.txt, cleaned sequences and ran ESMFold; they disagreed on
residue sanitizing, infill length caps and OOM handling. This merges them.

    --mode stage1
        One design.txt holding mixed autoregressive and infilling rows.
        Each row is routed by `task_type` from the reference JSONL and
        filed under <output_dir>/<task_type>/cluster_<id>/.
        Infill spans capped at 50 residues; unfulfilled masks are deleted
        from the scaffold and the row is still folded.

    --mode downstream
        One design.txt per dataset subdirectory of --checkpoint_dir.
        Ligand rows are cleaned as full sequences. Antibody rows are
        rebuilt from their HCDR spans (cap 35); a row missing <SEP> or any
        span is dropped, since a partial heavy chain is not meaningful.

Both modes write one PDB per design plus a mapping CSV for the scorer.

Examples:

    # Stage 1
    python fold_designs.py --mode stage1 \\
        --design_txt      $OUT/stage1_eval/design.txt \\
        --reference_jsonl $DATA/stage1/test.jsonl \\
        --output_dir      $OUT/stage1_eval_pdb \\
        --mapped_csv      $OUT/stage1_eval_pdb/mapped.csv

    # Stage 2
    python fold_designs.py --mode downstream \\
        --checkpoint_dir $OUT/ligand_eval_w_modality \\
        --output_dir     $OUT/ligand_eval_pdb \\
        --run_name       ligand_eval_w_modality \\
        --datasets       test_ligand,test_ligand_unseen
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

# Counters for every way a generation can be malformed. Printed before
# folding starts, because a high count here usually means the generation
# step used the wrong flags.
STATS = defaultdict(int)

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

MASK_PATTERN = re.compile(r"<M([1-9]|[1-2][0-9]|30)>")
SPECIAL_PATTERN = re.compile(r"<SEP>|<M\d+>")
# A run of 20+ repeats of a 1-20mer: the classic decoding collapse.
KMER_LOOP_PATTERN = re.compile(r"([A-Z]{1,20})\1{19,}")

# Infill span caps. Stage 1 spans are general; HCDRs are short by construction.
STAGE1_MAX_INFILL = 50
HCDR_MAX_INFILL = 35


# ---------------------------------------------------------------------------
# sequence cleaning (shared by both modes)
# ---------------------------------------------------------------------------

def sanitize_residues(seq):
    """Map non-standard codes (B, Z, U, O, X...) to X.

    The ESMFold tokenizer does not carry the full extended alphabet, and an
    unknown character raises mid-batch, taking the whole batch with it.
    """
    return "".join(c if c in VALID_RESIDUES else "X" for c in seq)


def break_kmer_loops(seq, tag):
    """Collapse a degenerate repeat run to 5 copies."""
    match = KMER_LOOP_PATTERN.search(seq)
    if match:
        STATS[f"{tag}_kmer_loops_broken"] += 1
        seq = seq[:match.start()] + match.group(1) * 5
    return seq


def residues_only(seq):
    return re.sub(r"[^A-Z]", "", seq)


def parse_autoregressive(raw):
    """Clean a full-sequence generation."""
    if SPECIAL_PATTERN.search(raw):
        STATS["ar_hallucinated_special_tokens"] += 1
        raw = SPECIAL_PATTERN.sub("", raw)
    raw = break_kmer_loops(raw, "ar")
    return sanitize_residues(residues_only(raw))


def parse_infilling(raw, max_infill_length, require_all_masks, tag="infill"):
    """Rebuild a sequence from SCAFFOLD<SEP><M1>span<M2>span...

    Returns the reconstructed sequence, or None if it cannot be rebuilt.
    With require_all_masks, a row missing any requested span is dropped;
    otherwise the unfulfilled masks are deleted from the scaffold.
    """
    splits = raw.split("<SEP>")
    if len(splits) < 2:
        STATS[f"{tag}_missing_sep"] += 1
        return None

    scaffold = splits[0]
    answers = "<SEP>".join(splits[1:])
    if "<SEP>" in answers:
        STATS[f"{tag}_multiple_seps_swallowed"] += 1

    found = list(MASK_PATTERN.finditer(answers))
    extracted = {}
    for i, match in enumerate(found):
        mask = match.group()
        start = match.end()
        end = found[i + 1].start() if i + 1 < len(found) else len(answers)
        span = residues_only(answers[start:end])

        if len(span) > max_infill_length:
            STATS[f"{tag}_masks_truncated_by_cap"] += 1
            span = span[:max_infill_length]

        if mask in extracted:
            STATS[f"{tag}_duplicate_masks_ignored"] += 1
        elif not span:
            STATS[f"{tag}_empty_masks"] += 1
        else:
            extracted[mask] = span

    requested = [m.group() for m in MASK_PATTERN.finditer(scaffold)]
    STATS[f"{tag}_total_masks_requested"] += len(requested)

    if require_all_masks and any(m not in extracted for m in requested):
        STATS[f"{tag}_dropped_incomplete"] += 1
        return None
    if require_all_masks and not requested:
        STATS[f"{tag}_no_masks_in_scaffold"] += 1
        return None

    rebuilt = scaffold
    fulfilled = 0
    for mask in requested:
        if mask in extracted:
            rebuilt = rebuilt.replace(mask, extracted[mask])
            fulfilled += 1
        else:
            rebuilt = rebuilt.replace(mask, "")

    STATS[f"{tag}_total_masks_fulfilled"] += fulfilled
    if requested:
        if fulfilled == len(requested):
            STATS[f"{tag}_sequences_complete"] += 1
        elif fulfilled == 0:
            STATS[f"{tag}_sequences_failed"] += 1
        else:
            STATS[f"{tag}_sequences_partial"] += 1
            STATS[f"{tag}_masks_deleted_from_scaffold"] += len(requested) - fulfilled

    cleaned = sanitize_residues(residues_only(rebuilt))
    return cleaned or None


# ---------------------------------------------------------------------------
# reading design.txt
# ---------------------------------------------------------------------------

def read_design_txt(path):
    """Yield (orig_id, raw_sequence) from a '<id>\\t<seq>' file."""
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                STATS["malformed_lines_skipped"] += 1
                continue
            try:
                yield int(parts[0]), parts[1]
            except ValueError:
                STATS["malformed_lines_skipped"] += 1


# ---------------------------------------------------------------------------
# mode: stage1
# ---------------------------------------------------------------------------

def build_tasks_stage1(args):
    """Route mixed AR/infilling rows into per-cluster output folders."""
    print("Reading reference JSONL for cluster and task_type ...")
    meta = {}
    missing_cluster = 0
    with open(args.reference_jsonl) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            cluster = (row.get("metadata") or {}).get("cluster")
            if cluster is None:
                missing_cluster += 1
            meta[idx] = {
                "cluster_id": f"cluster_{cluster}" if cluster is not None else "cluster_unknown",
                "task_type": row.get("task_type", "autoregressive"),
            }
    print(f"  {len(meta):,} reference rows")
    if missing_cluster:
        print(f"  WARNING: {missing_cluster:,} rows have no metadata.cluster; "
              f"those designs cannot be scored against a cluster.")

    tasks = []
    for orig_id, raw in read_design_txt(args.design_txt):
        info = meta.get(orig_id)
        if info is None:
            STATS["unmapped_generation_ids"] += 1
            continue

        task_type = info["task_type"]
        if task_type == "infilling":
            seq = parse_infilling(raw, STAGE1_MAX_INFILL, require_all_masks=False)
            prefix = "infill"
        else:
            seq = parse_autoregressive(raw)
            prefix = "ar"

        if not seq:
            STATS["dropped_empty_generations"] += 1
            continue

        rel = os.path.join(task_type, info["cluster_id"],
                           f"instructpro2_{prefix}_{orig_id:05d}.pdb")
        tasks.append({
            "task_type": task_type,
            "cluster_id": info["cluster_id"],
            "orig_id": orig_id,
            "sid": f"instructpro2_{prefix}_{orig_id:05d}",
            "original_design_seq": raw,
            "design_seq": seq,
            "out_rel": rel,
            "out_path": os.path.join(args.output_dir, rel),
        })
    return tasks


# ---------------------------------------------------------------------------
# mode: downstream
# ---------------------------------------------------------------------------

def build_tasks_downstream(args):
    """One design.txt per dataset subdirectory of --checkpoint_dir."""
    tasks = []
    per_dataset = {d: {"ok": 0, "dropped": 0} for d in args.datasets}

    for dataset in args.datasets:
        design_txt = os.path.join(args.checkpoint_dir, dataset, "design.txt")
        if not os.path.exists(design_txt):
            print(f"  skipping {dataset}: no design.txt at {design_txt}")
            continue

        is_antibody = "antibody" in dataset
        for orig_id, raw in read_design_txt(design_txt):
            if is_antibody:
                seq = parse_infilling(raw, HCDR_MAX_INFILL,
                                      require_all_masks=True, tag="hcdr")
            else:
                seq = parse_autoregressive(raw)

            if not seq:
                per_dataset[dataset]["dropped"] += 1
                continue

            sid = f"instructpro2_{dataset}_{orig_id:04d}"
            rel = os.path.join(dataset, f"{sid}.pdb")
            tasks.append({
                "dataset": dataset,
                "orig_id": orig_id,
                "sid": sid,
                "original_design_seq": raw,
                "design_seq": seq,
                "out_rel": rel,
                "out_path": os.path.join(args.output_dir, rel),
            })
            per_dataset[dataset]["ok"] += 1

    for dataset, counts in per_dataset.items():
        print(f"  {dataset:<28s} {counts['ok']:>6,} folded  "
              f"{counts['dropped']:>5,} dropped")
    return tasks


# ---------------------------------------------------------------------------
# folding
# ---------------------------------------------------------------------------

def fold(tasks, args):
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    from tqdm import tqdm

    if not torch.cuda.is_available():
        sys.exit("No CUDA device available; ESMFold needs a GPU.")

    device = torch.device("cuda")
    print("\nLoading ESMFold ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.to(device)
    model.trunk.set_chunk_size(args.chunk_size)
    model.eval()

    todo = [t for t in tasks if not os.path.exists(t["out_path"])]
    # Shortest first so each batch pads as little as possible.
    todo.sort(key=lambda t: len(t["design_seq"]))
    print(f"{len(todo):,} of {len(tasks):,} designs need folding "
          f"({len(tasks) - len(todo):,} already on disk).")

    for t in todo:
        os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)

    def fold_one(task, chunk):
        """Fold a single design at a given trunk chunk size. True on success."""
        model.trunk.set_chunk_size(chunk)
        try:
            enc = tokenizer([task["design_seq"]], return_tensors="pt",
                            add_special_tokens=False, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            pdb = model.output_to_pdb(model(**enc))[0]
            with open(task["out_path"], "w") as f:
                f.write(pdb)
            return True
        except (RuntimeError, ValueError):
            return False
        finally:
            model.trunk.set_chunk_size(args.chunk_size)

    done = failed = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="Folding"):
            batch = todo[i:i + args.batch_size]
            seqs = [t["design_seq"] for t in batch]

            try:
                enc = tokenizer(seqs, return_tensors="pt",
                                add_special_tokens=False, padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                for task, pdb in zip(batch, model.output_to_pdb(model(**enc))):
                    with open(task["out_path"], "w") as f:
                        f.write(pdb)
                done += len(batch)
                continue
            except (RuntimeError, ValueError) as err:
                oom = "out of memory" in str(err).lower()
                longest = max(len(s) for s in seqs)
                print(f"\n{'OOM' if oom else 'Error'} on a batch of "
                      f"{len(batch)} (longest {longest}): {str(err)[:120]}")
                print("  retrying one at a time ...")
                torch.cuda.empty_cache()

            # Per-sequence retry, escalating the chunking as memory allows.
            for task in batch:
                if os.path.exists(task["out_path"]):
                    continue
                if fold_one(task, 64) or fold_one(task, 16):
                    done += 1
                else:
                    failed += 1
                    STATS["fold_failures"] += 1
                    print(f"  could not fold {task['sid']} "
                          f"(length {len(task['design_seq'])})")
                torch.cuda.empty_cache()

    print(f"\nFolded {done:,} designs; {failed:,} failed.")


# ---------------------------------------------------------------------------

def report_stats():
    print("\n--- parsing diagnostics " + "-" * 44)
    if not STATS:
        print("  no anomalies")
    else:
        for key in sorted(STATS):
            print(f"  {key:<48s} {STATS[key]:,}")
    print("-" * 68)


def get_args():
    p = argparse.ArgumentParser(
        description="Fold InstructPro2 generations with ESMFold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", required=True, choices=["stage1", "downstream"])
    p.add_argument("--output_dir", required=True,
                   help="root for the generated PDBs")
    p.add_argument("--mapped_csv", default=None,
                   help="where to write the mapping CSV "
                        "(default: <output_dir>/mapped_<run_name>.csv)")
    p.add_argument("--run_name", default="eval",
                   help="label used in the default mapping CSV name")

    # stage1
    p.add_argument("--design_txt", help="[stage1] design.txt to fold")
    p.add_argument("--reference_jsonl",
                   help="[stage1] test split providing metadata.cluster and task_type")

    # downstream
    p.add_argument("--checkpoint_dir",
                   help="[downstream] directory holding one subdir per dataset")
    p.add_argument("--datasets", default="test_ligand,test_ligand_unseen",
                   help="[downstream] comma-separated dataset subdirectories")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=None,
                   help="ESMFold trunk chunk size; None is fastest, "
                        "lower values use less memory")
    p.add_argument("--no_fold", action="store_true",
                   help="parse and write the mapping CSV, then stop "
                        "(no GPU needed; useful for checking parse rates)")

    args = p.parse_args()

    if args.mode == "stage1":
        missing = [f for f in ("design_txt", "reference_jsonl")
                   if not getattr(args, f)]
        if missing:
            p.error("--mode stage1 requires " + " and ".join("--" + m for m in missing))
    else:
        if not args.checkpoint_dir:
            p.error("--mode downstream requires --checkpoint_dir")
        args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.mapped_csv is None:
        args.mapped_csv = os.path.join(args.output_dir, f"mapped_{args.run_name}.csv")
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing generations ({args.mode}) ...")
    tasks = (build_tasks_stage1(args) if args.mode == "stage1"
             else build_tasks_downstream(args))
    report_stats()

    if not tasks:
        sys.exit("No foldable designs were parsed. Check design.txt and the "
                 "generation flags that produced it.")

    os.makedirs(os.path.dirname(os.path.abspath(args.mapped_csv)), exist_ok=True)
    pd.DataFrame(tasks).to_csv(args.mapped_csv, index=False)
    print(f"\n{len(tasks):,} designs mapped -> {args.mapped_csv}")

    if args.no_fold:
        print("--no_fold set; stopping before ESMFold.")
        return

    fold(tasks, args)
    report_stats()


if __name__ == "__main__":
    main()