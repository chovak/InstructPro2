#!/usr/bin/env python3
"""Score folded InstructPro2 designs against ground-truth structures.

Replaces two scripts that scored the same way against differently organised
ground truth:
    eval/evaluate_structures.py   (Stage 2: one target PDB per row)
    eval_esmfold_results.py       (Stage 1: a cluster of targets per row)

    --mode row       each design has one target, looked up by row id
    --mode cluster   each design is scored against every member of its
                     cluster, and the best match is kept

Mode is inferred from the mapping CSV when not given: a `cluster_id` column
means cluster mode.

Targets are resolved through targets.csv (built by build_targets_manifest.py)
rather than by globbing filenames, so a design is never silently scored
against the wrong structure. Paths in that manifest are relative to the
structures root, so the tree can be moved or re-downloaded freely.

Metrics, unchanged from both originals:
    ref_rmsd   cmd.align(cycles=5, cutoff=2.0)  outlier-rejecting, core
    glb_rmsd   cmd.super(cycles=0)              no rejection, whole topology
    pLDDT      mean CA B-factor, which is where ESMFold stores it
For a cluster, the lowest RMSD over members is reported along with which
member produced it.

Designs with no ground truth are kept in the output with their pLDDT and a
status saying why, so a run over new inputs still produces a useful table.
With --plddt_only, no targets are read and PyMOL is never imported.

usage:
  python score_structures.py \
      --mapped_csv   $OUT/ligand_pdb/mapped_ligand.csv \
      --targets_csv  $DATA/eval_structures/targets.csv \
      --targets_root $DATA/eval_structures \
      --output_csv   $OUT/ligand_pdb/scores.csv \
      --run_name     "ligand (w_modality)"
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# pLDDT without PyMOL: ESMFold writes it into the B-factor column, which is
# columns 61-66 of an ATOM record. Reading it directly means --plddt_only
# needs nothing but the standard library.
# ---------------------------------------------------------------------------

def mean_plddt(pdb_path):
    total = n = 0.0
    try:
        with open(pdb_path, errors="replace") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        total += float(line[60:66])
                        n += 1
                    except ValueError:
                        pass
    except OSError:
        return None
    return round(total / n, 2) if n else None


# ---------------------------------------------------------------------------
# targets.csv
# ---------------------------------------------------------------------------

def load_targets(path):
    """(split, orig_id) -> {'kind', 'target', 'cluster_id', 'status'}."""
    index = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["split"], int(row["orig_id"]))
            except (KeyError, ValueError):
                continue
            index[key] = row
    return index


def resolve_targets(entry, targets_root):
    """Absolute paths of every ground-truth PDB for one design."""
    if not entry or not entry.get("target"):
        return []
    target = os.path.join(targets_root, entry["target"])
    if os.path.isdir(target):
        return sorted(os.path.join(target, n) for n in os.listdir(target)
                      if n.endswith(".pdb"))
    return [target] if os.path.exists(target) else []


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

class Scorer:
    """Thin wrapper so PyMOL is imported only when targets are scored."""

    def __init__(self):
        import pymol
        from pymol import cmd
        pymol.finish_launching(["pymol", "-qc"])
        self.cmd = cmd

    def compare(self, design_path, target_path):
        cmd = self.cmd
        try:
            cmd.reinitialize()
            cmd.load(design_path, "design_obj")
            cmd.load(target_path, "target_obj")
            ref = cmd.align("design_obj", "target_obj", cycles=5, cutoff=2.0)
            glb = cmd.super("design_obj", "target_obj", cycles=0)
            return {
                "ref_rmsd": round(ref[0], 3),
                "ref_atoms": int(ref[1]),
                "glb_rmsd": round(glb[0], 3),
                "glb_atoms": int(glb[1]),
            }
        except Exception as exc:  # PyMOL raises bare Exception for bad input
            return {"error": str(exc)}


def design_path_for(row, designs_root):
    """Prefer the relative path; fall back to the absolute one."""
    rel = row.get("out_rel")
    if isinstance(rel, str) and rel and designs_root:
        candidate = os.path.join(designs_root, rel)
        if os.path.exists(candidate):
            return candidate
    absolute = row.get("out_path")
    if isinstance(absolute, str) and os.path.exists(absolute):
        return absolute
    if isinstance(rel, str) and rel:
        return os.path.join(designs_root or "", rel)
    return absolute or ""


def main():
    ap = argparse.ArgumentParser(
        description="Score folded designs against ground-truth structures.")
    ap.add_argument("--mapped_csv", required=True,
                    help="mapping CSV written by fold_designs.py")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--targets_csv", default=None,
                    help="targets.csv from the eval_structures tree "
                         "(omit only with --plddt_only)")
    ap.add_argument("--targets_root", default=None,
                    help="eval_structures directory; defaults to the "
                         "directory holding targets.csv")
    ap.add_argument("--designs_root", default=None,
                    help="root for the mapping CSV's out_rel paths; "
                         "defaults to the directory holding the mapping CSV")
    ap.add_argument("--mode", choices=["row", "cluster"], default=None,
                    help="default: cluster if the mapping CSV has cluster_id")
    ap.add_argument("--split", default=None,
                    help="targets.csv split name, when it differs from the "
                         "mapping CSV's dataset column (cluster mode "
                         "defaults to stage1_test)")
    ap.add_argument("--run_name", default="InstructPro2 evaluation")
    ap.add_argument("--plddt_only", action="store_true",
                    help="report pLDDT and stop; no targets, no PyMOL. "
                         "Use this for designs from your own inputs.")
    ap.add_argument("--rmsd_threshold", type=float, default=2.0,
                    help="success cutoff in angstroms (default 2.0)")
    args = ap.parse_args()

    df = pd.read_csv(args.mapped_csv)
    if df.empty:
        sys.exit("mapping CSV is empty; nothing to score.")

    mode = args.mode or ("cluster" if "cluster_id" in df.columns else "row")
    designs_root = args.designs_root or os.path.dirname(
        os.path.abspath(args.mapped_csv))

    targets = {}
    targets_root = None
    if not args.plddt_only:
        if not args.targets_csv:
            sys.exit("--targets_csv is required unless --plddt_only is set.")
        targets = load_targets(args.targets_csv)
        targets_root = args.targets_root or os.path.dirname(
            os.path.abspath(args.targets_csv))

    print("=" * 70)
    print(f"{args.run_name}")
    print("=" * 70)
    print(f"mode     : {mode}")
    print(f"designs  : {len(df):,} rows from {args.mapped_csv}")
    if args.plddt_only:
        print("targets  : none (--plddt_only)")
    else:
        print(f"targets  : {len(targets):,} entries from {args.targets_csv}")

    scorer = None
    records = []
    counts = defaultdict(int)

    for i, row in df.iterrows():
        orig_id = int(row["orig_id"])
        # Group label: the dataset for stage 2, the task type for stage 1.
        group = row.get("dataset") or row.get("task_type") or "all"
        split = args.split or ("stage1_test" if mode == "cluster"
                               else row.get("dataset", ""))

        design = design_path_for(row, designs_root)
        record = {
            "group": group,
            "orig_id": orig_id,
            "sid": row.get("sid", ""),
            "cluster_id": row.get("cluster_id", ""),
            "design_plddt": None,
            "n_targets": 0,
            "min_refined_rmsd": None,
            "best_target_for_refined": None,
            "min_global_rmsd": None,
            "best_target_for_global": None,
            "avg_target_plddt": None,
            "status": "",
            "all_target_scores_json": "{}",
        }

        if not design or not os.path.exists(design):
            record["status"] = "design_pdb_missing"
            counts["design_pdb_missing"] += 1
            records.append(record)
            continue

        record["design_plddt"] = mean_plddt(design)

        if args.plddt_only:
            record["status"] = "plddt_only"
            counts["plddt_only"] += 1
            records.append(record)
            continue

        entry = targets.get((split, orig_id))
        target_files = resolve_targets(entry, targets_root)
        record["n_targets"] = len(target_files)

        if not target_files:
            record["status"] = (entry or {}).get("status") or "no_target"
            counts[record["status"]] += 1
            records.append(record)
            continue

        if scorer is None:
            print("\nstarting PyMOL ...")
            scorer = Scorer()

        comparisons = {}
        best_ref = (float("inf"), None)
        best_glb = (float("inf"), None)
        target_plddts = []

        for target in target_files:
            name = os.path.splitext(os.path.basename(target))[0]
            metrics = scorer.compare(design, target)
            if "error" in metrics:
                continue
            comparisons[name] = metrics
            target_plddts.append(mean_plddt(target) or 0.0)
            if metrics["ref_rmsd"] < best_ref[0]:
                best_ref = (metrics["ref_rmsd"], name)
            if metrics["glb_rmsd"] < best_glb[0]:
                best_glb = (metrics["glb_rmsd"], name)

        if comparisons:
            record.update({
                "min_refined_rmsd": best_ref[0],
                "best_target_for_refined": best_ref[1],
                "min_global_rmsd": best_glb[0],
                "best_target_for_global": best_glb[1],
                "avg_target_plddt": round(sum(target_plddts) / len(target_plddts), 2),
                "status": "scored",
                "all_target_scores_json": json.dumps(comparisons),
            })
            counts["scored"] += 1
        else:
            record["status"] = "alignment_failed"
            counts["alignment_failed"] += 1

        records.append(record)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(df)}] processed")

    out = pd.DataFrame(records)
    parent = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(parent, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"\nwrote {len(out):,} rows to {args.output_csv}")
    for key in sorted(counts):
        print(f"  {key:<22s} {counts[key]:,}")

    summarize(out, args)


def summarize(df, args):
    thr = args.rmsd_threshold

    def block(sub, label):
        total = len(sub)
        if total == 0:
            return
        print(f"\n{label}  (n = {total:,})")
        print("-" * 58)
        plddt = sub["design_plddt"].dropna()
        if len(plddt):
            print(f"  {'mean pLDDT (design)':<34s} {plddt.mean():.2f}")

        scored = sub[sub["status"] == "scored"]
        if scored.empty:
            print("  no designs were scored against a target")
            return
        print(f"  {'scored against ground truth':<34s} {len(scored):,}")
        tp = scored["avg_target_plddt"].dropna()
        if len(tp):
            print(f"  {'mean pLDDT (target)':<34s} {tp.mean():.2f}")
        print(f"  {'mean min refined RMSD':<34s} {scored['min_refined_rmsd'].mean():.2f} A")
        print(f"  {'mean min global RMSD':<34s} {scored['min_global_rmsd'].mean():.2f} A")
        ref_ok = (scored["min_refined_rmsd"] < thr).sum()
        glb_ok = (scored["min_global_rmsd"] < thr).sum()
        n = len(scored)
        print(f"  {f'refined < {thr} A':<34s} {ref_ok:,} ({ref_ok / n * 100:.1f}%)")
        print(f"  {f'global  < {thr} A':<34s} {glb_ok:,} ({glb_ok / n * 100:.1f}%)")

    print("\n" + "=" * 70)
    print(f"{args.run_name.upper()}")
    print("=" * 70)
    block(df, "OVERALL")

    groups = [g for g in df["group"].dropna().unique()]
    if len(groups) > 1:
        for group in sorted(groups):
            block(df[df["group"] == group], str(group).upper())
    print()


if __name__ == "__main__":
    main()