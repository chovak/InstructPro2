#!/usr/bin/env python3
"""Structural evaluation of folded designs against ground-truth PDBs, via PyMOL.

Reads the mapping CSV written by fold_instructpro2_downstream_output.py, aligns
each design against every matching target structure, and reports refined
(core-matching) and global (topology-matching) RMSD plus pLDDT.
"""

import os
import glob
import json
import argparse
import pandas as pd
import pymol
from pymol import cmd


def get_args():
    parser = argparse.ArgumentParser(description="Structural evaluation pipeline using PyMOL.")
    parser.add_argument("--mapped_csv", required=True,
                        help="Path to the generation mapping CSV from the ESMFold script")
    parser.add_argument("--targets_dir", required=True,
                        help="Directory containing target/ground-truth PDBs")
    parser.add_argument("--output_csv", required=True,
                        help="Where to save the final evaluation metrics CSV")
    parser.add_argument("--run_name", default="Evaluation Run",
                        help="Name to display in the summary printout")
    parser.add_argument("--antibody_targets_dir", default=None,
                        help="Ground-truth directory for antibody datasets. Defaults to "
                             "<targets_dir>/<dataset>, same as every other dataset; set this "
                             "only if the antibody PDBs live somewhere else.")
    return parser.parse_args()


def get_avg_plddt(obj_name):
    """Extracts average pLDDT (stored in B-factors by ESMFold)."""
    b_factors = []
    cmd.iterate(f"{obj_name} and name CA", "b_factors.append(b)", space={'b_factors': b_factors})
    if not b_factors:
        return 0.0
    return round(sum(b_factors) / len(b_factors), 2)


def calculate_metrics(design_path, target_path):
    """Calculates refined RMSD, global RMSD, and target pLDDT."""
    try:
        cmd.reinitialize()
        cmd.load(design_path, "design_obj")
        cmd.load(target_path, "target_obj")

        target_plddt = get_avg_plddt("target_obj")

        # Refined RMSD (core matching): outlier-rejecting superposition.
        ref_res = cmd.align("design_obj", "target_obj", cycles=5, cutoff=2.0)

        # Global RMSD (topology matching): no outlier rejection.
        glb_res = cmd.super("design_obj", "target_obj", cycles=0)

        return {
            "target_plddt": target_plddt,
            "ref_rmsd": round(ref_res[0], 3),
            "ref_atoms": int(ref_res[1]),
            "glb_rmsd": round(glb_res[0], 3),
            "glb_atoms": int(glb_res[1])
        }
    except Exception as e:
        return {"error": str(e)}


def find_target_files(args, dataset, orig_id, padded_id):
    """Locate ground-truth PDBs for one design.

    Looks in a dataset-specific subfolder first, then falls back to the root of
    targets_dir, trying both the zero-padded and unpadded id.
    """
    if "antibody" in dataset and args.antibody_targets_dir:
        target_dataset_dir = args.antibody_targets_dir
    else:
        target_dataset_dir = os.path.join(args.targets_dir, dataset)

    for directory in (target_dataset_dir, args.targets_dir):
        if not os.path.exists(directory):
            continue
        for needle in (padded_id, orig_id):
            hits = glob.glob(os.path.join(directory, f"*{needle}*.pdb"))
            if hits:
                return hits
    return []


def main():
    args = get_args()

    print(f"Initializing PyMOL for {args.run_name}...")
    pymol.finish_launching(['pymol', '-c', '-q'])

    print(f"Loading mapped generations from {args.mapped_csv}...")
    df_mapped = pd.read_csv(args.mapped_csv)
    master_data = []

    for i, row in df_mapped.iterrows():
        dataset = row['dataset']
        orig_id = row['orig_id']
        padded_id = f"{int(orig_id):04d}"
        design_path = row['out_path']

        if not os.path.exists(design_path):
            continue

        cmd.reinitialize()
        cmd.load(design_path, "design_obj")
        design_plddt = get_avg_plddt("design_obj")

        target_files = find_target_files(args, dataset, orig_id, padded_id)

        # If no target found, log the generation metrics anyway.
        if not target_files:
            master_data.append({
                "dataset": dataset,
                "orig_id": orig_id,
                "baseline_plddt": design_plddt,
                "avg_target_plddt": None,
                "min_refined_rmsd": None,
                "best_target_for_refined": None,
                "min_global_rmsd": None,
                "best_target_for_global": None,
                "status": "Missing Target PDB"
            })
            continue

        cluster_comparisons = {}
        min_ref_val = float('inf')
        best_ref_target = None
        min_glb_val = float('inf')
        best_glb_target = None

        # Compare the design against all matching targets (clusters may have
        # several representative structures).
        for t_path in target_files:
            t_id = os.path.basename(t_path).replace('.pdb', '')
            metrics = calculate_metrics(design_path, t_path)

            if "error" not in metrics:
                cluster_comparisons[t_id] = metrics

                if metrics["ref_rmsd"] < min_ref_val:
                    min_ref_val = metrics["ref_rmsd"]
                    best_ref_target = t_id

                if metrics["glb_rmsd"] < min_glb_val:
                    min_glb_val = metrics["glb_rmsd"]
                    best_glb_target = t_id

        if cluster_comparisons:
            # Per-target pLDDT, averaged over all matched targets for this
            # design: visibility into target-structure confidence, not just
            # the design's own pLDDT.
            avg_target_plddt = round(
                sum(m["target_plddt"] for m in cluster_comparisons.values()) / len(cluster_comparisons), 2
            )

            master_data.append({
                "dataset": dataset,
                "orig_id": orig_id,
                "baseline_plddt": design_plddt,
                "avg_target_plddt": avg_target_plddt,
                "min_refined_rmsd": min_ref_val,
                "best_target_for_refined": best_ref_target,
                "min_global_rmsd": min_glb_val,
                "best_target_for_global": best_glb_target,
                "status": "Evaluated",
                "all_target_scores_json": json.dumps(cluster_comparisons)
            })

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(df_mapped)}] generations evaluated...")

    df = pd.DataFrame(master_data)

    out_parent = os.path.dirname(args.output_csv)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 50)
    print(f"Evaluation complete. Saved {len(df)} records to {args.output_csv}")
    print("=" * 50)

    # --- summary statistics --------------------------------------------------
    evaluated_df = df[df['status'] == "Evaluated"]
    total = len(evaluated_df)

    if total == 0:
        print("\nNo data was successfully evaluated against target PDBs. Check --targets_dir.")
        return

    print(f"\n{args.run_name.upper()} RESULTS")
    print("-" * 60)
    print(f"{'Metric':<30} | {'Value'}")
    print("-" * 60)
    print(f"{'Total Evaluated':<30} | {total} / {len(df)} total rows")
    print(f"{'Overall Avg pLDDT (Design)':<30} | {evaluated_df['baseline_plddt'].mean():.2f}")
    print(f"{'Overall Avg pLDDT (Target)':<30} | {evaluated_df['avg_target_plddt'].mean():.2f}")
    print(f"{'Overall Avg Min Global RMSD':<30} | {evaluated_df['min_global_rmsd'].mean():.2f} A")
    print(f"{'Overall Avg Min Refined RMSD':<30} | {evaluated_df['min_refined_rmsd'].mean():.2f} A")

    succ_glb = (evaluated_df['min_global_rmsd'] < 2.0).sum()
    succ_ref = (evaluated_df['min_refined_rmsd'] < 2.0).sum()
    print(f"{'Overall Global < 2.0A':<30} | {succ_glb} ({succ_glb/total*100:.1f}%)")
    print(f"{'Overall Refined < 2.0A':<30} | {succ_ref} ({succ_ref/total*100:.1f}%)")
    print("-" * 60)

    print("\nBREAKDOWN BY DATASET")
    print("-" * 60)
    for name, group in evaluated_df.groupby('dataset'):
        g_total = len(group)
        g_succ_ref = (group['min_refined_rmsd'] < 2.0).sum()
        g_succ_glb = (group['min_global_rmsd'] < 2.0).sum()
        print(f"Dataset: {name.upper()} (n={g_total})")
        print(f"   Avg pLDDT (Design): {group['baseline_plddt'].mean():.2f}")
        print(f"   Avg pLDDT (Target): {group['avg_target_plddt'].mean():.2f}")
        print(f"   Avg Refined RMSD:   {group['min_refined_rmsd'].mean():.2f} A")
        print(f"   Avg Global RMSD:    {group['min_global_rmsd'].mean():.2f} A")
        print(f"   Success Refined (< 2.0A): {g_succ_ref} ({g_succ_ref/g_total*100:.1f}%)")
        print(f"   Success Global (< 2.0A):  {g_succ_glb} ({g_succ_glb/g_total*100:.1f}%)")
    print("-" * 60)


if __name__ == "__main__":
    main()
