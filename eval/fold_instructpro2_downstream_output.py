#!/usr/bin/env python3
"""Fold InstructPro2 downstream generations with ESMFold.

Parses design.txt from each dataset directory, reconstructs full sequences
(antibody heavy chains are rebuilt from their HCDR infill spans), folds them
with ESMFold, and writes one PDB per design plus a mapping CSV consumed by
evaluate_structures.py.
"""

import os
import re
import json
import argparse

import torch
import pandas as pd
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm

# Datasets processed when --datasets is not given.
DEFAULT_DATASETS = ["test_antibody_v2", "test_ligand", "test_ligand_unseen"]

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids


def sanitize_residues(seq):
    """Replace non-standard amino acid codes (Z, B, U, O, ...) with X, since
    the ESMFold tokenizer's vocabulary may not include them."""
    return ''.join(c if c in VALID_RESIDUES else 'X' for c in seq)


def parse_datasets_arg(value):
    """Accepts a JSON/Python-style list as one string, or a comma-separated list."""
    value = value.strip()
    if value.startswith('['):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # Allow single quotes (Python-style) instead of JSON double quotes.
            parsed = json.loads(value.replace("'", '"'))
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError("--datasets list must be a list of strings")
        return [str(d).strip() for d in parsed]
    return [d.strip() for d in value.split(',') if d.strip()]


def get_args():
    parser = argparse.ArgumentParser(description="Fold InstructPro2 downstream outputs using ESMFold.")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Input directory containing the dataset folders")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save generated PDBs and the mapping CSV")
    parser.add_argument("--run_name", required=True,
                        help="Name of the run (used for naming the mapping CSV)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for ESMFold (default: 16)")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Chunk size for the ESMFold trunk (default: None)")
    parser.add_argument("--datasets", type=parse_datasets_arg, default=DEFAULT_DATASETS,
                        help='Datasets to process: a JSON/Python list, e.g. '
                             '\'["test_ligand", "test_ligand_unseen"]\', or comma-separated, '
                             'e.g. test_ligand,test_ligand_unseen')
    return parser.parse_args()


def parse_antibody_sequence(raw_seq):
    """Rebuild an antibody heavy chain from its generated HCDR infill spans.

    The generation has the form SCAFFOLD<SEP><M1>hcdr1<M2>hcdr2<M3>hcdr3; each
    span is capped at 35 residues and substituted back into the scaffold.
    Returns None if the <SEP> marker or any HCDR span is missing.
    """
    sep_splits = raw_seq.split('<SEP>')
    if len(sep_splits) < 2:
        return None  # missing SEP tag

    lhs = sep_splits[0]
    rhs = '<SEP>'.join(sep_splits[1:])

    extracted_tags = {'<M1>': '', '<M2>': '', '<M3>': ''}
    tags_found = list(re.finditer(r'<M[123]>', rhs))

    for i, match in enumerate(tags_found):
        tag = match.group()
        start_idx = match.end()
        end_idx = tags_found[i + 1].start() if i + 1 < len(tags_found) else len(rhs)
        extracted_tags[tag] = rhs[start_idx:end_idx][:35]  # enforce length cap

    if any(val == '' for val in extracted_tags.values()):
        return None  # missing an HCDR span

    reconstructed_hc = lhs
    for tag, span in extracted_tags.items():
        reconstructed_hc = reconstructed_hc.replace(tag, span)

    cleaned = re.sub(r'[^A-Z]', '', reconstructed_hc)
    return sanitize_residues(cleaned)


def parse_standard_sequence(raw_seq):
    """Clean a full-sequence generation and break degenerate repeat loops.

    A run of 20+ repeats of a short k-mer is a known greedy-decoding failure
    mode; it is collapsed to 5 repeats so ESMFold is not handed a pathological
    input.
    """
    cleaned = re.sub(r'[^A-Z]', '', raw_seq)
    cleaned = sanitize_residues(cleaned)

    repeat_pattern = re.compile(r'([A-Z]{1,20})\1{19,}')
    match = repeat_pattern.search(cleaned)

    if match:
        start_idx = match.start()
        kmer = match.group(1)
        cleaned = cleaned[:start_idx] + (kmer * 5)

    return cleaned


def map_generations(args):
    """Parse design.txt across all dataset directories into fold tasks."""
    print("1. Parsing generation text files across datasets...")
    tasks = []
    stats = {d: {"success": 0, "failed": 0} for d in args.datasets}

    mapped_csv_out = os.path.join(args.output_dir, f"stage2_generation_mapped_{args.run_name}.csv")

    for dataset in args.datasets:
        design_txt = os.path.join(args.checkpoint_dir, dataset, "design.txt")

        if not os.path.exists(design_txt):
            print(f"   Skipping {dataset}: {design_txt} not found.")
            continue

        out_dataset_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(out_dataset_dir, exist_ok=True)

        with open(design_txt, 'r') as f_design:
            for line in f_design:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                orig_id = int(parts[0])
                raw_design = parts[1]

                if "antibody" in dataset:
                    clean_design = parse_antibody_sequence(raw_design)
                else:
                    clean_design = parse_standard_sequence(raw_design)

                if clean_design:
                    padded_id = f"{orig_id:04d}"
                    sid = f"instructpro2_{dataset}_{padded_id}"
                    tasks.append({
                        'dataset': dataset,
                        'orig_id': orig_id,
                        'sid': sid,
                        'original_design_seq': raw_design,
                        'design_seq': clean_design,
                        'out_dir': out_dataset_dir,
                        'out_path': os.path.join(out_dataset_dir, f"{sid}.pdb"),
                    })
                    stats[dataset]["success"] += 1
                else:
                    stats[dataset]["failed"] += 1

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(tasks).to_csv(mapped_csv_out, index=False)

    print(f"   Mapped {len(tasks)} sequences. Saved mapping to {mapped_csv_out}")
    for d, s in stats.items():
        print(f"      - {d}: {s['success']} valid, {s['failed']} failed/dropped")

    return tasks


def main():
    args = get_args()
    print(f"   Using datasets: {args.datasets}")
    tasks = map_generations(args)

    if not tasks:
        print("No valid tasks found. Exiting.")
        return

    if not torch.cuda.is_available():
        print("No GPU found. Exiting.")
        return

    device = torch.device("cuda")
    print("\n2. Loading ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.to(device)
    model.trunk.set_chunk_size(args.chunk_size)
    model.eval()

    tasks_to_run = [t for t in tasks if not os.path.exists(t['out_path'])]

    # Sort by length so batches pad as little as possible.
    tasks_to_run.sort(key=lambda x: len(x['design_seq']))
    print(f"   Found {len(tasks_to_run)} sequences that need folding. Sorted by length.")

    success = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(tasks_to_run), args.batch_size), desc="Folding batches"):
            batch_tasks = tasks_to_run[i:i + args.batch_size]
            seqs = [t['design_seq'] for t in batch_tasks]

            # Should be unreachable given map_generations, but log rather than crash.
            bad = [(t['sid'], t['design_seq']) for t in batch_tasks
                   if not isinstance(t['design_seq'], str) or len(t['design_seq']) == 0]
            if bad:
                print(f"Bad design_seq entries found: {bad}")

            try:
                inputs = tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                pdb_strings = model.output_to_pdb(outputs)

                for task, pdb_str in zip(batch_tasks, pdb_strings):
                    with open(task['out_path'], "w") as f:
                        f.write(pdb_str)
                success += len(batch_tasks)

            except (RuntimeError, ValueError) as e:
                if "out of memory" not in str(e).lower():
                    print(f"\nTokenizer/RuntimeError on batch: {e}")
                    print(f"   Sequences in batch: {[(t['sid'], t['design_seq']) for t in batch_tasks]}")
                    continue

                max_len = max(len(s) for s in seqs)
                print(f"\nOOM on batch (size {len(batch_tasks)}, max len {max_len}). "
                      f"Falling back to one-by-one with chunking...")
                torch.cuda.empty_cache()

                model.trunk.set_chunk_size(64)
                for task in batch_tasks:
                    if os.path.exists(task['out_path']):
                        continue
                    try:
                        single_input = tokenizer([task['design_seq']], return_tensors="pt",
                                                 add_special_tokens=False, padding=True)
                        single_input = {k: v.to(device) for k, v in single_input.items()}

                        out = model(**single_input)
                        pdb = model.output_to_pdb(out)[0]
                        with open(task['out_path'], "w") as f:
                            f.write(pdb)
                        success += 1
                    except RuntimeError as e2:
                        if "out of memory" not in str(e2).lower():
                            print(f"\nError on {task['sid']}: {e2}")
                            continue
                        print(f"\nStill OOM for {task['sid']}. Trying maximum chunking (16)...")
                        torch.cuda.empty_cache()
                        try:
                            model.trunk.set_chunk_size(16)
                            out = model(**single_input)
                            pdb = model.output_to_pdb(out)[0]
                            with open(task['out_path'], "w") as f:
                                f.write(pdb)
                            success += 1
                        except Exception as e3:
                            print(f"\nFinal failure on {task['sid']}: {e3}")
                    finally:
                        model.trunk.set_chunk_size(64)

                model.trunk.set_chunk_size(args.chunk_size)  # reset for the next batch

    print(f"\nFinished. Successfully folded {success}/{len(tasks_to_run)} sequences.")


if __name__ == "__main__":
    main()
