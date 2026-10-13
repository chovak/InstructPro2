#!/usr/bin/env python3
"""Prepare a fairseq checkpoint for public release.

A training checkpoint carries far more than the weights: the Adam optimizer
state (two momentum tensors per parameter, so roughly 2x the model size), the
dataloader position, and the full training config -- which embeds the absolute
paths, save directories and W&B run names from the machine it was trained on.

This script drops everything that is not needed to load the model for
inference, and blanks the config fields that would leak local paths.

    python scripts/strip_checkpoint.py checkpoint_last.pt instructpro2_stage1.pt
    python scripts/strip_checkpoint.py --dry-run checkpoint_last.pt

Several at once, writing into a directory:

    python scripts/strip_checkpoint.py -o released/ run1/checkpoint_best.pt run2/*.pt

Verify afterwards by loading the stripped file and running generation; the
model weights are untouched, so results should be identical.
"""

import argparse
import os
import sys

import torch

# Dropped entirely: not needed to run the model.
DROP_KEYS = [
    "last_optimizer_state",   # Adam moments -- the bulk of the file
    "optimizer_history",      # per-update optimizer bookkeeping
    "criterion",              # criterion state, rebuilt from config
]

# Config fields that name a location on the training machine. Blanked rather
# than deleted, so the config still validates when the checkpoint is loaded.
PATH_FIELDS = {
    "checkpoint": ["save_dir", "restore_file", "finetune_from_model",
                   "checkpoint_suffix"],
    "common": ["wandb_project", "tensorboard_logdir", "log_file",
               "azureml_logging", "user_dir"],
    "task": ["data", "pretrained_model_path"],
    "model": ["pretrained_model_path"],
    "distributed_training": ["distributed_init_method"],
}


def blank_paths(cfg, report):
    """Blank location-bearing config fields in place."""
    for section, fields in PATH_FIELDS.items():
        node = cfg.get(section) if hasattr(cfg, "get") else getattr(cfg, section, None)
        if node is None:
            continue
        for field in fields:
            try:
                value = node[field] if hasattr(node, "__getitem__") else getattr(node, field, None)
            except Exception:
                continue
            if isinstance(value, str) and value:
                report.append(f"    {section}.{field} = {value!r}")
                try:
                    node[field] = ""
                except Exception:
                    try:
                        setattr(node, field, "")
                    except Exception:
                        pass


def strip_one(in_path, out_path, dry_run):
    print(f"Loading {in_path} ...")
    state = torch.load(in_path, map_location="cpu", weights_only=False)
    size_before = os.path.getsize(in_path)

    print("\nRemoving:")
    for key in DROP_KEYS:
        if key in state:
            print(f"  - {key}")
            if not dry_run:
                del state[key]
        else:
            print(f"  . {key} (not present)")

    # extra_state holds the dataloader position and epoch bookkeeping. Keep the
    # training metadata that documents the checkpoint, drop the iterator.
    extra = state.get("extra_state")
    if isinstance(extra, dict):
        for key in ["train_iterator", "previous_training_time"]:
            if key in extra:
                print(f"  - extra_state.{key}")
                if not dry_run:
                    del extra[key]

    report = []
    if state.get("cfg") is not None:
        blank_paths(state["cfg"], report)
    if report:
        print("\nBlanking local paths in cfg:")
        for line in report:
            print(line)

    if "model" not in state:
        print("  ERROR: no 'model' key -- is this a fairseq checkpoint? Skipping.")
        return False
    n_params = sum(v.numel() for v in state["model"].values() if hasattr(v, "numel"))
    print(f"\nModel weights kept: {len(state['model'])} tensors, {n_params/1e9:.2f}B parameters")

    if dry_run:
        print("\n(dry run -- nothing written)")
        return True

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(state, out_path)
    size_after = os.path.getsize(out_path)
    print(f"\nWrote {out_path}")
    print(f"  {size_before/1e9:.2f} GB -> {size_after/1e9:.2f} GB "
          f"({(1 - size_after/size_before)*100:.0f}% smaller)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Strip fairseq checkpoints for release.")
    parser.add_argument("inputs", nargs="+", help="training checkpoint(s) (.pt)")
    parser.add_argument("output", nargs="?",
                        help="output path, when exactly one input is given")
    parser.add_argument("-o", "--out-dir",
                        help="directory to write stripped checkpoints into. Each keeps its "
                             "own filename; when two inputs share a filename, the parent "
                             "directory name is prefixed so neither is overwritten.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be removed without writing anything")
    args = parser.parse_args()

    inputs = list(args.inputs)
    if args.output:
        inputs.append(args.output)
        args.output = None

    # With neither -o nor --dry-run, the two-argument form is
    # "<input> <output>", so the trailing path is the destination.
    if not args.out_dir and not args.dry_run and len(inputs) == 2:
        args.output = inputs.pop()

    if not args.dry_run and not args.out_dir and not args.output:
        sys.exit("Give an output path, pass -o OUT_DIR, or pass --dry-run.")
    if args.output and len(inputs) > 1:
        sys.exit("A single output path only works with one input. Use -o OUT_DIR instead.")

    # Disambiguate inputs that share a basename (checkpoint_last.pt is common).
    names = {}
    for p in inputs:
        names.setdefault(os.path.basename(p), []).append(p)

    def out_path_for(p):
        if args.output:
            return args.output
        base = os.path.basename(p)
        if len(names[base]) > 1:
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            base = f"{parent}_{base}"
        return os.path.join(args.out_dir, base)

    ok = 0
    for i, p in enumerate(inputs):
        if len(inputs) > 1:
            print(f"\n{'='*70}\n[{i+1}/{len(inputs)}] {p}\n{'='*70}")
        if strip_one(p, None if args.dry_run else out_path_for(p), args.dry_run):
            ok += 1

    if len(inputs) > 1:
        print(f"\n{ok}/{len(inputs)} checkpoints processed.")


if __name__ == "__main__":
    main()
