#!/usr/bin/env python3
"""Migrate a released InstructPro2 checkpoint to the current model code.

The Stage 2 checkpoints were trained when DNA was a fourth modality:

    old code:  modality_type_embedding = nn.Embedding(4, n_embd)
               # 0 = ligand, 1 = dna, 2 = antigen, 3 = padding/extra
    now:       MODALITY_LIGAND = 0, MODALITY_ANTIGEN = 1, NUM_MODALITIES = 2

so the checkpoint's [4, n_embd] embedding no longer matches the [2, n_embd]
the model builds, and loading fails. A shape mismatch is fatal even under
strict=False, which is why the released Stage 2 weights cannot be loaded by
the released code at all.

This rewrites the checkpoint:

  * modality_type_embedding.weight  ->  rows [0, 2] in that order, so
    ligand stays index 0 and antigen moves from 2 to 1.
  * dna tensors (dna_encoder.*, dna_mapping_layer.*, modality_scale.dna)
    are dropped. The current model never builds them; they load harmlessly
    under strict=False but add weight to the file and confuse readers.
  * with --fix-config, stale names in the saved config are corrected:
    criterion instructpro_loss -> instructpro2_loss, and the arch name,
    which read 700M for what is a 3B model.

Everything else, including every trained weight, is copied byte for byte.

    # look first
    python migrate_checkpoint.py --dry-run instructpro2_stage2_ligand.pt

    # write the migrated file
    python migrate_checkpoint.py instructpro2_stage2_ligand.pt \
        -o instructpro2_stage2_ligand_fixed.pt --fix-config

Verify by loading the result and generating; the kept weights are identical,
so output should match whatever the old code produced.
"""

import argparse
import os
import sys

import torch

EMBEDDING_KEY = "modality_type_embedding.weight"

# Old row -> meaning. The current model keeps ligand and antigen, in the
# order the current constants define.
OLD_ROWS = {0: "ligand", 1: "dna", 2: "antigen", 3: "padding/extra"}
KEEP_ROWS = [0, 2]           # ligand, antigen
NEW_ROWS = ["ligand (0)", "antigen (1)"]

# Config values that lag behind renames in the code. Each entry is
# (section, field, {old: new}). A value not in the mapping is left alone.
#
# These matter because the saved config supplies any flag the command line
# does not. architecture_mode is the dangerous one: a checkpoint still
# saying "flamingo_full" is unusable unless the caller happens to pass
# --architecture-mode.
ARCH_NAMES = {
    "language_guided_protein_design_model_700M":
        "language_guided_protein_design_model_3B",
}
CRITERION_NAMES = {"instructpro_loss": "instructpro2_loss"}
DATASET_IMPLS = {
    "instructpro": "instructpro2",
    "instructpro_ligand": "instructpro2_ligand",
    "instructpro_antibody": "instructpro2_antibody",
}
# Old freezing-mode names -> the modes the current code registers.
ARCHITECTURE_MODES = {
    "default": "stage1",
    "flamingo_full": "stage2",
    "flamingo": "stage2_frozen_decoder",
    "flamingo_partial": "stage2_partial",
    "flamingo_crossattn_only": "stage2_adapters_only",
}

CONFIG_FIXES = [
    ("criterion", "_name", CRITERION_NAMES),
    ("model", "criterion", CRITERION_NAMES),
    ("model", "_name", ARCH_NAMES),
    ("model", "arch", ARCH_NAMES),
    ("model", "architecture_mode", ARCHITECTURE_MODES),
    ("model", "dataset_impl", DATASET_IMPLS),
    ("task", "dataset_impl", DATASET_IMPLS),
]


def is_dna_key(key):
    return "dna" in key.lower()


def cfg_get(node, field):
    try:
        return node[field] if hasattr(node, "__getitem__") else getattr(node, field, None)
    except Exception:
        return None


def cfg_set(node, field, value):
    try:
        node[field] = value
        return True
    except Exception:
        try:
            setattr(node, field, value)
            return True
        except Exception:
            return False


def migrate(in_path, out_path, fix_config, dry_run):
    print("=" * 74)
    print(os.path.basename(in_path))
    print("=" * 74)

    state = torch.load(in_path, map_location="cpu", weights_only=False)
    model = state.get("model")
    if model is None:
        print("  no 'model' key -- not a fairseq checkpoint. Skipping.")
        return False

    changed = False

    # --- the embedding ----------------------------------------------------
    emb = model.get(EMBEDDING_KEY)
    if emb is None:
        print(f"  {EMBEDDING_KEY}: absent, nothing to slice")
    elif emb.shape[0] == len(KEEP_ROWS):
        print(f"  {EMBEDDING_KEY}: already {tuple(emb.shape)}, nothing to do")
    elif emb.shape[0] <= max(KEEP_ROWS):
        print(f"  {EMBEDDING_KEY}: {tuple(emb.shape)} has too few rows to "
              f"contain ligand and antigen. Refusing to guess.")
        return False
    else:
        print(f"  {EMBEDDING_KEY}: {tuple(emb.shape)} -> "
              f"({len(KEEP_ROWS)}, {emb.shape[1]})")
        for row in range(emb.shape[0]):
            label = OLD_ROWS.get(row, "?")
            kept = "KEPT" if row in KEEP_ROWS else "dropped"
            norm = emb[row].float().norm().item()
            print(f"      row {row} ({label:<13s}) norm {norm:9.4f}   {kept}")
        print(f"      new order: {', '.join(NEW_ROWS)}")
        if not dry_run:
            model[EMBEDDING_KEY] = emb[KEEP_ROWS].clone()
        changed = True

    # --- dna tensors ------------------------------------------------------
    dna_keys = [k for k in model if is_dna_key(k)]
    if dna_keys:
        n = sum(model[k].numel() for k in dna_keys if hasattr(model[k], "numel"))
        print(f"\n  dropping {len(dna_keys)} DNA tensors ({n / 1e6:.1f}M parameters)")
        shown = sorted(dna_keys)[:6]
        for key in shown:
            print(f"      {key}")
        if len(dna_keys) > len(shown):
            print(f"      ... and {len(dna_keys) - len(shown)} more")
        if not dry_run:
            for key in dna_keys:
                del model[key]
        changed = True
    else:
        print("\n  no DNA tensors present")

    # --- config -----------------------------------------------------------
    cfg = state.get("cfg")
    if fix_config and cfg is not None:
        fixes = []
        for section, field, mapping in CONFIG_FIXES:
            node = cfg_get(cfg, section)
            if node is None:
                continue
            value = cfg_get(node, field)
            new = mapping.get(value) if isinstance(value, str) else None
            if new is not None and new != value:
                fixes.append(f"      {section}.{field}: {value!r} -> {new!r}")
                if not dry_run:
                    cfg_set(node, field, new)
        if fixes:
            print("\n  correcting stale names in the saved config")
            for line in fixes:
                print(line)
            changed = True
        else:
            print("\n  saved config needs no name corrections")

    total = sum(v.numel() for v in model.values() if hasattr(v, "numel"))
    print(f"\n  resulting model: {len(model):,} tensors, {total / 1e9:.2f}B parameters")

    if dry_run:
        print("\n  (dry run -- nothing written)")
        return True
    if not changed:
        print("\n  nothing to change; not writing an output file")
        return True

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(state, out_path)
    before = os.path.getsize(in_path)
    after = os.path.getsize(out_path)
    print(f"\n  wrote {out_path}")
    print(f"  {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Migrate InstructPro2 checkpoints to the current model code.")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("-o", "--output",
                    help="output path for a single input")
    ap.add_argument("-d", "--out-dir",
                    help="directory to write migrated checkpoints into, "
                         "keeping each filename")
    ap.add_argument("--fix-config", action="store_true",
                    help="also correct stale criterion and arch names in the "
                         "saved config")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    if not args.dry_run and not args.output and not args.out_dir:
        sys.exit("Give -o OUTPUT, -d OUT_DIR, or --dry-run.")
    if args.output and len(args.checkpoints) > 1:
        sys.exit("-o takes a single input; use -d OUT_DIR for several.")

    ok = 0
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        out = args.output or (
            os.path.join(args.out_dir, os.path.basename(path))
            if args.out_dir else None)
        if out and os.path.abspath(out) == os.path.abspath(path):
            sys.exit(f"refusing to overwrite the input in place: {path}")
        if migrate(path, out, args.fix_config, args.dry_run):
            ok += 1
        print()

    print(f"{ok}/{len(args.checkpoints)} checkpoints processed.")


if __name__ == "__main__":
    main()