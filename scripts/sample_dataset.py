#!/usr/bin/env python3
"""Build a release-sized sample of an InstructPro2Bench shard.

The full Stage 1 corpus is far too large to distribute, so the released sample
is a subset of one shard, with construction-time metadata pruned.

    # look first: field sizes averaged over the first 1000 rows
    python scripts/sample_dataset.py --inspect shard_0.jsonl

    # 200k rows, pruned, written as train.jsonl
    python scripts/sample_dataset.py shard_0.jsonl sample/train.jsonl -n 200000

Fields the loader requires are never dropped (see
fairseq/data/indexed_dataset.py):

    tokenized_instruction   PubMedBERT token ids -- the model input
    output                  target protein (read as target_protein, then output)

`instruction` is kept by default: it is the human-readable text, and without it
the data cannot be re-tokenized for a different text encoder.
"""

import argparse
import json
import os
import sys

REQUIRED = ["tokenized_instruction", "output", "target_protein"]

# Modality inputs. Absent from Stage 1 rows, required by the Stage 2 readers.
MODALITY = ["ligand", "antigen"]

# Metadata subfields worth keeping: provenance and the InterPro annotations the
# functional evaluation needs. Everything else is construction bookkeeping.
KEEP_METADATA = ["id", "cluster", "interpro_all"]


def inspect(path, n_rows):
    """Report each field's average serialized size over the first n_rows."""
    totals, counts, samples = {}, {}, {}
    rows = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows += 1
            for k, v in d.items():
                totals[k] = totals.get(k, 0) + len(json.dumps(v))
                counts[k] = counts.get(k, 0) + 1
                samples.setdefault(k, v)
            if rows >= n_rows:
                break

    total = sum(totals.values())
    print(f"Averaged over {rows} rows:\n")
    print(f"{'field':28s} {'avg bytes':>10s} {'% of row':>9s}  {'present':>7s}  sample")
    print("-" * 100)
    for k in sorted(totals, key=lambda k: -totals[k]):
        avg = totals[k] / rows
        pct = totals[k] / total * 100
        s = str(samples[k])[:34].replace("\n", " ")
        print(f"{k:28s} {avg:10.0f} {pct:8.1f}%  {counts[k]*100//rows:6d}%  {s}")

    meta = samples.get("metadata")
    if isinstance(meta, dict):
        print(f"\nmetadata subfields: {list(meta.keys())}")
        print(f"  kept by default : {KEEP_METADATA}")
        dropped = [k for k in meta if k not in KEEP_METADATA]
        print(f"  dropped         : {dropped}")
    print(f"\navg row: {total/rows:,.0f} bytes")


def prune(row, keep_instruction, keep_metadata):
    out = {}
    for k in REQUIRED + MODALITY:
        if k in row:
            out[k] = row[k]
    if keep_instruction:
        # Stage 1 shards call it "instruction"; the Stage 2 sets call it "text".
        for k in ("instruction", "text"):
            if k in row:
                out[k] = row[k]
    for k in ("task_type", "modality_type"):
        if k in row:
            out[k] = row[k]
    meta = row.get("metadata")
    if isinstance(meta, dict) and keep_metadata:
        if keep_metadata == ["*"]:
            out["metadata"] = meta
        else:
            kept = {k: meta[k] for k in keep_metadata if k in meta}
            if kept:
                out["metadata"] = kept
    return out


def main():
    p = argparse.ArgumentParser(description="Sample and prune an InstructPro2Bench shard.")
    p.add_argument("input", help="source .jsonl shard")
    p.add_argument("output", nargs="?", help="destination .jsonl")
    p.add_argument("-n", "--num-rows", type=int, default=200000,
                   help="rows to write (default: 200000)")
    p.add_argument("--inspect", action="store_true",
                   help="report field sizes and exit without writing")
    p.add_argument("--inspect-rows", type=int, default=1000,
                   help="rows to average over when inspecting (default: 1000)")
    p.add_argument("--no-prune", action="store_true",
                   help="copy rows verbatim instead of pruning metadata")
    p.add_argument("--drop-instruction", action="store_true",
                   help="also drop the human-readable instruction text")
    p.add_argument("--keep-metadata", default=",".join(KEEP_METADATA),
                   help=f"comma-separated metadata subfields to keep "
                        f"(default: {','.join(KEEP_METADATA)})")
    args = p.parse_args()

    if args.inspect:
        inspect(args.input, args.inspect_rows)
        return
    if not args.output:
        sys.exit("Give an output path, or pass --inspect.")

    keep_metadata = [k for k in args.keep_metadata.split(",") if k]
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    written = skipped = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if written >= args.num_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "tokenized_instruction" not in row or not (
                    "output" in row or "target_protein" in row):
                skipped += 1
                continue
            fout.write(json.dumps(
                row if args.no_prune else prune(row, not args.drop_instruction, keep_metadata)
            ) + "\n")
            written += 1

    size_in = os.path.getsize(args.input)
    size_out = os.path.getsize(args.output)
    print(f"Wrote {written:,} rows to {args.output}")
    if skipped:
        print(f"Skipped {skipped:,} malformed or incomplete rows")
    print(f"  source: {size_in/1e9:.1f} GB    sample: {size_out/1e6:.0f} MB")
    print(f"  avg row: {size_out/max(written,1):,.0f} bytes")


if __name__ == "__main__":
    main()
