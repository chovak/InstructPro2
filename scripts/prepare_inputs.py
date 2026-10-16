#!/usr/bin/env python3
"""Turn plain text into an InstructPro2 input file.

The model does not read sentences at inference: every row must already
carry `tokenized_instruction`, a list of PubMedBERT token ids. This script
produces them with the same tokenizer the model uses, so a description you
write is encoded exactly as the training data was.

This script writes a JSONL the loaders accept, for all three models:

    --task stage1     design a protein from a description
    --task ligand     design a binder, conditioned on SMILES
    --task antibody   fill HCDR1-3 in a scaffold, conditioned on an antigen

Start by looking at how the released data is written, which is also the
quickest way to confirm the tokenizer matches:

    python prepare_inputs.py --inspect $WORK_ROOT/data/stage1/test.jsonl

One design from the command line:

    python prepare_inputs.py --task stage1 \\
        --text "A thermostable bacterial lipase active at alkaline pH." \\
        --out my_designs/test_mine.jsonl

A batch from a CSV with columns text, ligand, antigen, scaffold (only
`text` is required; ligand and antigen may hold several entries separated
by ';'):

    python prepare_inputs.py --task ligand --input designs.csv \\
        --out my_designs/test_ligand_mine.jsonl

Then generate, pointing the scripts at the directory and split name:

    MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_ligand_mine \\
        bash infer.sh

A NOTE ON THE TARGET SEQUENCE
-----------------------------
Every row needs a `target_protein`, because the evaluation path computes a
loss before it generates. For de novo design nobody has one, so a
placeholder goes in; generation starts from the beginning and never reads
it. Its only real effect is length: a row whose target exceeds the model's
1024-token context is skipped entirely, so the placeholder is kept short.

For infilling the target is not a placeholder: the scaffold up to and
including `<SEP>` is the generation prefix, so that part has to be real.
"""

import argparse
import csv
import json
import os
import re
import sys

MASK = re.compile(r"<M([1-9]|[1-2][0-9]|30)>")
TEXT_ENCODER = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Mirrors fairseq/data/indexed_dataset.py.
MAX_ANTIGEN_TOKENS = 1024
MAX_LIGAND_TOKENS = 512
MAX_POSITIONS = 1024

PLACEHOLDER_RESIDUE = "A"


def load_tokenizer():
    """The text tokenizer the model uses.

    The model itself also registers two unused tokens (<FUNCTION> and
    </FUNCTION>) left over from an earlier version and resizes the text
    embedding for them. They are never present in the data, and because
    added tokens take ids past the end of the vocabulary, ordinary text
    encodes identically with or without them. Nothing to do here.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers is required: run this in the InstructPro2 environment.")
    return AutoTokenizer.from_pretrained(TEXT_ENCODER)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def inspect(path, n):
    """Decode released rows so the file's conventions are visible."""
    tok = load_tokenizer()
    print(f"tokenizer: {TEXT_ENCODER}  ({len(tok):,} tokens)\n")

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n or not line.strip():
                break
            row = json.loads(line)
            print("=" * 70)
            print(f"row {i}   keys: {', '.join(sorted(row))}")

            ids = row.get("tokenized_instruction")
            if ids:
                print(f"  tokenized_instruction: {len(ids)} ids, "
                      f"first 12 {ids[:12]}")
                print(f"  decoded: {tok.decode(ids)[:300]}")

            for key in ("text", "instruction"):
                if row.get(key):
                    print(f"  {key}: {str(row[key])[:200]}")

            target = row.get("target_protein") or row.get("output") or ""
            kind = "infilling" if "<SEP>" in target else "full sequence"
            print(f"  target ({kind}, {len(target)} chars): {target[:160]}")

            for key in ("ligand", "antigen"):
                if row.get(key):
                    vals = row[key]
                    print(f"  {key}: {len(vals)} entry(ies), "
                          f"first {str(vals[0])[:80]}")
            if row.get("task_type"):
                print(f"  task_type: {row['task_type']}")
    print("=" * 70)
    print("\nWrite your own instructions in the same style as the decoded")
    print("text above: the model responds to the phrasing it was trained on.")


# ---------------------------------------------------------------------------
# building rows
# ---------------------------------------------------------------------------

def split_multi(value):
    """'a;b' or a list -> ['a', 'b']; empty -> []."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(";") if part.strip()]


def build_row(record, task, tok, args, index):
    text = (record.get("text") or "").strip()
    if not text:
        raise ValueError(f"row {index}: 'text' is required and must not be empty")

    ids = tok(text, truncation=True,
              max_length=args.max_text_tokens)["input_ids"]

    ligand = split_multi(record.get("ligand"))
    antigen = split_multi(record.get("antigen"))
    scaffold = (record.get("scaffold") or "").strip()

    if task == "antibody":
        if not scaffold:
            raise ValueError(
                f"row {index}: --task antibody needs a 'scaffold': the heavy "
                f"chain with <M1>, <M2>, <M3> where the HCDRs belong")
        masks = [m.group() for m in MASK.finditer(scaffold)]
        if not masks:
            raise ValueError(
                f"row {index}: scaffold contains no <M1>..<M30> markers, so "
                f"there is nothing to infill")
        # Everything up to and including <SEP> is the generation prefix, so
        # the target is the scaffold itself; the model writes the spans after.
        target = scaffold if "<SEP>" in scaffold else scaffold + "<SEP>"
        if not antigen:
            print(f"  note: row {index} has no antigen; the design will be "
                  f"unconditioned on structure", file=sys.stderr)
    elif scaffold:
        target = scaffold if "<SEP>" in scaffold else scaffold + "<SEP>"
    else:
        # De novo: a placeholder the generator never reads (see module docs).
        target = PLACEHOLDER_RESIDUE * args.length

    if task == "ligand" and not ligand:
        print(f"  note: row {index} has no ligand SMILES; --task ligand "
              f"without one is just text-conditioned design", file=sys.stderr)

    # The text and the protein share the 1024-position context.
    budget = len(ids) + len(re.sub(r"<[^>]*>", "", target)) + 2
    if budget > MAX_POSITIONS:
        print(f"  warning: row {index} needs about {budget} positions, over "
              f"the {MAX_POSITIONS} limit; this row will be skipped at "
              f"generation. Shorten the text or the scaffold.", file=sys.stderr)

    row = {
        "text": text,
        "tokenized_instruction": ids,
        "target_protein": target,
    }
    if ligand:
        row["ligand"] = ligand
    if antigen:
        row["antigen"] = antigen
    if task == "stage1":
        # The Stage 1 reader accepts either key; `output` matches the
        # released splits, and task_type drives the stage-1 fold step.
        row["output"] = row.pop("target_protein")
        row["task_type"] = "infilling" if "<SEP>" in target else "autoregressive"
    return row


def read_records(args):
    """Records from --input (csv/jsonl) or from the --text flags."""
    if args.input:
        ext = os.path.splitext(args.input)[1].lower()
        with open(args.input, newline="") as f:
            if ext in (".jsonl", ".json"):
                return [json.loads(line) for line in f if line.strip()]
            return list(csv.DictReader(f))

    if not args.text:
        sys.exit("Give --text (repeatable) or --input FILE. "
                 "Use --inspect FILE to look at the released format first.")

    records = []
    for i, text in enumerate(args.text):
        record = {"text": text}
        if args.ligand:
            record["ligand"] = args.ligand[i] if i < len(args.ligand) else args.ligand[-1]
        if args.antigen:
            record["antigen"] = args.antigen[i] if i < len(args.antigen) else args.antigen[-1]
        if args.scaffold:
            record["scaffold"] = args.scaffold[i] if i < len(args.scaffold) else args.scaffold[-1]
        records.append(record)
    return records


def main():
    ap = argparse.ArgumentParser(
        description="Build an InstructPro2 input JSONL from plain text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--inspect", metavar="JSONL",
                    help="decode an existing split and exit")
    ap.add_argument("--inspect-rows", type=int, default=2)

    ap.add_argument("--task", choices=["stage1", "ligand", "antibody"])
    ap.add_argument("--out", help="output .jsonl; its basename is the split name")

    ap.add_argument("--text", action="append",
                    help="one instruction; repeat for several designs")
    ap.add_argument("--ligand", action="append",
                    help="SMILES for the matching --text; ';' separates several")
    ap.add_argument("--antigen", action="append",
                    help="antigen chain for the matching --text; ';' separates several")
    ap.add_argument("--scaffold", action="append",
                    help="sequence with <M1>..<M30> marking spans to infill")
    ap.add_argument("--input", help="CSV or JSONL with text/ligand/antigen/scaffold")

    ap.add_argument("--length", type=int, default=300,
                    help="placeholder target length for de novo design "
                         "(default 300; it only affects length filtering)")
    ap.add_argument("--max-text-tokens", type=int, default=512,
                    help="truncate instructions to this many tokens (default 512)")
    ap.add_argument("--replicates", type=int, default=1, metavar="N",
                    help="write each design N times, so one generation pass "
                         "returns N independent samples. Sampling is "
                         "stochastic and a single sequence is one roll of the "
                         "dice: early stops and low-complexity runs are "
                         "common, so 8-20 replicates and picking the best is "
                         "the normal way to use this model.")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect, args.inspect_rows)
        return

    if not args.task or not args.out:
        sys.exit("--task and --out are required (or use --inspect FILE).")

    records = read_records(args)
    tok = load_tokenizer()

    rows, failures = [], 0
    for i, record in enumerate(records):
        try:
            row = build_row(record, args.task, tok, args, i)
        except ValueError as err:
            print(f"  skipped: {err}", file=sys.stderr)
            failures += 1
            continue
        # Replicates are identical rows; the sampler makes them differ.
        for _ in range(max(1, args.replicates)):
            rows.append(dict(row))

    if not rows:
        sys.exit("No usable rows; nothing written.")

    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    split = os.path.splitext(os.path.basename(args.out))[0]
    n_designs = len(rows) // max(1, args.replicates)
    print(f"\nwrote {len(rows)} row(s) to {args.out}"
          + (f" -- {n_designs} design(s) x {args.replicates} replicates"
             if args.replicates > 1 else "")
          + (f" ({failures} skipped)" if failures else ""))
    print("\nGenerate with:")
    print(f"  MODEL={args.task} DATA_PATH={parent} SUBSET={split} \\")
    print(f"      RESULTS_PATH=$WORK_ROOT/output/{split} bash infer.sh")
    print(f"\nDesigns land in $WORK_ROOT/output/{split}/{split}/design.txt")


if __name__ == "__main__":
    main()