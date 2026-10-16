# Example inputs

Three worked examples, one per model. Each is a file you can hand to
`scripts/prepare_inputs.py` unchanged, so you can confirm the whole path
works before writing your own.

```bash
# ligand-binding design (2 designs)
python3 scripts/prepare_inputs.py --task ligand \
    --input examples/ligand_designs.csv \
    --out $WORK_ROOT/my_designs/test_ligand_examples.jsonl

MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_ligand_examples \
    sbatch infer.sh
```

```bash
# text-only design (2 designs)
python3 scripts/prepare_inputs.py --task stage1 \
    --input examples/stage1_designs.csv \
    --out $WORK_ROOT/my_designs/test_stage1_examples.jsonl

MODEL=stage1 DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_stage1_examples \
    sbatch infer.sh
```

```bash
# antibody HCDR design (2 designs)
python3 scripts/prepare_inputs.py --task antibody \
    --input examples/antibody_designs.csv \
    --out $WORK_ROOT/my_designs/test_antibody_examples.jsonl

MODEL=antibody DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_antibody_examples \
    sbatch infer.sh
```

`antibody_design.jsonl` holds the same first design in JSONL form. Either
format works for any task: CSV is easier to edit in a spreadsheet, JSONL
avoids quoting trouble and takes real lists.

| | CSV | JSONL |
| --- | --- | --- |
| several ligands or antigen chains | one cell, separated by `;` | a JSON list |
| text containing commas or quotes | quote the field | nothing to do |
| editing by hand | spreadsheet | text editor |

## Writing instructions that work

**This matters more than any other setting.** The model was pretrained on
205.7M description/protein pairs from UniProtKB, and the instructions it
saw have a consistent shape. A short request like *"a small protein that
binds caffeine"* is far outside that distribution, and tends to produce
short, low-complexity sequences. The examples here are written the way the
training data is written.

For the text-conditioned and ligand tasks, an instruction has three parts:

1. **An opening request.** *"I want to build a protein that can perform the
   following function:"*, *"I'm looking for a protein that has the
   following function:"*, *"I want to make a protein that can have this
   function:"* — the phrasing varies, so match the register rather than any
   one template.
2. **A description of the protein's domains and activity**, which is the
   substance. The training descriptions are derived from InterPro
   annotations and read like them: which domain or family the protein
   belongs to, what that domain does mechanistically, what fold it adopts,
   what it binds or catalyzes. Several sentences, often several hundred
   words, with clauses like *"The protein possesses a … domain that
   functions in …"*, *"Moreover, the protein should include …"*,
   *"the protein belongs to the … family, which …"*.
3. **An optional closing request** — *"Can you generate its sequence?"*,
   *"Give the sequence."*

Be specific about mechanism and structure, not just the outcome. *"Binds
ATP"* says much less than a description of a P-loop NTPase fold that
positions the beta and gamma phosphates for in-line attack.

To read real instructions from the benchmark:

```bash
python3 scripts/prepare_inputs.py --inspect $WORK_ROOT/data/ligand/test_ligand.jsonl --inspect-rows 3
```

Antibody instructions follow a different, tighter template: the request,
then the format and germline (*"A human Fab (IGHV3/IGKV1-Kappa)"*), the
target antigen, a sentence or two on what that antigen does, and the
functional role being designed for (neutralizing, agonist, and so on).

## The antibody scaffold

The `scaffold` column is a heavy chain with `<M1>`, `<M2>` and `<M3>` where
HCDR1, HCDR2 and HCDR3 belong. Everything else is framework and is kept
verbatim — the model writes only the three spans, and `fold.sh` puts them
back to rebuild the chain.

The two examples use different germlines (IGHV3 and IGHV5) to show that the
framework is yours to choose; the mask markers sit at the CDR boundaries in
both. The second mirrors a row from the benchmark's antibody test split, so
you can see what a real input looks like.

The `antigen` field takes one entry per chain, and the benchmark's rows
commonly carry two — in CSV, separate them with `;` in a single cell.

## What you get back

`design.txt`, one line per design, as `<row id><TAB><sequence>`. Antibody
rows come back as `SCAFFOLD<SEP><M1>…<M2>…<M3>…`; rebuild the full chain
with `fold.sh`, which substitutes the spans back into the scaffold.

Decoding is stochastic (top-p 0.95, temperature 0.8), so a single sequence
is one sample. `--replicates N` writes a design N times and returns N
independent samples from one pass, which is a practical way to deal with a
short or degenerate result.