# InstructPro2

### Natural Language and Biological Grounding Guided De Novo Protein Design

[![arXiv](https://img.shields.io/badge/arXiv-2610.19592-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.19592)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Models-InstructPro2-FFD21E?style=flat-square)](https://huggingface.co/charlesnovak/InstructPro2)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-InstructPro2Bench-FFD21E?style=flat-square)](https://huggingface.co/datasets/charlesnovak/InstructPro2Bench)
[![python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/License-MIT-3DA639?style=flat-square)](LICENSE)

![InstructPro2 architecture](assets/architecture.png)

---

InstructPro2 designs proteins from natural-language function descriptions. A
text encoder reads the instruction, frozen ligand (ChemBERTa) and antigen
(ESM-2) encoders supply biological context through gated cross-attention
adapters, and a 3.2B-parameter decoder initialized from ProGen2-large-BFD90
generates the sequence.

## The three models

| `MODEL=` | Checkpoint | Conditioned on | Produces |
| --- | --- | --- | --- |
| `stage1` | `instructpro2_stage1.pt` | text | a full sequence, or spans in a scaffold |
| `ligand` | `instructpro2_stage2_ligand.pt` | text + SMILES | a full sequence |
| `antibody` | `instructpro2_stage2_antibody.pt` | text + antigen chains | HCDR1-3 in a heavy chain |

`infer.sh` picks the architecture mode, dataset reader and generation flags
to match the checkpoint. All three models also need
`progen2/progen2-large-BFD90/`, including at inference.


## Installation

Two environments are used: one to run the model, and an optional second to
fold and score the generated designs. You only need the second environment
if you want to evaluate structures.

```bash
git clone https://github.com/chovak/InstructPro2.git
cd InstructPro2

conda env create -f instructpro2.yml     # environment: InstructPro2
conda activate InstructPro2
bash setup.sh                            # builds the Cython extensions, installs the package
```

```bash
# optional, for folding and structural scoring
conda env create -f eval.yml             # environment: instructpro2-eval
```

## Weights and data

`WORK_ROOT` is where weights, data and output live. Every script reads it.

```bash
export WORK_ROOT=/path/to/workspace       # default: ~/instructpro2

huggingface-cli download charlesnovak/InstructPro2 --local-dir $WORK_ROOT/model
huggingface-cli download charlesnovak/InstructPro2Bench --repo-type dataset \
    --local-dir $WORK_ROOT/data
```

The model repository is 24 GB in total. To take only what one task needs:

```bash
huggingface-cli download charlesnovak/InstructPro2 \
    --include "instructpro2_stage2_ligand.pt" "progen2/*" \
    --local-dir $WORK_ROOT/model
```

```
$WORK_ROOT/
  model/
    instructpro2_stage1.pt
    instructpro2_stage2_ligand.pt
    instructpro2_stage2_antibody.pt
    progen2/progen2-large-BFD90/        required for every model
  data/
    stage1/     train, valid, test
    ligand/     train, valid, test_ligand, test_ligand_unseen
    antibody/   train, valid, final_antibody_test
    eval_structures/                     ground truth for Stage 1 scoring
  output/                                created by the scripts
```

Each split is a JSONL file. Every row carries `tokenized_instruction`
(PubMedBERT token ids) and the target protein, which may contain `<SEP>`
and `<M1>`–`<M30>` for infilling. The Stage 1 splits name that field
`output` and the Stage 2 splits name it `target_protein`; the reader
accepts either. `ligand` (SMILES) and `antigen` (chain sequences) are
present for the Stage 2 tasks.

`python3 scripts/verify_release_data.py $WORK_ROOT/data` checks that every
split is present and carries the fields the loaders need.


## Quickstart

Design a ligand-binding protein from one of the worked examples:

```bash
export WORK_ROOT=/path/to/workspace
conda activate InstructPro2

python3 scripts/prepare_inputs.py --task ligand \
    --input examples/ligand_designs.csv \
    --out $WORK_ROOT/my_designs/test_examples.jsonl

MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_examples bash infer.sh

cat $WORK_ROOT/output/ligand_test_examples/test_examples/design.txt
```

Each line is `<row id><TAB><sequence>`.

On a cluster, submit the same command — your account and partition go on
the command line, overriding the script's `#SBATCH` defaults:

```bash
MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_examples \
    sbatch -A myaccount --partition=gpu --time=04:00:00 --mem=64G infer.sh
```


## Designing your own proteins

### 1. Write the inputs

Use `scripts/prepare_inputs.py` to tokenize your text and supply a ligand,
antigen or scaffold, depending on the task.

```bash
# one design
python3 scripts/prepare_inputs.py --task ligand \
    --text "I want to build a protein that can perform the following function: ..." \
    --ligand "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" \
    --out $WORK_ROOT/my_designs/test_mine.jsonl

# a batch
python3 scripts/prepare_inputs.py --task ligand \
    --input designs.csv --out $WORK_ROOT/my_designs/test_batch.jsonl
```

A CSV takes columns `text`, `ligand`, `antigen`, `scaffold`; only `text` is
required, and `;` separates several ligands or antigen chains in one cell.
JSONL with the same keys also works. One file is one split, and a single
generation pass covers every row in it.

For the best results, format design requests like the training examples —
see [`examples/`](examples/) for worked inputs you can copy.

### 2. Generate

```bash
MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=test_mine bash infer.sh
```

| Variable | Default | |
| --- | --- | --- |
| `MODEL` | required | `stage1`, `ligand` or `antibody` |
| `DATA_PATH` | `$WORK_ROOT/data/<task>` | directory holding `<SUBSET>.jsonl` |
| `SUBSET` | the task's test split | split name, without `.jsonl` |
| `MODEL_CHECKPOINT` | the task's checkpoint | |
| `RESULTS_PATH` | `$WORK_ROOT/output/<run>` | designs land in `<RESULTS_PATH>/<SUBSET>/` |
| `TOP_P` / `TEMPERATURE` | `0.95` / `0.8` | `TOP_P=0.0` is greedy |
| `SEED` | `1` | |
| `MODALITY` | `on` | `off` ignores the ligand/antigen encoder |

Decoding is stochastic, so one sequence is one sample. `--replicates N` in
`prepare_inputs.py` writes a design N times and returns N independent
samples from a single pass.

### 3. Fold, optionally

```bash
conda activate instructpro2-eval

MODE=downstream RESULTS_PATH=$WORK_ROOT/output/ligand_test_mine \
    DATASETS=test_mine bash fold.sh

PLDDT_ONLY=1 PDB_OUT=$WORK_ROOT/output/ligand_test_mine_pdb bash score.sh
```

`fold.sh` writes one PDB per design, rebuilding antibody heavy chains from
their generated HCDR spans. `score.sh PLDDT_ONLY=1` reports per-design
confidence and needs no ground truth.

### Antibody scaffolds

For HCDR design, the scaffold is a heavy chain with `<M1>`, `<M2>` and
`<M3>` marking the spans to fill:

```
QVQLVQSGTEVRQPGASVRVSCKAS<M1>IHWVRQAPGQGLEWMGI<M2>...<M3>WGQGTLVTVSSA
```

The model writes the three spans and leaves everything else untouched;
`fold.sh` substitutes them back to rebuild the chain.

## Reproducing the paper

### Stage 1

Generate with greedy decoding, fold with ESMFold, score each design against
every structure in its cluster, and measure per-span sequence recovery:

```bash
sbatch eval_stage1.sh                    # the full 1,000-row test split
SUBSET=test_tiny bash eval_stage1.sh     # a slice, as a dry run
STEPS="fold score" bash eval_stage1.sh   # resume without regenerating
```

Needs both environments and `eval_structures/` from the dataset repo.

Greedy is the paper's setting and the default here. For designing proteins
rather than reproducing numbers, use `infer.sh`, which samples.

### Stage 2

The ligand and antibody results in the paper are computed from AlphaFold3
complexes. Ligand designs are folded together with their SMILES and
scored by ipTM, gnina affinity and pocket RMSD. The antibody designs are folded with the
light chain and antigen and scored by ipTM and DockQ. That code is
available on request.

This repository generates the designs (`infer.sh`) and can fold them with
ESMFold for a confidence read (`fold.sh`, then `score.sh PLDDT_ONLY=1`),
which is not the paper's metric.

## Training

```bash
# Stage 1 pretraining. Adjust the SBATCH resources for your cluster;
# a requeued job resumes from checkpoint_last.pt.
sbatch train_stage1.sh

# Stage 2 finetuning, from a Stage 1 checkpoint
STAGE1_CHECKPOINT=$WORK_ROOT/model/instructpro2_stage1.pt bash train_stage2_ligand.sh
STAGE1_CHECKPOINT=$WORK_ROOT/model/instructpro2_stage1.pt bash train_stage2_antibody.sh
```

`--architecture-mode` selects what is trainable:

| Mode | |
| --- | --- |
| `stage1` | text conditioning only, no modality encoders |
| `stage2` | the biological grounding stage reported in the paper |
| `stage2_frozen_decoder`, `stage2_partial`, `stage2_adapters_only` | ablations |


## Scripts

| | |
| --- | --- |
| `infer.sh` | generate designs with any of the three models |
| `fold.sh` | ESMFold on generated designs |
| `score.sh` | structural scoring, or pLDDT alone |
| `eval_stage1.sh` | the full Stage 1 reproduction |
| `train_stage1.sh`, `train_stage2_*.sh` | training |
| `scripts/prepare_inputs.py` | text to a model-ready input file |

Every script is configured by environment variables with defaults, and
carries `#SBATCH` headers you can override on the command line:

```bash
MODEL=ligand sbatch -A myaccount --partition=gpu --time=04:00:00 infer.sh
```

The InstructPro2 code lives inside the `fairseq/` package; this repository
is a fork of [fairseq](https://github.com/facebookresearch/fairseq).
`scripts/` also holds maintainer utilities for preparing releases.


## Citation

```bibtex
@article{instructpro2,
  title   = {InstructPro2: Natural Language and Biological Grounding Guided De Novo Protein Design},
  author  = {Charles Novak and Zhenqiao Song and Chuan Li and Jianwen Xie and Lei Li},
  journal = {arXiv preprint arXiv:2610.19592},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The base ProGen2 weights are BSD-3-Clause.

## Contributing

Issues and pull requests welcome.
Contact: [Charles Novak](mailto:cnovak2@andrew.cmu.edu)