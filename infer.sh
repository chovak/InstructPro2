#!/bin/bash
#SBATCH --job-name=instructpro2_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#
# Generate protein sequences with InstructPro2. This is the only script that
# runs the model; folding and scoring are separate (fold.sh, score.sh) so
# that designing sequences never requires ground-truth structures.
#
# Pick the model with MODEL; everything else has a working default.
#
#   MODEL=stage1    text-conditioned design and span infilling
#   MODEL=ligand    ligand-binding design, conditioned on SMILES
#   MODEL=antibody  HCDR1-3 infilling, conditioned on an antigen
#
# Examples:
#
#   # Stage 1 on the benchmark test split (paper setting: greedy)
#   MODEL=stage1 sbatch infer.sh
#
#   # Ligand design on your own inputs
#   MODEL=ligand DATA_PATH=$WORK_ROOT/my_designs SUBSET=my_ligands \
#       RESULTS_PATH=$WORK_ROOT/output/my_ligands bash infer.sh
#
#   # The no-modality arm of the paper's ablation
#   MODEL=ligand MODALITY=off sbatch infer.sh
#
# This file's #SBATCH lines are deliberately minimal; your cluster's account
# and partition go on the command line, where they override these:
#
#   MODEL=stage1 sbatch -A myaccount --partition=gpu --time=04:00:00 infer.sh
#
set -euo pipefail

# Find the repo. Under sbatch the batch script is copied to a spool
# directory, so paths relative to this file do not point at the repo;
# SLURM_SUBMIT_DIR is where it was submitted from. INSTRUCTPRO2_ROOT wins
# if set, which also covers submitting from somewhere else entirely.
ip2_locate_root() {
    local candidate
    for candidate in "${INSTRUCTPRO2_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
                     "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$(pwd)"
    do
        if [ -n "${candidate}" ] && [ -f "${candidate}/scripts/common.sh" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

INSTRUCTPRO2_ROOT="$(ip2_locate_root)" || {
    echo "ERROR: could not locate the InstructPro2 repository." >&2
    echo "       Submit from the repo root, or set INSTRUCTPRO2_ROOT:" >&2
    echo "         export INSTRUCTPRO2_ROOT=/path/to/InstructPro2" >&2
    exit 1
}
export INSTRUCTPRO2_ROOT

# shellcheck source=scripts/common.sh
source "${INSTRUCTPRO2_ROOT}/scripts/common.sh"

# --- what to run -----------------------------------------------------------
MODEL="${MODEL:?set MODEL to stage1, ligand, or antibody}"
# on  = modality encoder active (architecture-mode stage2)
# off = text only (architecture-mode stage1); the paper's ablation arm.
# Ignored for MODEL=stage1, which has no modality encoder.
MODALITY="${MODALITY:-on}"

case "${MODEL}" in
    stage1)
        DATA_PATH="${DATA_PATH:-${DATA_ROOT}/stage1}"
        SUBSET="${SUBSET:-test}"
        DATASET_IMPL="${DATASET_IMPL:-instructpro2}"
        MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MODEL_ROOT}/instructpro2_stage1.pt}"
        ARCH_MODE="stage1"
        USE_SEGMENT_ROPE=0
        # Stage 1 results in the paper are greedy: no temperature, no top-p.
        DEFAULT_TOP_P=0.0
        DEFAULT_TEMPERATURE=1.0
        DEFAULT_MAX_GEN=900
        ;;
    ligand)
        DATA_PATH="${DATA_PATH:-${DATA_ROOT}/ligand}"
        SUBSET="${SUBSET:-test_ligand}"
        DATASET_IMPL="${DATASET_IMPL:-instructpro2_ligand}"
        MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MODEL_ROOT}/instructpro2_stage2_ligand.pt}"
        ARCH_MODE=$([ "${MODALITY}" = "on" ] && echo stage2 || echo stage1)
        USE_SEGMENT_ROPE=1
        DEFAULT_TOP_P=0.95
        DEFAULT_TEMPERATURE=0.8
        DEFAULT_MAX_GEN=1024
        ;;
    antibody)
        DATA_PATH="${DATA_PATH:-${DATA_ROOT}/antibody}"
        SUBSET="${SUBSET:-final_antibody_test}"
        DATASET_IMPL="${DATASET_IMPL:-instructpro2_antibody}"
        MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MODEL_ROOT}/instructpro2_stage2_antibody.pt}"
        ARCH_MODE=$([ "${MODALITY}" = "on" ] && echo stage2 || echo stage1)
        USE_SEGMENT_ROPE=1
        DEFAULT_TOP_P=0.95
        DEFAULT_TEMPERATURE=0.8
        DEFAULT_MAX_GEN=1024
        ;;
    *)
        die "MODEL must be stage1, ligand, or antibody (got '${MODEL}')"
        ;;
esac

RUN_NAME="${RUN_NAME:-${MODEL}_${SUBSET}$([ "${MODALITY}" = off ] && echo _no_modality || echo '')}"
RESULTS_PATH="${RESULTS_PATH:-${WORK_ROOT}/output/${RUN_NAME}}"

# design.txt goes in a subdirectory named for the split, so one run
# directory can hold several splits of the same model (the ligand task has
# two) and fold.sh can pick them up by name.
SUBSET_PATH="${RESULTS_PATH}/${SUBSET}"

# --- decoding --------------------------------------------------------------
# top-p 0 means greedy; the task reads it that way.
TOP_P="${TOP_P:-${DEFAULT_TOP_P}}"
TEMPERATURE="${TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-${DEFAULT_MAX_GEN}}"
SEED="${SEED:-1}"

# Infilling derives one generation prefix per batch, so a batch holding rows
# with different prefixes would reuse the first row's. Until that is fixed,
# batch size stays at 1 for every split that can contain <SEP>.
BATCH_SIZE="${BATCH_SIZE:-1}"

ADAPTER_FREQUENCY="${ADAPTER_FREQUENCY:-4}"
GEN_ENV="${GEN_ENV:-InstructPro2}"

# --- checks before the GPU is touched --------------------------------------
require_dir  "${DATA_PATH}"        "data directory not found"
require_file "${DATA_PATH}/${SUBSET}.jsonl" \
    "split '${SUBSET}' not found in ${DATA_PATH}. Available: $(ls "${DATA_PATH}"/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
require_file "${MODEL_CHECKPOINT}" "checkpoint not found"
require_dir  "${PRETRAIN_MODEL_PATH}/${PRETRAIN_MODEL}" \
    "ProGen2 weights not found. They ship with the model repo and are needed
       even for inference: huggingface-cli download charlesnovak/InstructPro2 --local-dir \$WORK_ROOT/model"

mkdir -p "${SUBSET_PATH}" logs

echo "=============================================================="
echo "InstructPro2 inference: ${MODEL}"
echo "=============================================================="
ip2_print_paths
echo "checkpoint : ${MODEL_CHECKPOINT}"
echo "split      : ${DATA_PATH}/${SUBSET}.jsonl ($(wc -l < "${DATA_PATH}/${SUBSET}.jsonl") rows)"
echo "arch mode  : ${ARCH_MODE} (modality ${MODALITY})"
echo "decoding   : $([ "${TOP_P}" = "0.0" ] && echo "greedy" || echo "top-p ${TOP_P}, temperature ${TEMPERATURE}"), seed ${SEED}"
echo "output     : ${SUBSET_PATH}/design.txt"
echo "=============================================================="

ip2_activate "${GEN_ENV}"

EXTRA_ARGS=()
if [ "${USE_SEGMENT_ROPE}" = "1" ]; then
    # Must match training. The Stage 2 checkpoints were trained with these;
    # the Stage 1 checkpoint was not.
    EXTRA_ARGS+=(--use-segment-rope
                 --text-protein-rope-gap "${TEXT_PROTEIN_ROPE_GAP:-32}"
                 --text-rope-scale "${TEXT_ROPE_SCALE:-1.0}")
fi
if [ "${TOP_P}" != "0.0" ]; then
    EXTRA_ARGS+=(--top-p "${TOP_P}" --temperature "${TEMPERATURE}")
fi

set -x
python3 "${INSTRUCTPRO2_ROOT}/fairseq_cli/validate.py" "${DATA_PATH}" \
    --pretrained-model-path "${PRETRAIN_MODEL_PATH}" \
    --pretrained-model "${PRETRAIN_MODEL}" \
    --task language_guided_protein_design \
    --architecture-mode "${ARCH_MODE}" \
    --dataset-impl "${DATASET_IMPL}" \
    --adapter-frequency "${ADAPTER_FREQUENCY}" \
    --path "${MODEL_CHECKPOINT}" \
    --batch-size "${BATCH_SIZE}" \
    --results-path "${SUBSET_PATH}" \
    --valid-subset "${SUBSET}" \
    --max-generation-length "${MAX_GENERATION_LENGTH}" \
    --skip-invalid-size-inputs-valid-test \
    --generation \
    --bf16 \
    --distributed-world-size 1 \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
set +x

echo
echo "wrote $(wc -l < "${SUBSET_PATH}/design.txt" 2>/dev/null || echo 0) designs to ${SUBSET_PATH}/design.txt"
echo "Rows whose target exceeds 1024 tokens are skipped, so this can be"
echo "fewer than the number of rows in the split."