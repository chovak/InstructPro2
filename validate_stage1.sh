#!/bin/bash
#SBATCH --job-name=instructpro2_stage1_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=instructpro2_stage1_gen_%j.out
#SBATCH --error=instructpro2_stage1_gen_%j.err
#
# Stage 1 evaluation: generate protein sequences for the InstructPro2Bench test
# split from natural-language function descriptions alone (no modality encoder).
# Writes target.txt and design.txt into RESULTS_PATH; score them with the
# functional and structural metrics described in the README.
#
# Configure by exporting any of the variables below, or edit the defaults.
set -euo pipefail

INSTRUCTPRO2_ROOT="${INSTRUCTPRO2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_ROOT="${WORK_ROOT:-${HOME}/instructpro2}"

DATA_PATH="${DATA_PATH:-${WORK_ROOT}/data/stage1}"
LOCAL_ROOT="${LOCAL_ROOT:-${WORK_ROOT}/model}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-${LOCAL_ROOT}/progen2}"

# Stage 1 checkpoint to evaluate, and where generations are written.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to the Stage 1 checkpoint (.pt)}"
RESULTS_PATH="${RESULTS_PATH:-${LOCAL_ROOT}/output/stage1_eval}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
EVAL_ENV="${EVAL_ENV:-InstructPro2}"

VALID_SUBSET="${VALID_SUBSET:-test}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-900}"

# shellcheck disable=SC1090
[ -f "${CONDA_SH}" ] && source "${CONDA_SH}" && conda activate "${EVAL_ENV}"

mkdir -p "${RESULTS_PATH}"

python3 "${INSTRUCTPRO2_ROOT}/fairseq_cli/validate.py" "${DATA_PATH}" \
    --pretrained-model-path "${PRETRAIN_MODEL_PATH}" --pretrained-model progen2-large-BFD90 \
    --task language_guided_protein_design \
    --dataset-impl "instructpro2" \
    --path "${MODEL_CHECKPOINT}" \
    --batch-size 1 \
    --results-path "${RESULTS_PATH}" \
    --valid-subset "${VALID_SUBSET}" \
    --max-generation-length "${MAX_GENERATION_LENGTH}" \
    --skip-invalid-size-inputs-valid-test \
    --generation \
    --bf16 \
    --distributed-world-size 1
