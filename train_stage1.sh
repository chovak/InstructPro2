#!/bin/bash
#SBATCH --job-name=instructpro2_stage1
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=208
#SBATCH --time=12:00:00
#SBATCH --output=training_logs/instructpro2_stage1_%j.out
#SBATCH --error=training_logs/instructpro2_stage1_%j.err
#SBATCH --requeue
#
# Stage 1 pretraining on InstructPro2Bench: instruction-following over
# <description, protein> pairs, with the span-infilling objective mixed in.
#
# As submitted this runs on 15 nodes x 8 GPUs = 120 ranks. Adjust --nodes and
# --gres above together with GPUS_PER_NODE below; the world size is derived
# from them. The job restarts from ${OUTPUT_PATH}/checkpoint_last.pt if one
# exists, so a requeued job resumes rather than starting over.
#
# Configure by exporting any of the variables below, or edit the defaults.
set -euo pipefail

ip2_locate_root() {
    local candidate
    for candidate in "${INSTRUCTPRO2_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
                     "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$(pwd)"
    do
        if [ -n "${candidate}" ] && [ -f "${candidate}/scripts/common.sh" ]; then
            echo "${candidate}"; return 0
        fi
    done
    return 1
}
INSTRUCTPRO2_ROOT="$(ip2_locate_root)" || {
    echo "ERROR: could not locate the InstructPro2 repository." >&2
    echo "       Submit from the repo root, or export INSTRUCTPRO2_ROOT." >&2
    exit 1
}
export INSTRUCTPRO2_ROOT
# shellcheck source=scripts/common.sh
source "${INSTRUCTPRO2_ROOT}/scripts/common.sh"

DATA_PATH="${DATA_PATH:-${DATA_ROOT}/stage1}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_ROOT}/instructpro2_stage1}"

TRAIN_ENV="${TRAIN_ENV:-InstructPro2}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${NUM_NODES:-15}}"
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Weights & Biases is optional. Set WANDB_PROJECT to enable logging; put
# WANDB_API_KEY in your shell profile, never in this file.
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS=(--wandb-project "${WANDB_PROJECT}")
fi

# Extra flags for a restart that should not inherit optimizer/scheduler state,
# e.g. RESET_FLAGS="--reset-optimizer --reset-lr-scheduler --reset-dataloader".
RESET_FLAGS="${RESET_FLAGS:-}"

ip2_activate "${TRAIN_ENV}"

mkdir -p "${OUTPUT_PATH}" training_logs

# Rendezvous on the first node in the allocation.
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
    export MASTER_ADDR="${nodes[0]}"
else
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
export DIST_PORT="${DIST_PORT:-$((10000 + RANDOM % 20000))}"
export RESTORE_FILE="${RESTORE_FILE:-${OUTPUT_PATH}/checkpoint_last.pt}"

export INSTRUCTPRO2_ROOT DATA_PATH PRETRAIN_MODEL_PATH OUTPUT_PATH
export WORLD_SIZE GPUS_PER_NODE RESET_FLAGS

echo "Master node: ${MASTER_ADDR}   port: ${DIST_PORT}   world size: ${WORLD_SIZE}"
echo "Restore file: ${RESTORE_FILE}"

# SLURM_NODEID must expand on the worker, not here, hence the single quotes.
srun --export=ALL bash -c 'python3 "${INSTRUCTPRO2_ROOT}/fairseq_cli/train_parallel.py" "${DATA_PATH}" \
    --distributed-world-size ${WORLD_SIZE} \
    --distributed-port ${DIST_PORT} \
    --distributed-init-method "tcp://${MASTER_ADDR}:${DIST_PORT}" \
    --distributed-rank $((SLURM_NODEID * GPUS_PER_NODE)) \
    --heartbeat-timeout 3600 \
    --num-workers 0 \
    --save-dir "${OUTPUT_PATH}" \
    --task language_guided_protein_design \
    --dataset-impl instructpro2 \
    --criterion instructpro2_loss --design-factor 1.0 --text-loss-factor 0.5 \
    --arch language_guided_protein_design_model_3B \
    --pretrained-model-path "${PRETRAIN_MODEL_PATH}" \
    --pretrained-model progen2-large-BFD90 \
    --optimizer adam \
    --adam-betas "(0.9,0.98)" \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --stop-min-lr 1e-12 \
    --warmup-updates 10000 \
    --warmup-init-lr 1e-5 \
    --clip-norm 1.0 \
    --ddp-backend c10d \
    --bf16 \
    --log-format simple \
    --log-interval 10 \
    --max-tokens 8000 \
    --max-source-positions 1024 \
    --max-sentences 32 \
    --update-freq 2 \
    --validate-after-updates 1000 \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --valid-subset valid \
    --max-tokens-valid 1024 \
    --max-sentences-valid 14 \
    --max-update 10000000 \
    --validate-interval 1 \
    --save-interval 1 \
    --keep-interval-updates 4 \
    --keep-best-checkpoints 1 \
    --keep-last-epochs 1 \
    --skip-invalid-size-inputs-valid-test \
    --restore-file "${RESTORE_FILE}" \
    --max-epoch 40 \
    '"${WANDB_ARGS[*]}"' \
    ${RESET_FLAGS}'