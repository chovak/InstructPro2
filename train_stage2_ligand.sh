#!/bin/bash
# Stage 2 finetuning: ligand-binding protein design.
#
# Starts from a Stage 1 pretraining checkpoint and trains the gated
# cross-attention adapters together with the rest of the trainable stack
# (architecture-mode stage2). The ligand encoder and the text
# encoder stay frozen.
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

DATA_PATH="${DATA_PATH:-${DATA_ROOT}/ligand}"

# Stage 1 checkpoint to finetune from, and where Stage 2 checkpoints are written.
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT to the Stage 1 checkpoint (.pt)}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_ROOT}/stage2_ligand}"

TRAIN_ENV="${TRAIN_ENV:-InstructPro2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 20000))}"

# Weights & Biases is optional. Set WANDB_PROJECT to enable logging.
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS=(--wandb-project "${WANDB_PROJECT}")
fi

ip2_activate "${TRAIN_ENV}"

mkdir -p "${OUTPUT_PATH}"

python3 "${INSTRUCTPRO2_ROOT}/fairseq_cli/train_parallel.py" "${DATA_PATH}" \
    --distributed-port "${MASTER_PORT}" \
    --heartbeat-timeout 7200 \
    --num-workers 0 \
    --distributed-world-size 1 \
    --save-dir "${OUTPUT_PATH}" \
    --finetune-from-model "${STAGE1_CHECKPOINT}" \
    --task language_guided_protein_design \
    --dataset-impl "instructpro2_ligand" \
    --criterion instructpro2_loss --design-factor 1.0 --text-loss-factor 0.5 \
    --arch language_guided_protein_design_model_3B \
    --architecture-mode stage2 \
    --adapter-frequency 4 \
    --pretrained-model-path "${PRETRAIN_MODEL_PATH}" --pretrained-model progen2-large-BFD90 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 5e-5 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-12' --warmup-updates 2000 \
    --warmup-init-lr '1e-6' \
    --clip-norm 0.5 \
    --ddp-backend c10d \
    --find-unused-parameters \
    --log-format 'simple' --log-interval 10 \
    --max-tokens 1100 \
    --max-source-positions 1024 \
    --max-sentences 20 \
    --update-freq 1 \
    --max-update 60000000 \
    --max-epoch 20 \
    --validate-after-updates 10000 \
    --validate-interval-updates 10000 \
    --save-interval-updates 10000 \
    --keep-interval-updates 1 \
    --valid-subset valid \
    --max-tokens-valid 1024 \
    --max-sentences-valid 14 \
    --validate-interval 1 \
    --save-interval 1 \
    --no-epoch-checkpoints \
    --keep-last-epochs 1 \
    --skip-invalid-size-inputs-valid-test \
    --bf16 \
    --use-segment-rope \
    --text-protein-rope-gap 32 \
    --text-rope-scale 1.0 \
    "${WANDB_ARGS[@]}"