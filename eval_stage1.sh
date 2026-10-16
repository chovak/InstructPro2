#!/bin/bash
#SBATCH --job-name=instructpro2_eval_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_stage1_%j.out
#SBATCH --error=logs/eval_stage1_%j.err
#
# Reproduce the Stage 1 evaluation from the paper, end to end:
#
#   generate  greedy decoding over the InstructPro2Bench test split
#   fold      ESMFold on each design, routed into its cluster folder
#   score     PyMOL RMSD against every structure in that cluster
#   infill    per-mask sequence recovery for the span-infilling rows
#
# This is the one pipeline the repository reproduces in full. The ligand
# and antibody results in the paper come from AlphaFold3 complexes scored
# with ipTM and DockQ, which is a separate release; for those two models
# this repository gives you designs (infer.sh) and optional ESMFold
# confidence (fold.sh, then score.sh PLDDT_ONLY=1).
#
# Needs both environments -- InstructPro2 to generate, instructpro2-eval to
# fold and score -- and the ground-truth structures:
#
#   huggingface-cli download charlesnovak/InstructPro2Bench \
#       --repo-type dataset --local-dir $WORK_ROOT/data
#
# Examples:
#
#   sbatch eval_stage1.sh                      # the full test split
#   SUBSET=test_tiny bash eval_stage1.sh       # a slice, for a dry run
#   STEPS="fold score" bash eval_stage1.sh     # resume after generation
#
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

SUBSET="${SUBSET:-test}"
RUN_NAME="${RUN_NAME:-stage1_${SUBSET}}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MODEL_ROOT}/instructpro2_stage1.pt}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/stage1}"
REFERENCE_JSONL="${REFERENCE_JSONL:-${DATA_PATH}/${SUBSET}.jsonl}"

RESULTS_PATH="${RESULTS_PATH:-${WORK_ROOT}/output/${RUN_NAME}}"
SUBSET_PATH="${RESULTS_PATH}/${SUBSET}"
PDB_OUT="${PDB_OUT:-${RESULTS_PATH}_pdb}"
MAPPED_CSV="${PDB_OUT}/mapped_${RUN_NAME}.csv"
SCORES_CSV="${PDB_OUT}/scores_${RUN_NAME}.csv"
INFILL_OUT="${INFILL_OUT:-${PDB_OUT}/infill_metrics}"

TARGETS_ROOT="${TARGETS_ROOT:-${DATA_ROOT}/eval_structures}"
TARGETS_CSV="${TARGETS_CSV:-${TARGETS_ROOT}/targets.csv}"

GEN_ENV="${GEN_ENV:-InstructPro2}"
EVAL_ENV="${EVAL_ENV:-instructpro2-eval}"

# The paper's Stage 1 numbers are greedy: no temperature, no top-p. Set
# TOP_P/TEMPERATURE to sample instead, which gives better-looking designs
# but will not reproduce the published figures.
TOP_P="${TOP_P:-0.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-1}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-900}"
FOLD_BATCH_SIZE="${FOLD_BATCH_SIZE:-16}"

STEPS="${STEPS:-generate fold score infill}"
has_step() { case " ${STEPS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

require_file "${REFERENCE_JSONL}" "test split not found"
mkdir -p "${RESULTS_PATH}" "${PDB_OUT}" logs

echo "=============================================================="
echo "InstructPro2 Stage 1 evaluation: ${RUN_NAME}"
echo "=============================================================="
ip2_print_paths
echo "checkpoint : ${MODEL_CHECKPOINT}"
echo "split      : ${REFERENCE_JSONL} ($(wc -l < "${REFERENCE_JSONL}") rows)"
echo "decoding   : $([ "${TOP_P}" = "0.0" ] && echo 'greedy (paper setting)' \
                   || echo "top-p ${TOP_P}, temperature ${TEMPERATURE}")"
echo "steps      : ${STEPS}"
echo "=============================================================="

# --- 1. generation ---------------------------------------------------------
if has_step generate; then
    echo ""
    echo ">>> [1/4] generating designs <<<"
    MODEL=stage1 \
    SUBSET="${SUBSET}" \
    DATA_PATH="${DATA_PATH}" \
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
    RESULTS_PATH="${RESULTS_PATH}" \
    RUN_NAME="${RUN_NAME}" \
    TOP_P="${TOP_P}" TEMPERATURE="${TEMPERATURE}" SEED="${SEED}" \
    MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH}" \
    GEN_ENV="${GEN_ENV}" \
        bash "${INSTRUCTPRO2_ROOT}/infer.sh"
fi

require_file "${SUBSET_PATH}/design.txt" \
    "no designs; run with STEPS including 'generate'"

# --- 2. folding ------------------------------------------------------------
if has_step fold; then
    echo ""
    echo ">>> [2/4] folding with ESMFold <<<"
    MODE=stage1 \
    RESULTS_PATH="${RESULTS_PATH}" \
    SUBSET="${SUBSET}" \
    REFERENCE_JSONL="${REFERENCE_JSONL}" \
    PDB_OUT="${PDB_OUT}" \
    MAPPED_CSV="${MAPPED_CSV}" \
    RUN_NAME="${RUN_NAME}" \
    BATCH_SIZE="${FOLD_BATCH_SIZE}" \
    FOLD_ENV="${EVAL_ENV}" \
        bash "${INSTRUCTPRO2_ROOT}/fold.sh"
fi

# --- 3. structural scoring -------------------------------------------------
if has_step score; then
    echo ""
    echo ">>> [3/4] scoring against cluster structures <<<"
    if [ ! -f "${TARGETS_CSV}" ]; then
        echo "  targets.csv not found at ${TARGETS_CSV}; skipping."
        echo "  huggingface-cli download charlesnovak/InstructPro2Bench \\"
        echo "      --repo-type dataset --local-dir \$WORK_ROOT/data"
    else
        MODE=cluster \
        PDB_OUT="${PDB_OUT}" \
        MAPPED_CSV="${MAPPED_CSV}" \
        OUTPUT_CSV="${SCORES_CSV}" \
        TARGETS_ROOT="${TARGETS_ROOT}" \
        TARGETS_CSV="${TARGETS_CSV}" \
        RUN_NAME="${RUN_NAME}" \
        EVAL_ENV="${EVAL_ENV}" \
            bash "${INSTRUCTPRO2_ROOT}/score.sh"
    fi
fi

# --- 4. infilling recovery -------------------------------------------------
# Sequence recovery over the masked spans, for the infilling rows only.
# Rows without <SEP> in the target are autoregressive and are skipped.
if has_step infill; then
    echo ""
    echo ">>> [4/4] infilling recovery metrics <<<"
    ip2_activate "${EVAL_ENV}"
    python3 "${INSTRUCTPRO2_ROOT}/eval/get_infill_metrics.py" \
        --ckpt_dirs "${SUBSET_PATH}" \
        --out_dir "${INFILL_OUT}"
fi

echo ""
echo "=============================================================="
echo "Stage 1 evaluation complete"
echo "=============================================================="
echo "designs        : ${SUBSET_PATH}/design.txt"
echo "structures     : ${PDB_OUT}"
[ -f "${SCORES_CSV}" ] && echo "structural RMSD: ${SCORES_CSV}"
[ -d "${INFILL_OUT}" ] && echo "infill recovery: ${INFILL_OUT}"