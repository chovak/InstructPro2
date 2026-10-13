#!/bin/bash
# Ligand-binding protein design: generation -> folding -> structural evaluation.
#
# Runs the pipeline twice, once with the ligand encoder active ("w_modality",
# architecture-mode stage2) and once without it ("no_modality",
# architecture-mode stage1). The pair is the modality ablation reported in the
# paper; run only the first if you just want design output.
#
# Configure by exporting any of the variables below, or edit the defaults.
set -euo pipefail

# --- paths -------------------------------------------------------------------
INSTRUCTPRO2_ROOT="${INSTRUCTPRO2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_ROOT="${WORK_ROOT:-${HOME}/instructpro2}"

DATA_PATH="${DATA_PATH:-${WORK_ROOT}/data/ligand}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/model}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-${MODEL_ROOT}/progen2}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-${WORK_ROOT}/data/target_test_clusters/pdb_ground_truth}"

# Finetuned Stage 2 ligand checkpoint to evaluate.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to the Stage 2 ligand checkpoint (.pt)}"
BASE_RUN_NAME="${BASE_RUN_NAME:-instructpro2_ligand_eval}"

# Structure folding / evaluation scripts (vendored under eval/ in this repo).
EVAL_REPO="${EVAL_REPO:-${INSTRUCTPRO2_ROOT}/eval}"

# Conda environments: one for generation, one for ESMFold.
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
GEN_ENV="${GEN_ENV:-InstructPro2_Stage2}"
FOLD_ENV="${FOLD_ENV:-esmfold}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/huggingface_cache}"

SUBSETS="${SUBSETS:-test_ligand test_ligand_unseen}"
MODES="${MODES:-w_modality no_modality}"

echo "Starting ligand evaluation on $(hostname)"
# shellcheck disable=SC1090
source "${CONDA_SH}"

for MODE in ${MODES}; do
    echo ""
    echo "=========================================="
    echo ">>> ${MODE} <<<"
    echo "=========================================="

    # stage2 keeps the ligand encoder active; stage1 is text only
    if [ "${MODE}" == "w_modality" ]; then
        ARCH_MODE="stage2"
    else
        ARCH_MODE="stage1"
    fi

    RUN_NAME="${BASE_RUN_NAME}_${MODE}"
    GENERATION_OUT_DIR="${MODEL_ROOT}/output/${RUN_NAME}"
    TARGET_TEST_DIR="${WORK_ROOT}/data/target_test_clusters/${RUN_NAME}"
    MAPPED_CSV="${TARGET_TEST_DIR}/stage2_generation_mapped_${RUN_NAME}.csv"
    EVAL_CSV="${TARGET_TEST_DIR}/evaluation_results_${RUN_NAME}.csv"

    # --- 1. sequence generation ---------------------------------------------
    conda activate "${GEN_ENV}"

    for subset in ${SUBSETS}; do
        SUBSET_GEN_PATH="${GENERATION_OUT_DIR}/${subset}"
        mkdir -p "${SUBSET_GEN_PATH}"
        echo "--- generating: ${subset} | mode: ${ARCH_MODE} ---"

        python3 "${INSTRUCTPRO2_ROOT}/fairseq_cli/validate.py" "${DATA_PATH}" \
            --pretrained-model-path "${PRETRAIN_MODEL_PATH}" \
            --pretrained-model progen2-large-BFD90 \
            --task language_guided_protein_design \
            --architecture-mode "${ARCH_MODE}" \
            --dataset-impl instructpro2_ligand \
            --adapter-frequency 4 \
            --path "${MODEL_CHECKPOINT}" \
            --batch-size 1 \
            --results-path "${SUBSET_GEN_PATH}" \
            --valid-subset "${subset}" \
            --max-generation-length 1024 \
            --skip-invalid-size-inputs-valid-test \
            --generation \
            --bf16 \
            --distributed-world-size 1 \
            --top-p 0.95 \
            --temperature 0.8 \
            --use-segment-rope \
            --text-protein-rope-gap 32 \
            --text-rope-scale 1.0 \
            --seed 1
    done

    # --- 2. structure folding (ESMFold) --------------------------------------
    conda activate "${FOLD_ENV}"
    mkdir -p "${TARGET_TEST_DIR}"

    python -u "${EVAL_REPO}/fold_instructpro2_downstream_output.py" \
        --checkpoint_dir "${GENERATION_OUT_DIR}" \
        --output_dir "${TARGET_TEST_DIR}" \
        --run_name "${RUN_NAME}" \
        --batch_size 16 \
        --datasets "$(echo ${SUBSETS} | tr ' ' ',')"

    # --- 3. structural evaluation (PyMOL) ------------------------------------
    python "${EVAL_REPO}/evaluate_structures.py" \
        --mapped_csv "${MAPPED_CSV}" \
        --targets_dir "${GROUND_TRUTH_DIR}" \
        --output_csv "${EVAL_CSV}" \
        --run_name "${BASE_RUN_NAME}_sampling_${MODE}"
done

echo ""
echo ">>> ligand evaluation complete <<<"
