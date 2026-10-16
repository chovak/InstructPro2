#!/bin/bash
#SBATCH --job-name=instructpro2_fold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/fold_%j.out
#SBATCH --error=logs/fold_%j.err
#
# Fold generated designs with ESMFold. Needs no ground truth, so this is
# useful on your own designs as well as on the benchmark: it writes one PDB
# per design and reports pLDDT.
#
# Runs in the evaluation environment (eval.yml), not the generation one:
# ESMFold and PyMOL have their own dependency stack.
#
#   MODE=stage1      one design.txt of mixed autoregressive + infilling rows,
#                    routed into per-cluster folders
#   MODE=downstream  one design.txt per dataset subdirectory
#
# Examples:
#
#   # Stage 1
#   MODE=stage1 RESULTS_PATH=$WORK_ROOT/output/stage1_test \
#       sbatch fold.sh
#
#   # Stage 2, both ligand splits from one generation run
#   MODE=downstream RESULTS_PATH=$WORK_ROOT/output/ligand_eval \
#       DATASETS=test_ligand,test_ligand_unseen sbatch fold.sh
#
#   # Parse only, no GPU: check how many designs survive cleaning
#   MODE=downstream RESULTS_PATH=... NO_FOLD=1 bash fold.sh
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

MODE="${MODE:?set MODE to stage1 or downstream}"
# Where infer.sh wrote design.txt.
RESULTS_PATH="${RESULTS_PATH:?set RESULTS_PATH to the generation output directory}"
RUN_NAME="${RUN_NAME:-$(basename "${RESULTS_PATH}")}"
PDB_OUT="${PDB_OUT:-${RESULTS_PATH}_pdb}"
MAPPED_CSV="${MAPPED_CSV:-${PDB_OUT}/mapped_${RUN_NAME}.csv}"

FOLD_ENV="${FOLD_ENV:-instructpro2-eval}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# None is fastest; set a number (64, 32, 16) to trade speed for memory.
CHUNK_SIZE="${CHUNK_SIZE:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARGS=(--mode "${MODE}"
      --output_dir "${PDB_OUT}"
      --mapped_csv "${MAPPED_CSV}"
      --run_name "${RUN_NAME}"
      --batch_size "${BATCH_SIZE}")

case "${MODE}" in
    stage1)
        # infer.sh writes <RESULTS_PATH>/<split>/design.txt. Accept the
        # named split, a flat design.txt, or a lone subdirectory, so this
        # works whichever way the generation was produced.
        if [ -z "${DESIGN_TXT:-}" ]; then
            for candidate in "${RESULTS_PATH}/${SUBSET:-test}/design.txt" \
                             "${RESULTS_PATH}/design.txt"
            do
                [ -f "${candidate}" ] && { DESIGN_TXT="${candidate}"; break; }
            done
        fi
        if [ -z "${DESIGN_TXT:-}" ]; then
            found=$(find "${RESULTS_PATH}" -mindepth 2 -maxdepth 2 \
                        -name design.txt 2>/dev/null)
            [ "$(echo "${found}" | grep -c .)" = 1 ] && DESIGN_TXT="${found}"
        fi
        DESIGN_TXT="${DESIGN_TXT:-${RESULTS_PATH}/design.txt}"
        # Supplies metadata.cluster and task_type for each row.
        REFERENCE_JSONL="${REFERENCE_JSONL:-${DATA_ROOT}/stage1/test.jsonl}"
        require_file "${DESIGN_TXT}"      "no design.txt; run infer.sh first"
        require_file "${REFERENCE_JSONL}" "reference split not found"
        ARGS+=(--design_txt "${DESIGN_TXT}" --reference_jsonl "${REFERENCE_JSONL}")
        ;;
    downstream)
        DATASETS="${DATASETS:-test_ligand,test_ligand_unseen}"
        require_dir "${RESULTS_PATH}" "generation output not found"
        ARGS+=(--checkpoint_dir "${RESULTS_PATH}" --datasets "${DATASETS}")
        ;;
    *)
        die "MODE must be stage1 or downstream (got '${MODE}')"
        ;;
esac

[ -n "${CHUNK_SIZE}" ] && ARGS+=(--chunk_size "${CHUNK_SIZE}")
[ -n "${NO_FOLD:-}" ] && ARGS+=(--no_fold)

mkdir -p "${PDB_OUT}" logs

echo "=============================================================="
echo "InstructPro2 folding: ${MODE}"
echo "=============================================================="
ip2_print_paths
echo "designs    : ${RESULTS_PATH}"
echo "PDBs       : ${PDB_OUT}"
echo "mapping    : ${MAPPED_CSV}"
echo "=============================================================="

# --no_fold needs no GPU and no ESMFold, so skip the environment switch
# when the caller only wants parse statistics.
if [ -z "${NO_FOLD:-}" ]; then
    ip2_activate "${FOLD_ENV}"
fi

python3 "${INSTRUCTPRO2_ROOT}/eval/fold_designs.py" "${ARGS[@]}"

echo
echo "folded structures in ${PDB_OUT}"
echo "next: score.sh to compare against ground truth, or read pLDDT with"
echo "      score_structures.py --plddt_only for designs of your own"