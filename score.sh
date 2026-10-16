#!/bin/bash
#SBATCH --job-name=instructpro2_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
#
# Score folded designs against the published ground-truth structures.
# CPU only -- PyMOL alignment, no GPU needed.
#
# This step is for REPRODUCING THE PAPER. It needs eval_structures/, which
# ships with the InstructPro2Bench dataset:
#
#   huggingface-cli download charlesnovak/InstructPro2Bench \
#       --repo-type dataset --local-dir $WORK_ROOT/data
#
# Designing your own proteins needs none of this: there is no ground truth
# for a sequence nobody has solved. Use PLDDT_ONLY=1 to get per-design
# confidence from the folded structures instead, which needs no targets and
# no PyMOL.
#
#   MODE=row      one target per design, matched by row id
#                 (test_ligand, test_ligand_unseen, final_antibody_test)
#   MODE=cluster  scored against every member of the row's cluster (stage 1)
#   unset         inferred from the mapping CSV
#
# Examples:
#
#   MODE=row PDB_OUT=$WORK_ROOT/output/ligand_eval_pdb sbatch score.sh
#   MODE=cluster PDB_OUT=$WORK_ROOT/output/stage1_test_pdb sbatch score.sh
#   PLDDT_ONLY=1 PDB_OUT=$WORK_ROOT/output/my_designs_pdb bash score.sh
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

PDB_OUT="${PDB_OUT:?set PDB_OUT to the directory fold.sh wrote}"
RUN_NAME="${RUN_NAME:-$(basename "${PDB_OUT}")}"

# fold.sh names it mapped_<run>.csv; fall back to whatever single mapping
# CSV is in the directory, which is what a renamed run leaves behind.
if [ -z "${MAPPED_CSV:-}" ]; then
    MAPPED_CSV="${PDB_OUT}/mapped_${RUN_NAME}.csv"
    if [ ! -f "${MAPPED_CSV}" ]; then
        found=$(find "${PDB_OUT}" -maxdepth 1 -name 'mapped_*.csv' | head -1)
        [ -n "${found}" ] && MAPPED_CSV="${found}"
    fi
fi
require_file "${MAPPED_CSV}" \
    "mapping CSV not found; run fold.sh first, or set MAPPED_CSV"

TARGETS_ROOT="${TARGETS_ROOT:-${DATA_ROOT}/eval_structures}"
TARGETS_CSV="${TARGETS_CSV:-${TARGETS_ROOT}/targets.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PDB_OUT}/scores_${RUN_NAME}.csv}"
EVAL_ENV="${EVAL_ENV:-instructpro2-eval}"
RMSD_THRESHOLD="${RMSD_THRESHOLD:-2.0}"

ARGS=(--mapped_csv "${MAPPED_CSV}"
      --output_csv "${OUTPUT_CSV}"
      --designs_root "${PDB_OUT}"
      --run_name "${RUN_NAME}"
      --rmsd_threshold "${RMSD_THRESHOLD}")

[ -n "${MODE:-}" ]  && ARGS+=(--mode "${MODE}")
[ -n "${SPLIT:-}" ] && ARGS+=(--split "${SPLIT}")

if [ -n "${PLDDT_ONLY:-}" ]; then
    ARGS+=(--plddt_only)
else
    require_file "${TARGETS_CSV}" \
        "targets.csv not found. The ground-truth structures ship with the
       dataset:  huggingface-cli download charlesnovak/InstructPro2Bench \\
                     --repo-type dataset --local-dir \$WORK_ROOT/data
       To score your own designs instead, set PLDDT_ONLY=1."
    ARGS+=(--targets_csv "${TARGETS_CSV}" --targets_root "${TARGETS_ROOT}")
fi

mkdir -p logs

echo "=============================================================="
echo "InstructPro2 scoring: ${RUN_NAME}"
echo "=============================================================="
echo "designs  : ${PDB_OUT}"
echo "mapping  : ${MAPPED_CSV}"
if [ -n "${PLDDT_ONLY:-}" ]; then
    echo "targets  : none (pLDDT only)"
else
    echo "targets  : ${TARGETS_ROOT}"
fi
echo "output   : ${OUTPUT_CSV}"
echo "=============================================================="

# pLDDT is read straight from the PDB B-factor column, so that path needs
# neither PyMOL nor the eval environment.
if [ -z "${PLDDT_ONLY:-}" ]; then
    ip2_activate "${EVAL_ENV}"
fi

python3 "${INSTRUCTPRO2_ROOT}/eval/score_structures.py" "${ARGS[@]}"

echo
echo "per-design scores: ${OUTPUT_CSV}"