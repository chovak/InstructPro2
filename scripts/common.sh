#!/bin/bash
# Shared setup for the InstructPro2 scripts. Sourced, never run directly.
#
# Provides:
#   ip2_activate <env>   activate a conda environment safely
#   die <message>        exit with a message on stderr
#   require_file <path> <hint>
#   require_dir  <path> <hint>
#
# and the common path defaults: INSTRUCTPRO2_ROOT, WORK_ROOT, DATA_ROOT,
# MODEL_ROOT, PRETRAIN_MODEL_PATH, HF_HOME.

# --- paths -----------------------------------------------------------------
# The caller normally sets INSTRUCTPRO2_ROOT before sourcing this, because
# under sbatch the batch script lives in a spool directory and cannot find
# the repo from its own path. When this file is sourced directly (an
# interactive shell, say) its own location is a safe fallback, since it
# really does live in the repo.
if [ -z "${INSTRUCTPRO2_ROOT:-}" ]; then
    IP2_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    INSTRUCTPRO2_ROOT="$(cd "${IP2_COMMON_DIR}/.." && pwd)"
fi

WORK_ROOT="${WORK_ROOT:-${HOME}/instructpro2}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/model}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-${MODEL_ROOT}/progen2}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-progen2-large-BFD90}"

# The model pulls PubMedBERT, ChemBERTa and ESM-2 from the Hub when it is
# built, including at inference. Keeping the cache under WORK_ROOT means a
# compute node without internet can still find them, provided the cache was
# warmed once from a node that has it.
export HF_HOME="${HF_HOME:-${WORK_ROOT}/huggingface_cache}"

export INSTRUCTPRO2_ROOT WORK_ROOT DATA_ROOT MODEL_ROOT
export PRETRAIN_MODEL_PATH PRETRAIN_MODEL

# --- helpers ---------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || die "${2:-missing file}: $1"
}

require_dir() {
    [ -d "$1" ] || die "${2:-missing directory}: $1"
}

# Locate conda's profile script. CONDA_SH wins; otherwise ask conda itself,
# then fall back to the usual install locations. Miniforge, Miniconda,
# Anaconda and Mambaforge all land somewhere in this list.
ip2_find_conda_sh() {
    if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
        echo "${CONDA_SH}"
        return 0
    fi

    local base
    if base="$(conda info --base 2>/dev/null)" && [ -f "${base}/etc/profile.d/conda.sh" ]; then
        echo "${base}/etc/profile.d/conda.sh"
        return 0
    fi

    local candidate
    for candidate in \
        "${HOME}/conda" "${HOME}/miniforge3" "${HOME}/mambaforge" \
        "${HOME}/miniconda3" "${HOME}/anaconda3" \
        "/opt/conda" "/usr/local/conda"
    do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            echo "${candidate}/etc/profile.d/conda.sh"
            return 0
        fi
    done

    return 1
}

# Activate a conda environment.
#
# Conda's own activation scripts reference unset variables -- MKL's
# activation hook reads MKL_INTERFACE_LAYER before setting it -- so under
# `set -u` sourcing them aborts the job with an error that points at MKL
# rather than at the shell option. Strict mode is suspended across the
# activation only, then restored.
ip2_activate() {
    local env_name="$1"
    local conda_sh
    local had_u=0
    case "$-" in *u*) had_u=1 ;; esac

    conda_sh="$(ip2_find_conda_sh)" || die \
        "could not find conda.sh. Set CONDA_SH to the full path, e.g.
       export CONDA_SH=\$HOME/conda/etc/profile.d/conda.sh"

    set +u
    # shellcheck disable=SC1090
    source "${conda_sh}" || { [ "${had_u}" = 1 ] && set -u; die "failed to source ${conda_sh}"; }
    conda activate "${env_name}" || { [ "${had_u}" = 1 ] && set -u; die \
        "could not activate conda environment '${env_name}'.
       Create it with:  conda env create -f ${INSTRUCTPRO2_ROOT}/instructpro2.yml
       Or point the script at an existing one, e.g.  export GEN_ENV=myenv"; }
    [ "${had_u}" = 1 ] && set -u

    echo "conda env: ${env_name}   ($(python3 --version 2>&1))"
}

# One-line summary of where everything is, printed at job start so a log
# is self-describing when someone reads it a week later.
ip2_print_paths() {
    echo "repo       : ${INSTRUCTPRO2_ROOT}"
    echo "work root  : ${WORK_ROOT}"
    echo "data       : ${DATA_ROOT}"
    echo "models     : ${MODEL_ROOT}"
    echo "HF cache   : ${HF_HOME}"
    echo "host       : $(hostname)"
}