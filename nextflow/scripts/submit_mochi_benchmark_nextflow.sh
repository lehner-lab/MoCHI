#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEXTFLOW_ROOT="${REPO_ROOT}/nextflow"
RUN_NAME="${RUN_NAME:-mochi-benchmark-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT%/}/${RUN_NAME}/work}"
MOCHI_VENV="${MOCHI_VENV:-${REPO_ROOT}/.venv}"
MODEL_DESIGN="${MODEL_DESIGN:-/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/model_design_mochi_pool2_abs.txt}"
QUEUE="${QUEUE:-gpu-normal}"
CPU_QUEUE="${CPU_QUEUE:-normal}"
WORKFLOW_MODE="${WORKFLOW_MODE:-parallel_folds}"
SEED="${SEED:-1}"
K_FOLDS="${K_FOLDS:-10}"
PARALLEL_FOLDS="${PARALLEL_FOLDS:-${K_FOLDS}}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
NUM_EPOCHS_GRID="${NUM_EPOCHS_GRID:-100}"
GRID_MEMORY="${GRID_MEMORY:-24 GB}"
GRID_MEMORY_MAX="${GRID_MEMORY_MAX:-50 GB}"
FOLD_MEMORY="${FOLD_MEMORY:-24 GB}"
FOLD_MEMORY_MAX="${FOLD_MEMORY_MAX:-50 GB}"
MERGE_MEMORY="${MERGE_MEMORY:-24 GB}"
MERGE_MEMORY_MAX="${MERGE_MEMORY_MAX:-50 GB}"
MAX_MEMORY_RETRIES="${MAX_MEMORY_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARN_RATE="${LEARN_RATE:-}"
L1_REGULARIZATION_FACTOR="${L1_REGULARIZATION_FACTOR:-}"
L2_REGULARIZATION_FACTOR="${L2_REGULARIZATION_FACTOR:-}"
SPARSE_METHOD="${SPARSE_METHOD:-}"
GMODEL="${GMODEL:-}"
MIG_PROFILE="${MIG_PROFILE:-}"
GPU_AFFINITY="${GPU_AFFINITY:-}"
GPU_GMEM="${GPU_GMEM:-}"
GPU_GPACK="${GPU_GPACK:-yes}"
GRID_GPU_MODE="${GRID_GPU_MODE:-shared}"
FOLD_GPU_MODE="${FOLD_GPU_MODE:-exclusive_process}"
HOST_FILTER="${HOST_FILTER:-}"
GPU_HOST_EXCLUDE="${GPU_HOST_EXCLUDE:-farm22-gpu0203}"
RESUME="${RESUME:-0}"

build_host_exclude_select() {
    local raw_hosts="${1:-}"
    local expr=""
    local host=""
    for host in ${raw_hosts//,/ }; do
        if [ -n "${host}" ]; then
            if [ -n "${expr}" ]; then
                expr="${expr} && "
            fi
            expr="${expr}hname!='${host}'"
        fi
    done
    printf '%s' "${expr}"
}

build_gpu_request() {
    local mode="${1}"
    local request="num=1:mode=${mode}:j_exclusive=no:gpack=${GPU_GPACK}"
    if [ -n "${GMODEL}" ]; then
        request="${request}:gmodel=${GMODEL}"
    fi
    if [ -n "${MIG_PROFILE}" ]; then
        request="${request}:mig=${MIG_PROFILE}"
    fi
    if [ -n "${GPU_AFFINITY}" ]; then
        request="${request}:aff=${GPU_AFFINITY}"
    fi
    if [ -n "${GPU_GMEM}" ]; then
        request="${request}:gmem=${GPU_GMEM}"
    fi
    printf '%s' "${request}"
}

build_cluster_options() {
    local gpu_mode="${1}"
    local options="-gpu '$(build_gpu_request "${gpu_mode}")'"
    if [ -n "${HOST_EXCLUDE_SELECT}" ]; then
        options="${options} -R \"select[${HOST_EXCLUDE_SELECT}]\""
    fi
    if [ -n "${HOST_FILTER}" ]; then
        options="${options} -m ${HOST_FILTER}"
    fi
    printf '%s' "${options}"
}

HOST_EXCLUDE_SELECT="$(build_host_exclude_select "${GPU_HOST_EXCLUDE}")"
GRID_CLUSTER_OPTIONS="$(build_cluster_options "${GRID_GPU_MODE}")"
FOLD_CLUSTER_OPTIONS="$(build_cluster_options "${FOLD_GPU_MODE}")"

mkdir -p "${OUTPUT_ROOT%/}/${RUN_NAME}"

if [ ! -x "${MOCHI_VENV}/bin/python" ]; then
    echo "MoCHI environment not found at ${MOCHI_VENV}. Run bootstrap_mochi_uv.sh from the MoCHI root first." >&2
    exit 1
fi

source /etc/profile.d/modules.sh
module load HGI/common/nextflow/25.10.4

export MOCHI_GPU_QUEUE="${QUEUE}"
export MOCHI_GPU_CLUSTER_OPTIONS="${GRID_CLUSTER_OPTIONS}"
export MOCHI_GRID_GPU_CLUSTER_OPTIONS="${GRID_CLUSTER_OPTIONS}"
export MOCHI_FOLD_GPU_CLUSTER_OPTIONS="${FOLD_CLUSTER_OPTIONS}"
export MOCHI_MAX_MEMORY_RETRIES="${MAX_MEMORY_RETRIES}"
export MOCHI_PARALLEL_FOLDS="${PARALLEL_FOLDS}"

nextflow_args=(
    run "${NEXTFLOW_ROOT}/main.nf"
    -c "${NEXTFLOW_ROOT}/nextflow.config"
    -work-dir "${WORK_DIR}"
    --repo_root "${REPO_ROOT}"
    --nextflow_root "${NEXTFLOW_ROOT}"
    --mochi_venv "${MOCHI_VENV}"
    --model_design "${MODEL_DESIGN}"
    --output_root "${OUTPUT_ROOT}"
    --run_name "${RUN_NAME}"
    --workflow_mode "${WORKFLOW_MODE}"
    --gpu_queue "${QUEUE}"
    --cpu_queue "${CPU_QUEUE}"
    --gpu_cluster_options "${GRID_CLUSTER_OPTIONS}"
    --grid_gpu_cluster_options "${GRID_CLUSTER_OPTIONS}"
    --fold_gpu_cluster_options "${FOLD_CLUSTER_OPTIONS}"
    --grid_memory "${GRID_MEMORY}"
    --grid_memory_max "${GRID_MEMORY_MAX}"
    --fold_memory "${FOLD_MEMORY}"
    --fold_memory_max "${FOLD_MEMORY_MAX}"
    --merge_memory "${MERGE_MEMORY}"
    --merge_memory_max "${MERGE_MEMORY_MAX}"
    --max_memory_retries "${MAX_MEMORY_RETRIES}"
    --seed "${SEED}"
    --k_folds "${K_FOLDS}"
    --parallel_folds "${PARALLEL_FOLDS}"
    --num_epochs "${NUM_EPOCHS}"
    --num_epochs_grid "${NUM_EPOCHS_GRID}"
)

if [ -n "${BATCH_SIZE}" ]; then
    nextflow_args+=(--batch_size "${BATCH_SIZE}")
fi
if [ -n "${LEARN_RATE}" ]; then
    nextflow_args+=(--learn_rate "${LEARN_RATE}")
fi
if [ -n "${L1_REGULARIZATION_FACTOR}" ]; then
    nextflow_args+=(--l1_regularization_factor "${L1_REGULARIZATION_FACTOR}")
fi
if [ -n "${L2_REGULARIZATION_FACTOR}" ]; then
    nextflow_args+=(--l2_regularization_factor "${L2_REGULARIZATION_FACTOR}")
fi
if [ -n "${SPARSE_METHOD}" ]; then
    nextflow_args+=(--sparse_method "${SPARSE_METHOD}")
fi
if [ "${RESUME}" = "1" ]; then
    nextflow_args+=(-resume)
fi

nextflow "${nextflow_args[@]}"
