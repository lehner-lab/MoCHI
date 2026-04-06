#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEXTFLOW_ROOT="${REPO_ROOT}/nextflow"
RUN_NAME="${RUN_NAME:-mochi-benchmark-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT%/}/${RUN_NAME}/work}"
MOCHI_VENV="${MOCHI_VENV:-${REPO_ROOT}/.venv}"
MODEL_DESIGN="${MODEL_DESIGN:-/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/model_design_mochi_pool2_abs.txt}"
EXPECTED_DATASET="${EXPECTED_DATASET:-/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/FYN_BIBD_4sG2_mochi_pool2.txt}"
QUEUE="${QUEUE:-gpu-normal}"
RUN_TIME="${RUN_TIME:-12h}"
WORKFLOW_MODE="${WORKFLOW_MODE:-parallel_folds}"
SEED="${SEED:-1}"
K_FOLDS="${K_FOLDS:-10}"
PARALLEL_FOLDS="${PARALLEL_FOLDS:-${K_FOLDS}}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
NUM_EPOCHS_GRID="${NUM_EPOCHS_GRID:-100}"
GRID_MEMORY="${GRID_MEMORY:-8 GB}"
GRID_MEMORY_MAX="${GRID_MEMORY_MAX:-50 GB}"
FOLD_MEMORY="${FOLD_MEMORY:-8 GB}"
FOLD_MEMORY_MAX="${FOLD_MEMORY_MAX:-32 GB}"
MAX_MEMORY_RETRIES="${MAX_MEMORY_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARN_RATE="${LEARN_RATE:-}"
GMODEL="${GMODEL:-}"
MIG_PROFILE="${MIG_PROFILE:-}"
GPU_AFFINITY="${GPU_AFFINITY:-}"
GPU_GMEM="${GPU_GMEM:-}"
GPU_GPACK="${GPU_GPACK:-yes}"
HOST_FILTER="${HOST_FILTER:-}"
RESUME="${RESUME:-0}"

GPU_REQUEST="num=1:mode=shared:j_exclusive=no:gpack=${GPU_GPACK}"
if [ -n "${GMODEL}" ]; then
    GPU_REQUEST="${GPU_REQUEST}:gmodel=${GMODEL}"
fi
if [ -n "${MIG_PROFILE}" ]; then
    GPU_REQUEST="${GPU_REQUEST}:mig=${MIG_PROFILE}"
fi
if [ -n "${GPU_AFFINITY}" ]; then
    GPU_REQUEST="${GPU_REQUEST}:aff=${GPU_AFFINITY}"
fi
if [ -n "${GPU_GMEM}" ]; then
    GPU_REQUEST="${GPU_REQUEST}:gmem=${GPU_GMEM}"
fi

CLUSTER_OPTIONS="-gpu '${GPU_REQUEST}'"
if [ -n "${HOST_FILTER}" ]; then
    CLUSTER_OPTIONS="${CLUSTER_OPTIONS} -m ${HOST_FILTER}"
fi

mkdir -p "${OUTPUT_ROOT%/}/${RUN_NAME}"

if [ ! -x "${MOCHI_VENV}/bin/python" ]; then
    echo "MoCHI environment not found at ${MOCHI_VENV}. Run nextflow/scripts/bootstrap_mochi_uv.sh first." >&2
    exit 1
fi

source /etc/profile.d/modules.sh
module load HGI/common/nextflow/23.10.0

nextflow_args=(
    run "${NEXTFLOW_ROOT}/main.nf"
    -c "${NEXTFLOW_ROOT}/nextflow.config"
    -work-dir "${WORK_DIR}"
    --repo_root "${REPO_ROOT}"
    --nextflow_root "${NEXTFLOW_ROOT}"
    --mochi_venv "${MOCHI_VENV}"
    --model_design "${MODEL_DESIGN}"
    --expected_dataset "${EXPECTED_DATASET}"
    --output_root "${OUTPUT_ROOT}"
    --run_name "${RUN_NAME}"
    --workflow_mode "${WORKFLOW_MODE}"
    --gpu_queue "${QUEUE}"
    --gpu_cluster_options "${CLUSTER_OPTIONS}"
    --grid_memory "${GRID_MEMORY}"
    --grid_memory_max "${GRID_MEMORY_MAX}"
    --fold_memory "${FOLD_MEMORY}"
    --fold_memory_max "${FOLD_MEMORY_MAX}"
    --max_memory_retries "${MAX_MEMORY_RETRIES}"
    --seed "${SEED}"
    --k_folds "${K_FOLDS}"
    --parallel_folds "${PARALLEL_FOLDS}"
    --num_epochs "${NUM_EPOCHS}"
    --num_epochs_grid "${NUM_EPOCHS_GRID}"
    -process.clusterOptions "${CLUSTER_OPTIONS}"
)

if [ -n "${RUN_TIME}" ]; then
    nextflow_args+=(--gpu_time "${RUN_TIME}")
fi

if [ -n "${BATCH_SIZE}" ]; then
    nextflow_args+=(--batch_size "${BATCH_SIZE}")
fi
if [ -n "${LEARN_RATE}" ]; then
    nextflow_args+=(--learn_rate "${LEARN_RATE}")
fi
if [ "${RESUME}" = "1" ]; then
    nextflow_args+=(-resume)
fi

nextflow "${nextflow_args[@]}"
