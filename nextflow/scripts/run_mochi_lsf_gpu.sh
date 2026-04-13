#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MOCHI_REPO="${MOCHI_REPO:-${REPO_ROOT}}"
MOCHI_VENV="${MOCHI_VENV:-${MOCHI_REPO}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${MOCHI_VENV}/bin/python}"
MODEL_DESIGN="${MODEL_DESIGN:-${REPO_ROOT}/model_design_all_programmed_variants_abs_g1234567.txt}"
EXPECTED_DATASET="${EXPECTED_DATASET:-}"
RUN_LABEL="${RUN_LABEL:-mochi_batch_compare}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT%/}/${RUN_LABEL}}"
MOCHI_OUTPUT_DIRECTORY="${MOCHI_OUTPUT_DIRECTORY:-${OUTPUT_DIR}}"
PROJECT_NAME="${PROJECT_NAME:-mochi_project}"
MAX_INTERACTION_ORDER="${MAX_INTERACTION_ORDER:-2}"
MOCHI_PHASE="${MOCHI_PHASE:-full}"
SPARSE_STAGE_INDEX="${SPARSE_STAGE_INDEX:-}"
MOCHI_SEED="${MOCHI_SEED:-1}"
MOCHI_FOLD="${MOCHI_FOLD:-}"
K_FOLDS="${K_FOLDS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
NUM_EPOCHS_GRID="${NUM_EPOCHS_GRID:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARN_RATE="${LEARN_RATE:-}"
L1_REGULARIZATION_FACTOR="${L1_REGULARIZATION_FACTOR:-}"
L2_REGULARIZATION_FACTOR="${L2_REGULARIZATION_FACTOR:-}"
SPARSE_METHOD="${SPARSE_METHOD:-}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT%/}/cache}"
LOCAL_UV_CACHE="${LOCAL_UV_CACHE:-${OUTPUT_DIR}/uv-cache}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"
TIME_LOG="${TIME_LOG:-${OUTPUT_DIR}/time.log}"
RESOURCE_LOG="${RESOURCE_LOG:-${OUTPUT_DIR}/resource_snapshots.log}"
JOB_META="${JOB_META:-${OUTPUT_DIR}/job_metadata.txt}"
RUN_INFO_FILE="${RUN_INFO_FILE:-${OUTPUT_DIR}/run_info.env}"
STATUS_FILE="${STATUS_FILE:-${OUTPUT_DIR}/job_status.txt}"
COMMAND_FILE="${COMMAND_FILE:-${OUTPUT_DIR}/job_command.txt}"
PID_FILE="${PID_FILE:-${OUTPUT_DIR}/mochi.pid}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-${OUTPUT_DIR}/benchmark_manifest.env}"
PHASE_MANIFEST="${PHASE_MANIFEST:-${OUTPUT_DIR}/phase_manifest.env}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${LOCAL_UV_CACHE}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Expected Python interpreter not found at ${PYTHON_BIN}" >&2
    echo "Run nextflow/scripts/bootstrap_mochi_uv.sh first." >&2
    exit 1
fi

if [ ! -f "${MODEL_DESIGN}" ]; then
    echo "Model design file not found at ${MODEL_DESIGN}" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${MODEL_DESIGN}" "${EXPECTED_DATASET}" <<'PY'
import csv
import sys
from pathlib import Path

model_design = Path(sys.argv[1])
expected_dataset = sys.argv[2].strip()

with model_design.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    files = [row["file"].strip() for row in reader if row.get("file")]

if not files:
    raise SystemExit(f"No dataset entries found in {model_design}")

missing = [path for path in files if not Path(path).exists()]
if missing:
    raise SystemExit(
        "The following dataset paths referenced by the model design do not exist:\n"
        + "\n".join(missing)
    )

if expected_dataset and expected_dataset not in files:
    raise SystemExit(
        f"Expected dataset {expected_dataset} was not found in {model_design}"
    )
PY

export MOCHI_FEATURES_UINT8="${MOCHI_FEATURES_UINT8:-1}"
export MOCHI_AMP="${MOCHI_AMP:-auto}"
export MOCHI_DEVICE="${MOCHI_DEVICE:-cuda}"
export MOCHI_XOHI_CACHE_DIR="${CACHE_DIR}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${LOCAL_UV_CACHE}"
export XDG_CACHE_HOME="${LOCAL_UV_CACHE}"

MOCHI_CMD=(
    "${PYTHON_BIN}"
    "${MOCHI_REPO}/pymochi/bin/run_mochi.py"
    --model_design "${MODEL_DESIGN}"
    --output_directory "${MOCHI_OUTPUT_DIRECTORY}"
    --project_name "${PROJECT_NAME}"
    --max_interaction_order "${MAX_INTERACTION_ORDER}"
    --seed "${MOCHI_SEED}"
    --phase "${MOCHI_PHASE}"
)

if [ -n "${MOCHI_FOLD}" ]; then
    MOCHI_CMD+=(--fold "${MOCHI_FOLD}")
fi
if [ -n "${SPARSE_STAGE_INDEX}" ]; then
    MOCHI_CMD+=(--stage_index "${SPARSE_STAGE_INDEX}")
fi
if [ -n "${K_FOLDS}" ]; then
    MOCHI_CMD+=(--k_folds "${K_FOLDS}")
fi
if [ -n "${NUM_EPOCHS}" ]; then
    MOCHI_CMD+=(--num_epochs "${NUM_EPOCHS}")
fi
if [ -n "${NUM_EPOCHS_GRID}" ]; then
    MOCHI_CMD+=(--num_epochs_grid "${NUM_EPOCHS_GRID}")
fi
if [ -n "${BATCH_SIZE}" ]; then
    MOCHI_CMD+=(--batch_size "${BATCH_SIZE}")
fi
if [ -n "${LEARN_RATE}" ]; then
    MOCHI_CMD+=(--learn_rate "${LEARN_RATE}")
fi
if [ -n "${L1_REGULARIZATION_FACTOR}" ]; then
    MOCHI_CMD+=(--l1_regularization_factor "${L1_REGULARIZATION_FACTOR}")
fi
if [ -n "${L2_REGULARIZATION_FACTOR}" ]; then
    MOCHI_CMD+=(--l2_regularization_factor "${L2_REGULARIZATION_FACTOR}")
fi
if [ -n "${SPARSE_METHOD}" ]; then
    MOCHI_CMD+=(--sparse_method "${SPARSE_METHOD}")
fi

printf '%q ' "${MOCHI_CMD[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

{
    echo "job_id=${LSB_JOBID:-unknown}"
    echo "job_name=${LSB_JOBNAME:-unknown}"
    echo "queue=${LSB_QUEUE:-unknown}"
    echo "host=$(hostname -f || hostname)"
    echo "cwd=$(pwd)"
    echo "start_time=$(date -Is)"
    echo "repo_root=${REPO_ROOT}"
    echo "mochi_repo=${MOCHI_REPO}"
    echo "model_design=${MODEL_DESIGN}"
    echo "expected_dataset=${EXPECTED_DATASET}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "mochi_output_directory=${MOCHI_OUTPUT_DIRECTORY}"
    echo "cache_dir=${CACHE_DIR}"
    echo "project_name=${PROJECT_NAME}"
    echo "max_interaction_order=${MAX_INTERACTION_ORDER}"
    echo "mochi_phase=${MOCHI_PHASE}"
    echo "sparse_stage_index=${SPARSE_STAGE_INDEX}"
    echo "mochi_seed=${MOCHI_SEED}"
    echo "mochi_fold=${MOCHI_FOLD}"
    echo "k_folds=${K_FOLDS}"
    echo "num_epochs=${NUM_EPOCHS}"
    echo "num_epochs_grid=${NUM_EPOCHS_GRID}"
    echo "l1_regularization_factor=${L1_REGULARIZATION_FACTOR}"
    echo "l2_regularization_factor=${L2_REGULARIZATION_FACTOR}"
    echo "sparse_method=${SPARSE_METHOD}"
    echo "python=${PYTHON_BIN}"
    echo "python_version=$("${PYTHON_BIN}" -c 'import sys; print(sys.version.replace("\n", " "))')"
    echo "torch_version=$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)')"
    echo "torch_cuda_build=$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda)')"
    echo "MOCHI_DEVICE=${MOCHI_DEVICE}"
    echo "gpu_devices:"
    nvidia-smi -L || true
    echo "memory:"
    free -h || true
    echo "filesystems:"
    df -h "${OUTPUT_DIR}" "${CACHE_DIR}" || true
} | tee "${JOB_META}"

{
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "RUN_LOG=${RUN_LOG}"
    echo "TIME_LOG=${TIME_LOG}"
    echo "RESOURCE_LOG=${RESOURCE_LOG}"
    echo "JOB_META=${JOB_META}"
    echo "STATUS_FILE=${STATUS_FILE}"
    echo "COMMAND_FILE=${COMMAND_FILE}"
    echo "MODEL_DESIGN=${MODEL_DESIGN}"
    echo "EXPECTED_DATASET=${EXPECTED_DATASET}"
    echo "PROJECT_NAME=${PROJECT_NAME}"
    echo "MAX_INTERACTION_ORDER=${MAX_INTERACTION_ORDER}"
    echo "MOCHI_OUTPUT_DIRECTORY=${MOCHI_OUTPUT_DIRECTORY}"
    echo "MOCHI_PHASE=${MOCHI_PHASE}"
    echo "SPARSE_STAGE_INDEX=${SPARSE_STAGE_INDEX}"
    echo "MOCHI_SEED=${MOCHI_SEED}"
    echo "MOCHI_FOLD=${MOCHI_FOLD}"
    echo "K_FOLDS=${K_FOLDS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "NUM_EPOCHS_GRID=${NUM_EPOCHS_GRID}"
    echo "L1_REGULARIZATION_FACTOR=${L1_REGULARIZATION_FACTOR}"
    echo "L2_REGULARIZATION_FACTOR=${L2_REGULARIZATION_FACTOR}"
    echo "SPARSE_METHOD=${SPARSE_METHOD}"
    echo "PHASE_MANIFEST=${PHASE_MANIFEST}"
    echo "MOCHI_DEVICE=${MOCHI_DEVICE}"
    echo "START_TIME=$(date -Is)"
} > "${RUN_INFO_FILE}"

{
    echo "MOCHI_PHASE=${MOCHI_PHASE}"
    echo "SPARSE_STAGE_INDEX=${SPARSE_STAGE_INDEX}"
    echo "MOCHI_SEED=${MOCHI_SEED}"
    echo "MOCHI_FOLD=${MOCHI_FOLD}"
    echo "MOCHI_OUTPUT_DIRECTORY=${MOCHI_OUTPUT_DIRECTORY}"
    echo "MOCHI_PROJECT_DIR=${MOCHI_OUTPUT_DIRECTORY%/}/${PROJECT_NAME}"
    echo "MOCHI_TASK_DIR=${MOCHI_OUTPUT_DIRECTORY%/}/${PROJECT_NAME}/task_${MOCHI_SEED}"
    if [ -n "${MOCHI_FOLD}" ]; then
        echo "MOCHI_FOLD_DIR=${MOCHI_OUTPUT_DIRECTORY%/}/${PROJECT_NAME}/task_${MOCHI_SEED}/fold_${MOCHI_FOLD}"
    fi
} > "${PHASE_MANIFEST}"

monitor_resources() {
    while kill -0 "${MOCHI_PID}" 2>/dev/null; do
        {
            echo "timestamp=$(date -Is)"
            echo "pid_snapshot:"
            ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,state,cmd -p "${MOCHI_PID}" || true
            if [ -r "/proc/${MOCHI_PID}/status" ]; then
                echo "proc_status:"
                awk '/^(VmPeak|VmHWM|VmRSS|Threads|State):/' "/proc/${MOCHI_PID}/status" || true
            fi
            child_pids="$(pgrep -P "${MOCHI_PID}" || true)"
            if [ -n "${child_pids}" ]; then
                echo "child_snapshot:"
                ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,state,cmd -p ${child_pids} || true
            fi
            echo "gpu_snapshot:"
            nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits || true
            echo "gpu_processes:"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
            echo
        } >> "${RESOURCE_LOG}" 2>&1
        sleep "${MONITOR_INTERVAL_SECONDS}"
    done
}

{
    echo "[$(date -Is)] Starting MoCHI command"
    printf '%s\n' "$(sed 's/[[:space:]]*$//' "${COMMAND_FILE}")"
} >> "${RUN_LOG}"

set +e
(
    exec /usr/bin/time -v -o "${TIME_LOG}" "${MOCHI_CMD[@]}" >> "${RUN_LOG}" 2>&1
) &
MOCHI_PID=$!
set -e

printf '%s\n' "${MOCHI_PID}" > "${PID_FILE}"
printf 'MOCHI_PID=%s\n' "${MOCHI_PID}" >> "${RUN_INFO_FILE}"
printf 'PID_FILE=%s\n' "${PID_FILE}" >> "${RUN_INFO_FILE}"

monitor_resources &
MONITOR_PID=$!

set +e
wait "${MOCHI_PID}"
MOCHI_EXIT=$?
set -e

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

END_TIME="$(date -Is)"
END_EPOCH="$(date +%s)"
START_EPOCH="$(date -d "$(awk -F= '/^start_time=/{print $2; exit}' "${JOB_META}")" +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

{
    echo "end_time=${END_TIME}"
    echo "elapsed_seconds=${ELAPSED_SECONDS}"
    echo "exit_code=${MOCHI_EXIT}"
} >> "${JOB_META}"

{
    echo "job_id=${LSB_JOBID:-unknown}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "time_log=${TIME_LOG}"
    echo "resource_log=${RESOURCE_LOG}"
    echo "job_meta=${JOB_META}"
    echo "exit_code=${MOCHI_EXIT}"
    echo "end_time=${END_TIME}"
    echo "elapsed_seconds=${ELAPSED_SECONDS}"
} > "${STATUS_FILE}"

{
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "TIME_LOG=${TIME_LOG}"
    echo "RESOURCE_LOG=${RESOURCE_LOG}"
    echo "JOB_META=${JOB_META}"
    echo "STATUS_FILE=${STATUS_FILE}"
    echo "RUN_LOG=${RUN_LOG}"
    echo "EXIT_CODE=${MOCHI_EXIT}"
    echo "ELAPSED_SECONDS=${ELAPSED_SECONDS}"
    echo "PHASE_MANIFEST=${PHASE_MANIFEST}"
} > "${BENCHMARK_MANIFEST}"

exit "${MOCHI_EXIT}"
