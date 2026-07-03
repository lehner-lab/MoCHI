#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MOCHI_REPO="${MOCHI_REPO:-${REPO_ROOT}}"
MOCHI_VENV="${MOCHI_VENV:-${MOCHI_REPO}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${MOCHI_VENV}/bin/python}"
MOCHI_ARGS_FILE="${MOCHI_ARGS_FILE:-}"
RUN_LABEL="${RUN_LABEL:-mochi_batch_compare}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT%/}/${RUN_LABEL}}"
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

mkdir -p "${OUTPUT_DIR}" "${LOCAL_UV_CACHE}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Expected Python interpreter not found at ${PYTHON_BIN}" >&2
    echo "Run bootstrap_mochi_uv.sh from the MoCHI root first." >&2
    exit 1
fi

export MOCHI_AMP="${MOCHI_AMP:-auto}"
export MOCHI_DEVICE="${MOCHI_DEVICE:-cuda}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${LOCAL_UV_CACHE}"
export XDG_CACHE_HOME="${LOCAL_UV_CACHE}"

MOCHI_CMD=(
    "${PYTHON_BIN}"
    "${MOCHI_REPO}/pymochi/bin/run_mochi.py"
)

if [ -n "${MOCHI_ARGS_FILE}" ]; then
    if [ ! -f "${MOCHI_ARGS_FILE}" ]; then
        echo "MOCHI_ARGS_FILE not found at ${MOCHI_ARGS_FILE}" >&2
        exit 1
    fi
    while IFS= read -r arg || [ -n "${arg}" ]; do
        [ -z "${arg}" ] && continue
        MOCHI_CMD+=("${arg}")
    done < "${MOCHI_ARGS_FILE}"
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
    echo "output_dir=${OUTPUT_DIR}"
    echo "mochi_args_file=${MOCHI_ARGS_FILE}"
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
    df -h "${OUTPUT_DIR}" || true
} | tee "${JOB_META}"

{
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "RUN_LOG=${RUN_LOG}"
    echo "TIME_LOG=${TIME_LOG}"
    echo "RESOURCE_LOG=${RESOURCE_LOG}"
    echo "JOB_META=${JOB_META}"
    echo "STATUS_FILE=${STATUS_FILE}"
    echo "COMMAND_FILE=${COMMAND_FILE}"
    echo "PHASE_MANIFEST=${PHASE_MANIFEST}"
    echo "MOCHI_DEVICE=${MOCHI_DEVICE}"
    echo "START_TIME=$(date -Is)"
} > "${RUN_INFO_FILE}"

{
    echo "MOCHI_ARGS_FILE=${MOCHI_ARGS_FILE}"
    echo "COMMAND_FILE=${COMMAND_FILE}"
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
