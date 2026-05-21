#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEXTFLOW_ROOT="${REPO_ROOT}/nextflow"

param_value_from_args() {
    local key="${1}"
    shift
    local arg=""
    local next=""
    while [ "$#" -gt 0 ]; do
        arg="${1}"
        next="${2:-}"
        if [ "${arg}" = "--${key}" ] && [ -n "${next}" ]; then
            printf '%s' "${next}"
            return
        fi
        case "${arg}" in
            --${key}=*)
                printf '%s' "${arg#*=}"
                return
                ;;
        esac
        shift
    done
}

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
    local request="num=1:mode=${mode}:j_exclusive=no:gpack=${GPU_GPACK:-yes}"
    if [ -n "${GMODEL:-}" ]; then
        request="${request}:gmodel=${GMODEL}"
    fi
    if [ -n "${MIG_PROFILE:-}" ]; then
        request="${request}:mig=${MIG_PROFILE}"
    fi
    if [ -n "${GPU_AFFINITY:-}" ]; then
        request="${request}:aff=${GPU_AFFINITY}"
    fi
    if [ -n "${GPU_GMEM:-}" ]; then
        request="${request}:gmem=${GPU_GMEM}"
    fi
    printf '%s' "${request}"
}

build_cluster_options() {
    local gpu_mode="${1}"
    local host_exclude_select="${2}"
    local options="-gpu '$(build_gpu_request "${gpu_mode}")'"
    if [ -n "${host_exclude_select}" ]; then
        options="${options} -R \"select[${host_exclude_select}]\""
    fi
    if [ -n "${HOST_FILTER:-}" ]; then
        options="${options} -m ${HOST_FILTER}"
    fi
    printf '%s' "${options}"
}

run_nextflow_master() {
    local cli_run_name=""
    local cli_output_root=""
    cli_run_name="$(param_value_from_args run_name "$@")"
    cli_output_root="$(param_value_from_args output_root "$@")"

    RUN_NAME="${RUN_NAME:-${cli_run_name:-mochi-benchmark-$(date +%Y%m%d_%H%M%S)}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${cli_output_root:-/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev}}"
    WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT%/}/${RUN_NAME}/work}"
    MOCHI_VENV="${MOCHI_VENV:-${REPO_ROOT}/.venv}"
    RESUME="${RESUME:-0}"

    mkdir -p "${OUTPUT_ROOT%/}/${RUN_NAME}"

    if [ ! -x "${MOCHI_VENV}/bin/python" ]; then
        echo "MoCHI environment not found at ${MOCHI_VENV}. Run bootstrap_mochi_uv.sh from the MoCHI root first." >&2
        exit 1
    fi

    source /etc/profile.d/modules.sh
    module load HGI/common/nextflow/25.10.4

    local queue="${QUEUE:-gpu-normal}"
    local max_memory_retries="${MAX_MEMORY_RETRIES:-3}"
    local parallel_folds="${PARALLEL_FOLDS:-1}"
    local host_exclude_select=""
    local grid_cluster_options=""
    local fold_cluster_options=""
    host_exclude_select="$(build_host_exclude_select "${GPU_HOST_EXCLUDE:-farm22-gpu0203}")"
    grid_cluster_options="$(build_cluster_options "${GRID_GPU_MODE:-shared}" "${host_exclude_select}")"
    fold_cluster_options="$(build_cluster_options "${FOLD_GPU_MODE:-exclusive_process}" "${host_exclude_select}")"
    export MOCHI_GPU_QUEUE="${queue}"
    export MOCHI_GPU_CLUSTER_OPTIONS="${grid_cluster_options}"
    export MOCHI_GRID_GPU_CLUSTER_OPTIONS="${grid_cluster_options}"
    export MOCHI_FOLD_GPU_CLUSTER_OPTIONS="${fold_cluster_options}"
    export MOCHI_MAX_MEMORY_RETRIES="${max_memory_retries}"
    export MOCHI_PARALLEL_FOLDS="${parallel_folds}"

    nextflow_args=(
        run "${NEXTFLOW_ROOT}/main.nf"
        -c "${NEXTFLOW_ROOT}/nextflow.config"
        -work-dir "${WORK_DIR}"
        --repo_root "${REPO_ROOT}"
        --nextflow_root "${NEXTFLOW_ROOT}"
        --mochi_venv "${MOCHI_VENV}"
        --output_root "${OUTPUT_ROOT}"
        --run_name "${RUN_NAME}"
        "$@"
    )

    if [ "${RESUME}" = "1" ]; then
        nextflow_args+=(-resume)
    fi

    nextflow "${nextflow_args[@]}"
}

if [ "${1:-}" = "--run-nextflow-master" ]; then
    shift
    run_nextflow_master "$@"
    exit 0
fi

MASTER_QUEUE="${MASTER_QUEUE:-oversubscribed}"
MASTER_CPUS="${MASTER_CPUS:-1}"
MASTER_MEMORY_GB="${MASTER_MEMORY_GB:-4}"
MASTER_MEMORY_MB="${MASTER_MEMORY_MB:-$((MASTER_MEMORY_GB * 1024))}"

RUN_NAME="${RUN_NAME:-mochi-benchmark-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev}"
LOG_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
RESUME="${RESUME:-0}"
export RUN_NAME OUTPUT_ROOT RESUME

mkdir -p "${LOG_DIR}"

if [ "${RESUME}" = "1" ]; then
    DEFAULT_JOB_NAME="${RUN_NAME}-resume-nf"
else
    DEFAULT_JOB_NAME="${RUN_NAME}-nf"
fi
JOB_NAME="${JOB_NAME:-${DEFAULT_JOB_NAME}}"

JOB_SCRIPT="${LOG_DIR}/nextflow-master.submit.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${REPO_ROOT}"
    echo 'args=()'
    for arg in "$@"; do
        printf 'args+=(%q)\n' "${arg}"
    done
    printf 'bash %q --run-nextflow-master "${args[@]}"\n' "${BASH_SOURCE[0]}"
} > "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

echo "Submitting Nextflow master job"
echo "  queue: ${MASTER_QUEUE}"
echo "  cpus: ${MASTER_CPUS}"
echo "  memory_mb: ${MASTER_MEMORY_MB}"
echo "  job_name: ${JOB_NAME}"
echo "  logs: ${LOG_DIR}/nextflow-master.%J.{log,err}"
echo "  submit_script: ${JOB_SCRIPT}"

bsub \
    -env all \
    -q "${MASTER_QUEUE}" \
    -n "${MASTER_CPUS}" \
    -M "${MASTER_MEMORY_MB}" \
    -R "select[mem>${MASTER_MEMORY_MB}] rusage[mem=${MASTER_MEMORY_MB}] span[hosts=1]" \
    -J "${JOB_NAME}" \
    -o "${LOG_DIR}/nextflow-master.%J.log" \
    -e "${LOG_DIR}/nextflow-master.%J.err" \
    bash "${JOB_SCRIPT}"
