#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT_SCRIPT="${REPO_ROOT}/nextflow/scripts/submit_mochi_benchmark_nextflow.sh"

MASTER_QUEUE="${MASTER_QUEUE:-oversubscribed}"
MASTER_CPUS="${MASTER_CPUS:-1}"
MASTER_WALLTIME="${MASTER_WALLTIME:-72:00}"
MASTER_MEMORY_GB="${MASTER_MEMORY_GB:-24}"
MASTER_MEMORY_MB="${MASTER_MEMORY_MB:-$((MASTER_MEMORY_GB * 1024))}"

RUN_NAME="${RUN_NAME:-mochi-benchmark-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev}"
LOG_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
RESUME="${RESUME:-0}"

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
    echo "Submit script not found at ${SUBMIT_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

if [ "${RESUME}" = "1" ]; then
    DEFAULT_JOB_NAME="${RUN_NAME}-resume-nf"
else
    DEFAULT_JOB_NAME="${RUN_NAME}-nf"
fi
JOB_NAME="${JOB_NAME:-${DEFAULT_JOB_NAME}}"

passthrough_vars=(
    RUN_NAME
    OUTPUT_ROOT
    WORK_DIR
    MOCHI_VENV
    MODEL_DESIGN
    QUEUE
    CPU_QUEUE
    WORKFLOW_MODE
    SEED
    K_FOLDS
    PARALLEL_FOLDS
    NUM_EPOCHS
    NUM_EPOCHS_GRID
    GRID_MEMORY
    GRID_MEMORY_MAX
    FOLD_MEMORY
    FOLD_MEMORY_MAX
    MERGE_MEMORY
    MERGE_MEMORY_MAX
    MAX_MEMORY_RETRIES
    BATCH_SIZE
    LEARN_RATE
    L1_REGULARIZATION_FACTOR
    L2_REGULARIZATION_FACTOR
    SPARSE_METHOD
    GMODEL
    MIG_PROFILE
    GPU_AFFINITY
    GPU_GMEM
    GPU_GPACK
    GRID_GPU_MODE
    FOLD_GPU_MODE
    HOST_FILTER
    GPU_HOST_EXCLUDE
    RESUME
)

env_assignments=()
for name in "${passthrough_vars[@]}"; do
    if [ -v "${name}" ]; then
        env_assignments+=("${name}=${!name}")
    fi
done

JOB_SCRIPT="${LOG_DIR}/nextflow-master.submit.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${REPO_ROOT}"
    for assignment in "${env_assignments[@]}"; do
        printf 'export %q\n' "${assignment}"
    done
    printf 'bash %q\n' "${SUBMIT_SCRIPT}"
} > "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

echo "Submitting Nextflow master job"
echo "  queue: ${MASTER_QUEUE}"
echo "  cpus: ${MASTER_CPUS}"
echo "  memory_mb: ${MASTER_MEMORY_MB}"
echo "  walltime: ${MASTER_WALLTIME}"
echo "  job_name: ${JOB_NAME}"
echo "  logs: ${LOG_DIR}/nextflow-master.%J.{log,err}"
echo "  submit_script: ${JOB_SCRIPT}"

bsub \
    -q "${MASTER_QUEUE}" \
    -n "${MASTER_CPUS}" \
    -M "${MASTER_MEMORY_MB}" \
    -R "select[mem>${MASTER_MEMORY_MB}] rusage[mem=${MASTER_MEMORY_MB}] span[hosts=1]" \
    -W "${MASTER_WALLTIME}" \
    -J "${JOB_NAME}" \
    -o "${LOG_DIR}/nextflow-master.%J.log" \
    -e "${LOG_DIR}/nextflow-master.%J.err" \
    bash "${JOB_SCRIPT}"
