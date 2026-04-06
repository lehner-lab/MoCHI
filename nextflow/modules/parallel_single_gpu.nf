process RUN_GRID_SEARCH {
    label "mochi_grid_gpu"

    output:
    path "grid_search.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/grid_search"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    def mochiRepo = params.containsKey("mochi_repo") ? params["mochi_repo"] : "${params.repo_root}/MoCHI"
    def nextflowRoot = params.containsKey("nextflow_root") ? params["nextflow_root"] : "${mochiRepo}/nextflow"
    def runnerScript = params.containsKey("runner_script") ? params["runner_script"] : "${nextflowRoot}/scripts/run_mochi_lsf_gpu.sh"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${mochiRepo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="grid_search"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"

    "${runnerScript}"
    touch "grid_search.done"
    """
}

process RUN_FOLD {
    label "mochi_fold_gpu"

    input:
    tuple val(fold), path(grid_ready)

    output:
    path "fold_${fold}.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/fold_${fold}"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    def mochiRepo = params.containsKey("mochi_repo") ? params["mochi_repo"] : "${params.repo_root}/MoCHI"
    def nextflowRoot = params.containsKey("nextflow_root") ? params["nextflow_root"] : "${mochiRepo}/nextflow"
    def runnerScript = params.containsKey("runner_script") ? params["runner_script"] : "${nextflowRoot}/scripts/run_mochi_lsf_gpu.sh"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${mochiRepo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}-fold-${fold}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="fit_best"
    export MOCHI_SEED="${params.seed}"
    export MOCHI_FOLD="${fold}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"

    "${runnerScript}"
    touch "fold_${fold}.done"
    """
}

process MERGE_FOLDS {
    label "summary_local"

    input:
    path grid_ready
    path fold_done

    output:
    path "benchmark_manifest.env"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/merge"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    def mochiRepo = params.containsKey("mochi_repo") ? params["mochi_repo"] : "${params.repo_root}/MoCHI"
    def nextflowRoot = params.containsKey("nextflow_root") ? params["nextflow_root"] : "${mochiRepo}/nextflow"
    def runnerScript = params.containsKey("runner_script") ? params["runner_script"] : "${nextflowRoot}/scripts/run_mochi_lsf_gpu.sh"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${mochiRepo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="merge_folds"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"

    "${runnerScript}"
    cp "${jobOutputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}
