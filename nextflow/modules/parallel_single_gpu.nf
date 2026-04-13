process RUN_GRID_SEARCH {
    label "mochi_grid_gpu"

    output:
    path "grid_search.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/grid_search"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
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
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
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
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
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
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
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
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
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
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    cp "${jobOutputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}

process RUN_SPARSE_STAGE_GRID_SEARCH {
    label "mochi_grid_gpu"

    input:
    tuple val(stage), path(prev_stage_ready)

    output:
    tuple val(stage), path("sparse_stage_${stage}_grid.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/stage_${stage}/grid_search"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}-stage-${stage}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="sparse_grid_search"
    export SPARSE_STAGE_INDEX="${stage}"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    touch "sparse_stage_${stage}_grid.done"
    """
}

process RUN_SPARSE_STAGE_FOLD {
    label "mochi_fold_gpu"

    input:
    tuple val(stage), val(fold), path(grid_ready)

    output:
    path "sparse_stage_${stage}_fold_${fold}.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/stage_${stage}/fold_${fold}"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}-stage-${stage}-fold-${fold}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="sparse_fit_best"
    export SPARSE_STAGE_INDEX="${stage}"
    export MOCHI_SEED="${params.seed}"
    export MOCHI_FOLD="${fold}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    touch "sparse_stage_${stage}_fold_${fold}.done"
    """
}

process RUN_SPARSE_STAGE_MERGE {
    label "summary_local"

    input:
    tuple val(stage), path(grid_ready)
    path fold_done

    output:
    tuple val(stage), path("sparse_stage_${stage}_merge.done"), path("benchmark_manifest.env")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/stage_${stage}/merge"
    def cacheDir = "${params.cache_root}/${params.run_name}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export EXPECTED_DATASET="${params.expected_dataset}"
    export RUN_LABEL="${params.run_name}-stage-${stage}-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="sparse_merge_folds"
    export SPARSE_STAGE_INDEX="${stage}"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    touch "sparse_stage_${stage}_merge.done"
    cp "${jobOutputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}
