process RUN_GRID_SEARCH_CONDITION {
    label "mochi_grid_gpu"

    input:
    tuple val(condition), val(batchSize), val(learnRate), val(l1Factor), val(l2Factor)

    output:
    tuple val(condition), path("grid_condition_${condition}.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/grid_search/condition_${condition}"
    def cacheDir = "${params.cache_root}/${params.run_name}/grid_search/condition_${condition}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${params.run_name}-grid-${condition}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${jobOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="grid_search"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${batchSize}"
    export LEARN_RATE="${learnRate}"
    export L1_REGULARIZATION_FACTOR="${l1Factor}"
    export L2_REGULARIZATION_FACTOR="${l2Factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    touch "grid_condition_${condition}.done"
    """
}

process MERGE_GRID_SEARCH_CONDITIONS {
    label "summary_lsf_cpu"

    input:
    path grid_condition_done

    output:
    path "grid_search.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/grid_search_merge"
    def cacheDir = "${params.cache_root}/${params.run_name}/grid_search_merge"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${params.run_name}-grid-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="merge_grid_search"
    export MOCHI_DEVICE="cpu"
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
    label "summary_lsf_cpu"

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
    export RUN_LABEL="${params.run_name}-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="merge_folds"
    export MOCHI_DEVICE="cpu"
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

process RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION {
    label "mochi_grid_gpu"

    input:
    tuple val(stage), val(condition), val(batchSize), val(learnRate), val(l1Factor), val(l2Factor), path(prev_stage_ready)

    output:
    tuple val(stage), val(condition), path("sparse_stage_${stage}_grid_condition_${condition}.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}"
    def cacheDir = "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search/condition_${condition}"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${params.run_name}-stage-${stage}-grid-${condition}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${jobOutputDir}"
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
    export BATCH_SIZE="${batchSize}"
    export LEARN_RATE="${learnRate}"
    export L1_REGULARIZATION_FACTOR="${l1Factor}"
    export L2_REGULARIZATION_FACTOR="${l2Factor}"
    export SPARSE_METHOD="${params.sparse_method}"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    touch "sparse_stage_${stage}_grid_condition_${condition}.done"
    """
}

process MERGE_SPARSE_STAGE_GRID_SEARCH {
    label "summary_lsf_cpu"

    input:
    val stage
    path grid_condition_done

    output:
    tuple val(stage), path("sparse_stage_${stage}_grid.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = "${runOutputDir}/stage_${stage}/grid_search_merge"
    def cacheDir = "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search_merge"
    """
    mkdir -p "${jobOutputDir}"
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${params.run_name}-stage-${stage}-grid-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="sparse_merge_grid_search"
    export MOCHI_DEVICE="cpu"
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
    label "summary_lsf_cpu"

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
    export RUN_LABEL="${params.run_name}-stage-${stage}-merge"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${runOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="sparse_merge_folds"
    export MOCHI_DEVICE="cpu"
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
