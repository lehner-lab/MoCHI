def formatArgValue = { value ->
    value == null ? "" : value.toString().replace("'", "'\"'\"'")
}

def formatArgsFileValue = { value ->
    value == null ? "" : value.toString()
}

def mochiParamArgs = {
    params.findAll { key, value ->
        value != null && value.toString() != "" && !(key == "sparse_method" && value.toString().toLowerCase() == "none")
    }.collect { key, value ->
        def text = formatArgsFileValue(value)
        "--${key}\n${text}"
    }.join("\n")
}

def mochiJobScript = { Map opts ->
    def runOutputDir = "${params.output_root}/${params.run_name}"
    def jobOutputDir = opts.jobOutputDir
    def cacheDir = opts.cacheDir ?: "${params.cache_root}/${params.run_name}"
    def mochiOutputDirectory = opts.mochiOutputDirectory ?: runOutputDir
    def foldExport = opts.fold != null ? "export MOCHI_FOLD='${formatArgValue(opts.fold)}'" : ""
    def stageExport = opts.stage != null ? "export SPARSE_STAGE_INDEX='${formatArgValue(opts.stage)}'" : ""
    def deviceExport = opts.device ? "export MOCHI_DEVICE='${formatArgValue(opts.device)}'" : ""
    def batchSizeExport = opts.batchSize != null ? "export BATCH_SIZE='${formatArgValue(opts.batchSize)}'" : ""
    def learnRateExport = opts.learnRate != null ? "export LEARN_RATE='${formatArgValue(opts.learnRate)}'" : ""
    def l1Export = opts.l1Factor != null ? "export L1_REGULARIZATION_FACTOR='${formatArgValue(opts.l1Factor)}'" : ""
    def l2Export = opts.l2Factor != null ? "export L2_REGULARIZATION_FACTOR='${formatArgValue(opts.l2Factor)}'" : ""
    """
    mkdir -p "${jobOutputDir}"
    cat > mochi_nextflow_args.txt <<'EOF'
${mochiParamArgs()}
EOF

    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${opts.runLabel}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${mochiOutputDirectory}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_PHASE="${opts.phase}"
    export MOCHI_SEED="${params.seed}"
    export K_FOLDS="${params.k_folds}"
    export NUM_EPOCHS="${params.num_epochs}"
    export NUM_EPOCHS_GRID="${params.num_epochs_grid}"
    export BATCH_SIZE="${params.batch_size}"
    export LEARN_RATE="${params.learn_rate}"
    export L1_REGULARIZATION_FACTOR="${params.l1_regularization_factor}"
    export L2_REGULARIZATION_FACTOR="${params.l2_regularization_factor}"
    export SPARSE_METHOD="${params.sparse_method}"
    export MOCHI_ARGS_FILE="\$PWD/mochi_nextflow_args.txt"
    ${foldExport}
    ${stageExport}
    ${deviceExport}
    ${batchSizeExport}
    ${learnRateExport}
    ${l1Export}
    ${l2Export}

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    ${opts.afterRun ?: ""}
    """
}

process RUN_GRID_SEARCH_CONDITION {
    label "mochi_grid_gpu"

    input:
    tuple val(condition), val(batchSize), val(learnRate), val(l1Factor), val(l2Factor)

    output:
    tuple val(condition), path("grid_condition_${condition}.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-grid-${condition}",
        phase: "grid_search",
        jobOutputDir: "${runOutputDir}/grid_search/condition_${condition}",
        mochiOutputDirectory: "${runOutputDir}/grid_search/condition_${condition}",
        cacheDir: "${params.cache_root}/${params.run_name}/grid_search/condition_${condition}",
        batchSize: batchSize,
        learnRate: learnRate,
        l1Factor: l1Factor,
        l2Factor: l2Factor,
        afterRun: "touch \"grid_condition_${condition}.done\""
    )
}

process MERGE_GRID_SEARCH_CONDITIONS {
    label "summary_lsf_cpu"

    input:
    path grid_condition_done

    output:
    path "grid_search.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-grid-merge",
        phase: "merge_grid_search",
        device: "cpu",
        jobOutputDir: "${runOutputDir}/grid_search_merge",
        cacheDir: "${params.cache_root}/${params.run_name}/grid_search_merge",
        afterRun: "touch \"grid_search.done\""
    )
}

process RUN_FOLD {
    label "mochi_fold_gpu"

    input:
    tuple val(fold), path(grid_ready)

    output:
    path "fold_${fold}.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-fold-${fold}",
        phase: "fit_best",
        fold: fold,
        jobOutputDir: "${runOutputDir}/fold_${fold}",
        afterRun: "touch \"fold_${fold}.done\""
    )
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
    mochiJobScript(
        runLabel: "${params.run_name}-merge",
        phase: "merge_folds",
        device: "cpu",
        jobOutputDir: "${runOutputDir}/merge",
        afterRun: "cp \"${runOutputDir}/merge/benchmark_manifest.env\" \"benchmark_manifest.env\""
    )
}

process RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION {
    label "mochi_grid_gpu"

    input:
    tuple val(stage), val(condition), val(batchSize), val(learnRate), val(l1Factor), val(l2Factor), path(prev_stage_ready)

    output:
    tuple val(stage), val(condition), path("sparse_stage_${stage}_grid_condition_${condition}.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-grid-${condition}",
        phase: "sparse_grid_search",
        stage: stage,
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}",
        mochiOutputDirectory: "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}",
        cacheDir: "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search/condition_${condition}",
        batchSize: batchSize,
        learnRate: learnRate,
        l1Factor: l1Factor,
        l2Factor: l2Factor,
        afterRun: "touch \"sparse_stage_${stage}_grid_condition_${condition}.done\""
    )
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
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-grid-merge",
        phase: "sparse_merge_grid_search",
        stage: stage,
        device: "cpu",
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search_merge",
        cacheDir: "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search_merge",
        afterRun: "touch \"sparse_stage_${stage}_grid.done\""
    )
}

process RUN_SPARSE_STAGE_FOLD {
    label "mochi_fold_gpu"

    input:
    tuple val(stage), val(fold), path(grid_ready)

    output:
    path "sparse_stage_${stage}_fold_${fold}.done"

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-fold-${fold}",
        phase: "sparse_fit_best",
        stage: stage,
        fold: fold,
        jobOutputDir: "${runOutputDir}/stage_${stage}/fold_${fold}",
        afterRun: "touch \"sparse_stage_${stage}_fold_${fold}.done\""
    )
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
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-merge",
        phase: "sparse_merge_folds",
        stage: stage,
        device: "cpu",
        jobOutputDir: "${runOutputDir}/stage_${stage}/merge",
        afterRun: "touch \"sparse_stage_${stage}_merge.done\"\n    cp \"${runOutputDir}/stage_${stage}/merge/benchmark_manifest.env\" \"benchmark_manifest.env\""
    )
}
