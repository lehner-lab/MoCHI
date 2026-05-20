def formatArgValue = { value ->
    value == null ? "" : value.toString().replace("'", "'\"'\"'")
}

def formatArgsFileValue = { value ->
    value == null ? "" : value.toString()
}

def mochiArgsFileContent = { Map args ->
    args.findAll { key, value ->
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
    def deviceExport = opts.device ? "export MOCHI_DEVICE='${formatArgValue(opts.device)}'" : ""
    def cliArgs = [:] + params + (opts.args ?: [:])
    """
    mkdir -p "${jobOutputDir}"
    cat > mochi_nextflow_args.txt <<'EOF'
${mochiArgsFileContent(cliArgs)}
EOF

    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export RUN_LABEL="${opts.runLabel}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${jobOutputDir}"
    export CACHE_DIR="${cacheDir}"
    export MOCHI_PARALLEL_MODE="1"
    export MOCHI_ARGS_FILE="\$PWD/mochi_nextflow_args.txt"
    ${deviceExport}

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
        jobOutputDir: "${runOutputDir}/grid_search/condition_${condition}",
        cacheDir: "${params.cache_root}/${params.run_name}/grid_search/condition_${condition}",
        args: [
            output_directory: "${runOutputDir}/grid_search/condition_${condition}",
            phase: "grid_search",
            batch_size: batchSize,
            learn_rate: learnRate,
            l1_regularization_factor: l1Factor,
            l2_regularization_factor: l2Factor
        ],
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
        device: "cpu",
        jobOutputDir: "${runOutputDir}/grid_search_merge",
        cacheDir: "${params.cache_root}/${params.run_name}/grid_search_merge",
        args: [
            output_directory: runOutputDir,
            phase: "merge_grid_search"
        ],
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
        jobOutputDir: "${runOutputDir}/fold_${fold}",
        args: [
            output_directory: runOutputDir,
            phase: "fit_best",
            fold: fold
        ],
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
        device: "cpu",
        jobOutputDir: "${runOutputDir}/merge",
        args: [
            output_directory: runOutputDir,
            phase: "merge_folds"
        ],
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
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}",
        cacheDir: "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search/condition_${condition}",
        args: [
            output_directory: "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}",
            phase: "sparse_grid_search",
            stage_index: stage,
            batch_size: batchSize,
            learn_rate: learnRate,
            l1_regularization_factor: l1Factor,
            l2_regularization_factor: l2Factor
        ],
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
        device: "cpu",
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search_merge",
        cacheDir: "${params.cache_root}/${params.run_name}/stage_${stage}/grid_search_merge",
        args: [
            output_directory: runOutputDir,
            phase: "sparse_merge_grid_search",
            stage_index: stage
        ],
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
        jobOutputDir: "${runOutputDir}/stage_${stage}/fold_${fold}",
        args: [
            output_directory: runOutputDir,
            phase: "sparse_fit_best",
            stage_index: stage,
            fold: fold
        ],
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
        device: "cpu",
        jobOutputDir: "${runOutputDir}/stage_${stage}/merge",
        args: [
            output_directory: runOutputDir,
            phase: "sparse_merge_folds",
            stage_index: stage
        ],
        afterRun: "touch \"sparse_stage_${stage}_merge.done\"\n    cp \"${runOutputDir}/stage_${stage}/merge/benchmark_manifest.env\" \"benchmark_manifest.env\""
    )
}
