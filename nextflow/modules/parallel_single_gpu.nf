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
    def deviceExport = opts.device ? "export MOCHI_DEVICE='${formatArgValue(opts.device)}'" : ""
    def cliArgs = params.forwarded_mochi_args + (opts.args ?: [:])
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
    tag "stage-${stage}-grid-${condition}"

    input:
    tuple val(stage), val(condition), val(batchSize), val(learnRate), val(l1Factor), val(l2Factor), path(prev_stage_ready)

    output:
    tuple val(stage), val(condition), path("sparse_stage_${stage}_grid_condition_${condition}.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-grid-${condition}",
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search/condition_${condition}",
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
    tag "stage-${stage}-grid-merge"

    input:
    tuple val(stage), path(grid_condition_done)

    output:
    tuple val(stage), path("sparse_stage_${stage}_grid.done")

    script:
    def runOutputDir = "${params.output_root}/${params.run_name}"
    mochiJobScript(
        runLabel: "${params.run_name}-stage-${stage}-grid-merge",
        device: "cpu",
        jobOutputDir: "${runOutputDir}/stage_${stage}/grid_search_merge",
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
    tag "stage-${stage}-fold-${fold}"

    input:
    tuple val(stage), val(fold), path(grid_ready)

    output:
    tuple val(stage), val(fold), path("sparse_stage_${stage}_fold_${fold}.done")

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

process MERGE_SPARSE_STAGE_FOLDS {
    label "summary_lsf_cpu"
    tag "stage-${stage}-merge"

    input:
    tuple val(stage), path(fold_done)

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

workflow RUN_SPARSE_STAGE {
    take:
    stage_state

    main:
    def gridConditions = []
    int conditionIndex = 1
    def splitGridParam = { value ->
        def text = value == null ? "" : value.toString().trim()
        return text ? text.split(/\s*,\s*/).collect { it.toString() }.findAll { it } : [""]
    }
    def paramOr = { key, fallback ->
        params.containsKey(key) && params[key] != null && params[key].toString() != "" ? params[key] : fallback
    }
    splitGridParam(paramOr("batch_size", "")).each { batchSize ->
        splitGridParam(paramOr("learn_rate", "")).each { learnRate ->
            splitGridParam(paramOr("l1_regularization_factor", "")).each { l1Factor ->
                splitGridParam(paramOr("l2_regularization_factor", "")).each { l2Factor ->
                    gridConditions << tuple(conditionIndex, batchSize, learnRate, l1Factor, l2Factor)
                    conditionIndex += 1
                }
            }
        }
    }
    def foldCount = params.k_folds as int
    def sparseGridInputs = stage_state.flatMap { stage, previousReady ->
        gridConditions.collect { condition ->
            tuple(stage, condition[0], condition[1], condition[2], condition[3], condition[4], previousReady)
        }
    }
    def sparseGridDone = RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION(sparseGridInputs)
    def sparseMergedGrid = MERGE_SPARSE_STAGE_GRID_SEARCH(
        sparseGridDone.map { stage, condition, done -> tuple(stage, done) }.groupTuple(size: gridConditions.size())
    )
    def sparseFoldInputs = sparseMergedGrid.flatMap { stage, gridReady ->
        (1..foldCount).collect { fold -> tuple(stage, fold, gridReady) }
    }
    def sparseFoldDone = RUN_SPARSE_STAGE_FOLD(sparseFoldInputs)
    def sparseMergedStage = MERGE_SPARSE_STAGE_FOLDS(
        sparseFoldDone.map { stage, fold, done -> tuple(stage, done) }.groupTuple(size: foldCount)
    )

    emit:
    sparseMergedStage.map { stage, done, manifest -> tuple((stage as int) + 1, manifest) }
}
