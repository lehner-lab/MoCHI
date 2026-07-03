nextflow.enable.dsl=2
nextflow.preview.recursion=true

def inputParamKeys = new LinkedHashSet(params.keySet().collect { it.toString() })
def workflowOnlyParamKeys = [
    "nextflow_root",
    "repo_root",
    "mochi_repo",
    "mochi_venv",
    "output_root",
    "run_name",
    "workflow_mode",
    "parallel_folds",
    "queue",
    "gpu_queue",
    "gpu_cluster_options",
    "grid_gpu_cluster_options",
    "fold_gpu_cluster_options",
    "cpu_queue",
    "merge_memory",
    "merge_memory_max",
    "grid_memory",
    "grid_memory_max",
    "fold_memory",
    "fold_memory_max",
    "max_memory_retries"
] as Set

def hasParam = { key ->
    params.containsKey(key) && params[key] != null && params[key].toString() != ""
}

def paramOr = { key, fallback ->
    hasParam(key) ? params[key] : fallback
}

def requireParam = { key, context ->
    if (!hasParam(key)) {
        throw new IllegalArgumentException("${key} is required for ${context}")
    }
    return params[key]
}

params.nextflow_root = params.containsKey("nextflow_root") ? params["nextflow_root"] : workflow.projectDir.toString()
params.repo_root = params.containsKey("repo_root") ? params["repo_root"] : new File(params.nextflow_root.toString()).getCanonicalFile().getParent()
params.mochi_repo = paramOr("mochi_repo", "${params.repo_root}")
params.mochi_venv = paramOr("mochi_venv", "${params.repo_root}/.venv")
params.output_root = paramOr("output_root", paramOr("output_directory", workflow.launchDir.toString()))
params.run_name = paramOr("run_name", workflow.runName)
params.workflow_mode = paramOr("workflow_mode", "parallel_folds")
params.k_folds = paramOr("k_folds", 10)
params.forwarded_mochi_args = inputParamKeys.findAll { key ->
    !workflowOnlyParamKeys.contains(key) && hasParam(key)
}.collectEntries { key ->
    [(key): params[key]]
}

def splitGridParam = { value ->
    def text = value == null ? "" : value.toString().trim()
    return text ? text.split(/\s*,\s*/).collect { it.toString() }.findAll { it } : [""]
}

def buildGridConditions = {
    def batchSizes = splitGridParam(paramOr("batch_size", ""))
    def learnRates = splitGridParam(paramOr("learn_rate", ""))
    def l1Factors = splitGridParam(paramOr("l1_regularization_factor", ""))
    def l2Factors = splitGridParam(paramOr("l2_regularization_factor", ""))
    def conditions = []
    int conditionIndex = 1
    batchSizes.each { batchSize ->
        learnRates.each { learnRate ->
            l1Factors.each { l1Factor ->
                l2Factors.each { l2Factor ->
                    conditions << tuple(conditionIndex, batchSize, learnRate, l1Factor, l2Factor)
                    conditionIndex += 1
                }
            }
        }
    }
    return conditions
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

include {
    RUN_GRID_SEARCH_CONDITION
    MERGE_GRID_SEARCH_CONDITIONS
    RUN_FOLD
    MERGE_FOLDS
    RUN_SPARSE_STAGE
} from './modules/parallel_single_gpu'

workflow {
    def workflowMode = params.workflow_mode.toString()
    if (workflowMode == "legacy_full") {
        RUN_MOCHI()
    } else if (workflowMode != "parallel_folds") {
        throw new IllegalArgumentException("Unsupported workflow_mode: ${workflowMode}")
    } else {
        def foldCount = params.k_folds as int
        if (foldCount < 3) {
            throw new IllegalArgumentException("k_folds must be at least 3")
        }

        def gridConditions = buildGridConditions()
        if (hasParam("sparse_method")) {
            if (params.sparse_method != "sig_highestorder_step") {
                throw new IllegalArgumentException("Unsupported sparse_method for parallel_folds: ${params.sparse_method}")
            }
            def stageCount = (requireParam("max_interaction_order", "sparse parallel_folds workflow") as int) + 2
            RUN_SPARSE_STAGE
                .recurse(tuple(1, file(requireParam("model_design", "sparse parallel_folds workflow"))))
                .times(stageCount)
        } else {
            def gridConditionDone = RUN_GRID_SEARCH_CONDITION(Channel.fromList(gridConditions))
            def gridDone = MERGE_GRID_SEARCH_CONDITIONS(gridConditionDone.map { condition, done -> done }.collect())
            def foldInputs = gridDone.flatMap { gridReady ->
                (1..foldCount).collect { fold -> tuple(fold, gridReady) }
            }
            def foldDone = RUN_FOLD(foldInputs)
            MERGE_FOLDS(gridDone, foldDone.collect())
        }
    }
}

process RUN_MOCHI {
    label "mochi_gpu"
    tag "${params.run_name}"

    output:
    path "benchmark_manifest.env"

    script:
    def runLabel = params.run_name
    def outputDir = "${params.output_root}/${params.run_name}"
    def cliArgs = params.forwarded_mochi_args + [output_directory: outputDir]
    """
    cat > mochi_nextflow_args.txt <<'EOF'
${mochiArgsFileContent(cliArgs)}
EOF

    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export RUN_LABEL="${runLabel}"
    export OUTPUT_ROOT="${params.output_root}"
    export OUTPUT_DIR="${outputDir}"
    export MOCHI_ARGS_FILE="\$PWD/mochi_nextflow_args.txt"

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    cp "${outputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}
