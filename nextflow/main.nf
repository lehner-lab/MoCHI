nextflow.enable.dsl=2

params.nextflow_root = params.containsKey("nextflow_root") ? params["nextflow_root"] : workflow.projectDir.toString()
params.repo_root = params.containsKey("repo_root") ? params["repo_root"] : new File(params.nextflow_root.toString()).getCanonicalFile().getParent()
params.mochi_repo = params.containsKey("mochi_repo") ? params["mochi_repo"] : "${params.repo_root}"
params.model_design = params.containsKey("model_design") ? params["model_design"] : "/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/model_design_mochi_pool2_abs.txt"
params.output_root = params.containsKey("output_root") ? params["output_root"] : "/lustre/scratch124/humgen/teams_v2/hgi/eh19/work-data/mochi-dev"
params.cache_root = params.containsKey("cache_root") ? params["cache_root"] : "${params.output_root}/cache"
params.run_name = params.containsKey("run_name") ? params["run_name"] : "mochi-benchmark"
params.project_name = params.containsKey("project_name") ? params["project_name"] : "mochi_project"
params.max_interaction_order = params.containsKey("max_interaction_order") ? params["max_interaction_order"] : 2
params.mochi_venv = params.containsKey("mochi_venv") ? params["mochi_venv"] : "${params.repo_root}/.venv"
params.seed = params.containsKey("seed") ? params["seed"] : 1
params.k_folds = params.containsKey("k_folds") ? params["k_folds"] : 10
params.parallel_folds = params.containsKey("parallel_folds") ? params["parallel_folds"] : params.k_folds
params.num_epochs = params.containsKey("num_epochs") ? params["num_epochs"] : 1000
params.num_epochs_grid = params.containsKey("num_epochs_grid") ? params["num_epochs_grid"] : 100
params.batch_size = params.containsKey("batch_size") ? params["batch_size"] : ""
params.learn_rate = params.containsKey("learn_rate") ? params["learn_rate"] : ""
params.l1_regularization_factor = params.containsKey("l1_regularization_factor") ? params["l1_regularization_factor"] : ""
params.l2_regularization_factor = params.containsKey("l2_regularization_factor") ? params["l2_regularization_factor"] : ""
params.sparse_method = params.containsKey("sparse_method") ? params["sparse_method"] : ""
params.workflow_mode = params.containsKey("workflow_mode") ? params["workflow_mode"] : "parallel_folds"
params.gpu_queue = params.containsKey("gpu_queue") ? params["gpu_queue"] : "gpu-normal"
params.gpu_cluster_options = params.containsKey("gpu_cluster_options") ? params["gpu_cluster_options"] : "-gpu 'num=1:mode=shared:j_exclusive=no:gpack=yes' -R \"select[hname!='farm22-gpu0203']\""
params.grid_gpu_cluster_options = params.containsKey("grid_gpu_cluster_options") ? params["grid_gpu_cluster_options"] : params.gpu_cluster_options
params.fold_gpu_cluster_options = params.containsKey("fold_gpu_cluster_options") ? params["fold_gpu_cluster_options"] : "-gpu 'num=1:mode=exclusive_process:j_exclusive=no:gpack=yes' -R \"select[hname!='farm22-gpu0203']\""
params.cpu_queue = params.containsKey("cpu_queue") ? params["cpu_queue"] : "normal"
params.merge_memory = params.containsKey("merge_memory") ? params["merge_memory"] : "24 GB"
params.merge_memory_max = params.containsKey("merge_memory_max") ? params["merge_memory_max"] : "50 GB"
params.grid_memory = params.containsKey("grid_memory") ? params["grid_memory"] : "24 GB"
params.grid_memory_max = params.containsKey("grid_memory_max") ? params["grid_memory_max"] : "50 GB"
params.fold_memory = params.containsKey("fold_memory") ? params["fold_memory"] : "24 GB"
params.fold_memory_max = params.containsKey("fold_memory_max") ? params["fold_memory_max"] : "50 GB"
params.max_memory_retries = params.containsKey("max_memory_retries") ? params["max_memory_retries"] : 3

def splitGridParam = { value, defaults ->
    def text = value == null ? "" : value.toString().trim()
    def rawValues = text ? text.split(/\s*,\s*/) : defaults
    return rawValues.collect { it.toString() }.findAll { it }
}

def buildGridConditions = {
    def batchSizes = splitGridParam(params.batch_size, ["512", "1024", "2048"])
    def learnRates = splitGridParam(params.learn_rate, ["0.05"])
    def l1Factors = splitGridParam(params.l1_regularization_factor, ["0"])
    def l2Factors = splitGridParam(params.l2_regularization_factor, ["0.000001"])
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

def sparseRunRoot = new File("${params.output_root}/${params.run_name}")
def sparseShortcutDir = new File(sparseRunRoot, ".nextflow_shortcuts")

def ensureSparseShortcutFile = { name ->
    sparseShortcutDir.mkdirs()
    def shortcutFile = new File(sparseShortcutDir, name)
    if (!shortcutFile.exists()) {
        shortcutFile.text = "shortcut\n"
    }
    return file(shortcutFile.toString())
}

def sparseTaskDirectory = { stage ->
    new File(sparseRunRoot, "${params.project_name}/task_${stage}")
}

def sparseFoldSavedModelsExist = { stage, fold ->
    new File(sparseTaskDirectory(stage), "fold_${fold}/saved_models").exists()
}

def sparseStageSavedModelsExist = { stage ->
    new File(sparseTaskDirectory(stage), "saved_models").exists()
}

def sparseMergeManifestFile = { stage ->
    new File(sparseRunRoot, "stage_${stage}/merge/benchmark_manifest.env")
}

def sparseAllFoldArtifactsExist = { stage, foldCount ->
    (1..foldCount).every { fold -> sparseFoldSavedModelsExist(stage, fold) }
}

def sparseMergeShortcutReady = { stage, foldCount ->
    sparseStageSavedModelsExist(stage) &&
        sparseAllFoldArtifactsExist(stage, foldCount) &&
        sparseMergeManifestFile(stage).exists()
}

def sparseGridShortcutReady = { stage ->
    sparseStageSavedModelsExist(stage)
}

include {
    RUN_GRID_SEARCH_CONDITION
    MERGE_GRID_SEARCH_CONDITIONS
    RUN_FOLD
    MERGE_FOLDS
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_1
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_2
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_3
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_4
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_5
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_6
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_7
    RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION as RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_8
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_1
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_2
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_3
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_4
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_5
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_6
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_7
    MERGE_SPARSE_STAGE_GRID_SEARCH as MERGE_SPARSE_STAGE_GRID_SEARCH_8
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_1
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_2
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_3
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_4
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_5
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_6
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_7
    RUN_SPARSE_STAGE_FOLD as RUN_SPARSE_STAGE_FOLD_8
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_1
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_2
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_3
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_4
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_5
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_6
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_7
    RUN_SPARSE_STAGE_MERGE as RUN_SPARSE_STAGE_MERGE_8
} from './modules/parallel_single_gpu'

workflow {
    if ((params.k_folds as int) < 3) {
        throw new IllegalArgumentException("k_folds must be at least 3")
    }
    if (params.workflow_mode == "legacy_full") {
        RUN_MOCHI()
    } else if (params.workflow_mode == "parallel_folds") {
        def gridConditions = buildGridConditions()
        if (params.sparse_method) {
            if (params.sparse_method != "sig_highestorder_step") {
                throw new IllegalArgumentException("Unsupported sparse_method for parallel_folds: ${params.sparse_method}")
            }
            def foldCount = params.k_folds as int
            def stageCount = (params.max_interaction_order as int) + 2
            if (stageCount > 8) {
                throw new IllegalArgumentException("parallel_folds sparse workflow currently supports up to 8 stages; got ${stageCount}")
            }
            def prevStageReady = Channel.value(file(params.model_design))
            def finalManifest = null

            for (int stage = 1; stage <= stageCount; stage++) {
                def stageVal = stage
                def mergedStage

                if (sparseMergeShortcutReady(stageVal, foldCount)) {
                    mergedStage = Channel.value(tuple(
                        stageVal,
                        ensureSparseShortcutFile("sparse_stage_${stageVal}_merge.done"),
                        file(sparseMergeManifestFile(stageVal).toString())
                    ))
                } else {
                    def gridInput = prevStageReady.flatMap { ready ->
                        gridConditions.collect { condition ->
                            tuple(stageVal, condition[0], condition[1], condition[2], condition[3], condition[4], ready)
                        }
                    }

                    def gridDone
                    if (sparseGridShortcutReady(stageVal)) {
                        gridDone = Channel.value(tuple(
                            stageVal,
                            ensureSparseShortcutFile("sparse_stage_${stageVal}_grid.done")
                        ))
                    } else {
                        switch (stage) {
                            case 1: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_1(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_1(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 2: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_2(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_2(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 3: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_3(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_3(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 4: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_4(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_4(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 5: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_5(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_5(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 6: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_6(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_6(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 7: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_7(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_7(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                            case 8: gridDone = MERGE_SPARSE_STAGE_GRID_SEARCH_8(Channel.value(stageVal), RUN_SPARSE_STAGE_GRID_SEARCH_CONDITION_8(gridInput).map { sparseStage, condition, done -> done }.collect()); break
                        }
                    }

                    def shortcutFoldPaths = []
                    def missingFolds = []
                    (1..foldCount).each { fold ->
                        if (sparseFoldSavedModelsExist(stageVal, fold)) {
                            shortcutFoldPaths << ensureSparseShortcutFile("sparse_stage_${stageVal}_fold_${fold}.done")
                        } else {
                            missingFolds << fold
                        }
                    }

                    def shortcutFoldDone = Channel.fromList(shortcutFoldPaths)
                    def submittedFoldDone = Channel.empty()
                    if (missingFolds) {
                        def foldInputs = gridDone.flatMap { sparseStage, gridReady ->
                            missingFolds.collect { fold -> tuple(sparseStage, fold, gridReady) }
                        }
                        switch (stage) {
                            case 1: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_1(foldInputs); break
                            case 2: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_2(foldInputs); break
                            case 3: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_3(foldInputs); break
                            case 4: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_4(foldInputs); break
                            case 5: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_5(foldInputs); break
                            case 6: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_6(foldInputs); break
                            case 7: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_7(foldInputs); break
                            case 8: submittedFoldDone = RUN_SPARSE_STAGE_FOLD_8(foldInputs); break
                        }
                    }

                    def foldDone = missingFolds ? shortcutFoldDone.mix(submittedFoldDone) : shortcutFoldDone

                    switch (stage) {
                        case 1: mergedStage = RUN_SPARSE_STAGE_MERGE_1(gridDone, foldDone.collect()); break
                        case 2: mergedStage = RUN_SPARSE_STAGE_MERGE_2(gridDone, foldDone.collect()); break
                        case 3: mergedStage = RUN_SPARSE_STAGE_MERGE_3(gridDone, foldDone.collect()); break
                        case 4: mergedStage = RUN_SPARSE_STAGE_MERGE_4(gridDone, foldDone.collect()); break
                        case 5: mergedStage = RUN_SPARSE_STAGE_MERGE_5(gridDone, foldDone.collect()); break
                        case 6: mergedStage = RUN_SPARSE_STAGE_MERGE_6(gridDone, foldDone.collect()); break
                        case 7: mergedStage = RUN_SPARSE_STAGE_MERGE_7(gridDone, foldDone.collect()); break
                        case 8: mergedStage = RUN_SPARSE_STAGE_MERGE_8(gridDone, foldDone.collect()); break
                    }
                }
                prevStageReady = mergedStage.map { sparseStage, mergeReady, manifest -> mergeReady }
                finalManifest = mergedStage.map { sparseStage, mergeReady, manifest -> manifest }
            }

            finalManifest
        } else {
            def foldCount = params.k_folds as int
            def gridConditionDone = RUN_GRID_SEARCH_CONDITION(Channel.fromList(gridConditions))
            def gridDone = MERGE_GRID_SEARCH_CONDITIONS(gridConditionDone.map { condition, done -> done }.collect())
            def foldInputs = gridDone.flatMap { gridReady ->
                (1..foldCount).collect { fold -> tuple(fold, gridReady) }
            }
            def foldDone = RUN_FOLD(foldInputs)
            MERGE_FOLDS(gridDone, foldDone.collect())
        }
    } else {
        throw new IllegalArgumentException("Unsupported workflow_mode: ${params.workflow_mode}")
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
    def cacheDir = "${params.cache_root}/${params.run_name}"
    """
    export REPO_ROOT="${params.repo_root}"
    export MOCHI_REPO="${params.mochi_repo}"
    export MOCHI_VENV="${params.mochi_venv}"
    export MODEL_DESIGN="${params.model_design}"
    export RUN_LABEL="${runLabel}"
    export OUTPUT_ROOT="${params.output_root}/${params.run_name}"
    export OUTPUT_DIR="${outputDir}"
    export MOCHI_OUTPUT_DIRECTORY="${outputDir}"
    export CACHE_DIR="${cacheDir}"
    export PROJECT_NAME="${params.project_name}"
    export MAX_INTERACTION_ORDER="${params.max_interaction_order}"
    export MOCHI_PHASE="full"
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
    cp "${outputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}
