nextflow.enable.dsl=2

params.nextflow_root = params.containsKey("nextflow_root") ? params["nextflow_root"] : workflow.projectDir.toString()
params.repo_root = params.containsKey("repo_root") ? params["repo_root"] : new File(params.nextflow_root.toString()).getCanonicalFile().getParent()
params.mochi_repo = params.containsKey("mochi_repo") ? params["mochi_repo"] : "${params.repo_root}"
params.model_design = params.containsKey("model_design") ? params["model_design"] : "/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/model_design_mochi_pool2_abs.txt"
params.expected_dataset = params.containsKey("expected_dataset") ? params["expected_dataset"] : "/nfs/users/nfs_e/eh19/work/data/mochi-dev/dataset_for_order2_model_benchmarks/FYN_BIBD_4sG2_mochi_pool2.txt"
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
params.gpu_time = params.containsKey("gpu_time") ? params["gpu_time"] : "12h"
params.gpu_cluster_options = params.containsKey("gpu_cluster_options") ? params["gpu_cluster_options"] : "-gpu 'num=1:mode=shared:j_exclusive=no:gpack=yes'"
params.grid_memory = params.containsKey("grid_memory") ? params["grid_memory"] : "8 GB"
params.grid_memory_max = params.containsKey("grid_memory_max") ? params["grid_memory_max"] : "50 GB"
params.fold_memory = params.containsKey("fold_memory") ? params["fold_memory"] : "8 GB"
params.fold_memory_max = params.containsKey("fold_memory_max") ? params["fold_memory_max"] : "32 GB"
params.max_memory_retries = params.containsKey("max_memory_retries") ? params["max_memory_retries"] : 3

include {
    RUN_GRID_SEARCH
    RUN_FOLD
    MERGE_FOLDS
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_1
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_2
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_3
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_4
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_5
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_6
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_7
    RUN_SPARSE_STAGE_GRID_SEARCH as RUN_SPARSE_STAGE_GRID_SEARCH_8
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
                def gridInput = prevStageReady.map { ready -> tuple(stageVal, ready) }
                def gridDone
                switch (stage) {
                    case 1: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_1(gridInput); break
                    case 2: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_2(gridInput); break
                    case 3: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_3(gridInput); break
                    case 4: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_4(gridInput); break
                    case 5: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_5(gridInput); break
                    case 6: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_6(gridInput); break
                    case 7: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_7(gridInput); break
                    case 8: gridDone = RUN_SPARSE_STAGE_GRID_SEARCH_8(gridInput); break
                }
                def foldInputs = gridDone.flatMap { sparseStage, gridReady ->
                    (1..foldCount).collect { fold -> tuple(sparseStage, fold, gridReady) }
                }
                def foldDone
                switch (stage) {
                    case 1: foldDone = RUN_SPARSE_STAGE_FOLD_1(foldInputs); break
                    case 2: foldDone = RUN_SPARSE_STAGE_FOLD_2(foldInputs); break
                    case 3: foldDone = RUN_SPARSE_STAGE_FOLD_3(foldInputs); break
                    case 4: foldDone = RUN_SPARSE_STAGE_FOLD_4(foldInputs); break
                    case 5: foldDone = RUN_SPARSE_STAGE_FOLD_5(foldInputs); break
                    case 6: foldDone = RUN_SPARSE_STAGE_FOLD_6(foldInputs); break
                    case 7: foldDone = RUN_SPARSE_STAGE_FOLD_7(foldInputs); break
                    case 8: foldDone = RUN_SPARSE_STAGE_FOLD_8(foldInputs); break
                }
                def mergedStage
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
                prevStageReady = mergedStage.map { sparseStage, mergeReady, manifest -> mergeReady }
                finalManifest = mergedStage.map { sparseStage, mergeReady, manifest -> manifest }
            }

            finalManifest
        } else {
            def foldCount = params.k_folds as int
            def gridDone = RUN_GRID_SEARCH()
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
    export EXPECTED_DATASET="${params.expected_dataset}"
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
