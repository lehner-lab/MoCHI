nextflow.enable.dsl=2

params.nextflow_root = params.containsKey("nextflow_root") ? params["nextflow_root"] : workflow.projectDir.toString()
params.repo_root = params.containsKey("repo_root") ? params["repo_root"] : new File(params.nextflow_root.toString()).getCanonicalFile().getParent()
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
params.workflow_mode = params.containsKey("workflow_mode") ? params["workflow_mode"] : "parallel_folds"
params.gpu_queue = params.containsKey("gpu_queue") ? params["gpu_queue"] : "gpu-normal"
params.gpu_time = params.containsKey("gpu_time") ? params["gpu_time"] : "12h"
params.gpu_cluster_options = params.containsKey("gpu_cluster_options") ? params["gpu_cluster_options"] : "-gpu 'num=1:mode=shared:j_exclusive=no:gpack=yes'"
params.grid_memory = params.containsKey("grid_memory") ? params["grid_memory"] : "8 GB"
params.grid_memory_max = params.containsKey("grid_memory_max") ? params["grid_memory_max"] : "50 GB"
params.fold_memory = params.containsKey("fold_memory") ? params["fold_memory"] : "8 GB"
params.fold_memory_max = params.containsKey("fold_memory_max") ? params["fold_memory_max"] : "32 GB"
params.max_memory_retries = params.containsKey("max_memory_retries") ? params["max_memory_retries"] : 3

include { RUN_GRID_SEARCH; RUN_FOLD; MERGE_FOLDS } from './modules/parallel_single_gpu'

workflow {
    if ((params.k_folds as int) < 3) {
        throw new IllegalArgumentException("k_folds must be at least 3")
    }
    if (params.workflow_mode == "legacy_full") {
        RUN_MOCHI()
    } else if (params.workflow_mode == "parallel_folds") {
        def foldCount = params.k_folds as int
        def gridDone = RUN_GRID_SEARCH()
        def foldInputs = gridDone.flatMap { gridReady ->
            (1..foldCount).collect { fold -> tuple(fold, gridReady) }
        }
        def foldDone = RUN_FOLD(foldInputs)
        MERGE_FOLDS(gridDone, foldDone.collect())
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
    export MOCHI_REPO="${params.repo_root}"
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

    "${params.nextflow_root}/scripts/run_mochi_lsf_gpu.sh"
    cp "${outputDir}/benchmark_manifest.env" "benchmark_manifest.env"
    """
}
