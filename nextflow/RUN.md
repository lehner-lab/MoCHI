# Nextflow Run

Bootstrap the Python environment once:

```bash
cd MoCHI
bash bootstrap_mochi_uv.sh
```

Install [Nextflow](https://www.nextflow.io/) and Java, then run locally:

```bash
cd MoCHI
bash nextflow/scripts/run_mochi_nextflow.sh \
    --run_name mochi-parallel-order2-test \
    --model_design /path/to/model_design.tsv \
    --sparse_method sig_highestorder_step
```

The default `local` profile executes tasks on the current host. GPU tasks require a CUDA-capable local environment. Results and Nextflow work files default to `./results/<run_name>`.

## LSF

Use the optional LSF profile when compute nodes can access the repository, `.venv`, output directory, and work directory at the same paths:

```bash
cd MoCHI
QUEUE=gpu \
CPU_QUEUE=normal \
MASTER_QUEUE=normal \
MASTER_MEMORY_GB=24 \
bash nextflow/scripts/submit_mochi_master_lsf.sh \
    --run_name mochi-parallel-order2-test \
    --model_design /path/to/model_design.tsv \
    --sparse_method sig_highestorder_step
```

Set `MOCHI_GPU_CLUSTER_OPTIONS`, `MOCHI_GRID_GPU_CLUSTER_OPTIONS`, and `MOCHI_FOLD_GPU_CLUSTER_OPTIONS` for your site's LSF GPU request syntax. If Nextflow is supplied through Environment Modules, set `NEXTFLOW_MODULE`; otherwise it must be on `PATH` or specified with `NEXTFLOW_BIN`.

To resume either profile:

```bash
cd MoCHI
RESUME=1 \
bash nextflow/scripts/run_mochi_nextflow.sh \
    --run_name mochi-parallel-order2-test \
    --model_design /path/to/model_design.tsv \
    --sparse_method sig_highestorder_step
```

For LSF resume, use the same command with `submit_mochi_master_lsf.sh` and scheduler settings above.
