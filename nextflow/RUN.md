# Nextflow Run

Bootstrap once:

```bash
cd MoCHI
bash nextflow/scripts/bootstrap_mochi_uv.sh
```

Run:

```bash
cd MoCHI
RUN_NAME=mochi-parallel-order2-test \
MODEL_DESIGN=/path/to/model_design.tsv \
EXPECTED_DATASET=/path/to/expected_dataset.tsv \
bash nextflow/scripts/submit_mochi_benchmark_nextflow.sh
```

Resume:

```bash
cd MoCHI
RESUME=1 RUN_NAME=mochi-parallel-order2-test bash nextflow/scripts/submit_mochi_benchmark_nextflow.sh
```
