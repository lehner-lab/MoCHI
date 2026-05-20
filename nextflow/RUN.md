# Nextflow Run

Bootstrap once:

```bash
cd MoCHI
bash bootstrap_mochi_uv.sh
```

Submit the Nextflow master to the oversubscribed queue:

```bash
cd MoCHI
RUN_NAME=mochi-parallel-order2-test \
MODEL_DESIGN=/path/to/model_design.tsv \
SPARSE_METHOD=sig_highestorder_step \
MASTER_MEMORY_GB=24 \
bash nextflow/scripts/submit_mochi_master_lsf.sh
```

Resume:

```bash
cd MoCHI
RESUME=1 \
RUN_NAME=mochi-parallel-order2-test \
MODEL_DESIGN=/path/to/model_design.tsv \
SPARSE_METHOD=sig_highestorder_step \
bash nextflow/scripts/submit_mochi_master_lsf.sh
```
