# Nextflow Run

Bootstrap once:

```bash
cd MoCHI
bash bootstrap_mochi_uv.sh
```

Submit the Nextflow master to the oversubscribed queue:

```bash
cd MoCHI
MASTER_MEMORY_GB=24 \
bash nextflow/scripts/submit_mochi_master_lsf.sh \
    --run_name mochi-parallel-order2-test \
    --model_design /path/to/model_design.tsv \
    --sparse_method sig_highestorder_step
```

Resume:

```bash
cd MoCHI
RESUME=1 \
bash nextflow/scripts/submit_mochi_master_lsf.sh \
    --run_name mochi-parallel-order2-test \
    --model_design /path/to/model_design.tsv \
    --sparse_method sig_highestorder_step
```
