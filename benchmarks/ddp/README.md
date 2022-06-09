## Running with SLURM

```
sbatch --partition=[PARTITION] --nodes=[NUM_NODES] --gpus-per-task=[NUM_GPUS_PER_NODE] run.slurm
```

## Results

Environment: p4d.24xlarge

Storage: FSx for Lustre

| Nodes x GPUs | Avg Param Size | Param Count | torch.save | torchsnapshot |
| ------------ | -------------- | ----------- | ---------- | ------------- |
| 1 x 1 | 100MB | 180 | ~49s | ~45s |
| 1 x 8 | 100MB | 180 | ~49s | ~14s |
| 2 x 8 | 100MB | 180 | ~49s | ~7s |
| 4 x 8 | 100MB | 180 | ~49s | ~5s |
