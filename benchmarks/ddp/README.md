## Running with SLURM

```
sbatch --partition=[PARTITION] --nodes=[NUM_NODES] --gpus-per-task=[NUM_GPUS_PER_NODE] run.slurm
```

## Benchmark

PyTorch version: 1.13.0.dev20220915+cu113

Benchmark environment: p4d.24xlarge

Model size: 20GB

| Storage Type | Nodes x GPUs | torch.save | torchsnapshot |
| ------------ | ------------ | ---------- | ------------- |
| Local FS | 1 x 1 | ~32s | ~13.91s |
| Local FS | 1 x 8 | ~32s | ~3.38s |
| Local FS | 2 x 8 | ~32s | ~2.02s |
| Local FS | 4 x 8 | ~32s | ~1.29s |
| FSx for Lustre | 1 x 1 |  ~38s | ~14.52s |
| FSx for Lustre | 1 x 8 |  ~38s | ~7.61s |
| FSx for Lustre | 2 x 8 |  ~38s | ~4.61s |
| FSx for Lustre | 4 x 8 |  ~38s | ~2.68s |
