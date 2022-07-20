## Running with SLURM

```
sbatch --partition=[PARTITION] --nodes=[NUM_NODES] --gpus-per-task=[NUM_GPUS_PER_NODE] run.slurm
```


## Running locally

```
torchrun torchrun --nproc_per_node=[NUM_GPUS] main.py
```


## Results

Environment: p4d.24xlarge

Storage: FSx for Lustre

| Nodes x GPUs | Model Size | Param Count | torch.save | torchsnapshot |
| ------------ | -------------- | ----------- | ---------- | ------------- |
| 1 x 2 | 7.8GB | 1,947,477,189 | ~17.4s | ~3.6s | 
| 1 x 4 | 7.8GB | 1,947,477,189 | ~17.5s | ~2.1s |
| 1 x 8 | 7.8GB | 1,947,477,189 | ~17.6s | ~1.6s |

**TODO** debug FSDP failures when nodes > 1