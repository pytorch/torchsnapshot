import argparse
import json
import time
from functools import partial
from uuid import uuid4

import torch
import torchsnapshot
from torch import distributed as dist, nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    dist.init_process_group("nccl")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    rank = dist.get_rank()
    # assuming each machine has the same number of GPUs
    local_rank = rank % torch.cuda.device_count()

    benchmark = {
        "world_size": dist.get_world_size(),
        "gpus": torch.cuda.device_count(),
        "nodes": dist.get_world_size() // torch.cuda.device_count(),
    }

    # similar to GPT-2
    model = nn.Transformer(
        d_model=864,
        num_encoder_layers=1,
        num_decoder_layers=20,
        nhead=12,
        dim_feedforward=50257,
    ).to(local_rank)

    if rank == 0:
        sz = sum(t.nelement() * t.element_size() for t in model.parameters())
        count = sum(t.nelement() for t in model.parameters())
        sz_gb = sz / 1_000_000_000.0
        print(f"Model size: {sz_gb} GB")
        print(f"Model parameters: {count}")
        benchmark["size_gb"] = sz_gb
        benchmark["params"] = count

    model = FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                nn.TransformerDecoderLayer,
                nn.TransformerEncoderLayer,
            },
        ),
    )
    torch.cuda.set_device(local_rank)

    # TODO add FSDP optimizer benchmark
    app_state = {"model": model}

    dist.barrier()

    # torch.save benchmark
    t0 = time.monotonic()
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        # needs to be called on all ranks, but only aggregates to rank 0
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, f"{args.work_dir}/{uuid4()}.pt")
    t1 = time.monotonic()

    if rank == 0:
        duration = t1 - t0
        print(f"Took {duration} seconds with torch.save")
        benchmark["torchsave_sec"] = duration

    dist.barrier()

    # torchsnapshot benchmark
    t0 = time.monotonic()
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        torchsnapshot.Snapshot.take(
            path=f"{args.work_dir}/{uuid4()}",
            app_state=app_state,
        )
        t1 = time.monotonic()

        if rank == 0:
            duration = t1 - t0
            benchmark["torchsnapshot_sec"] = duration
            print(
                f"Took {duration} seconds with torchsnapshot using FSDP local state dict"
            )
            if args.json_out:
                json.dump(benchmark, open(f"{args.json_out}/{uuid4()}.json", "w"))


if __name__ == "__main__":
    main()
