#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import tempfile
import time
import uuid

import fsspec
import torch
from torchsnapshot import Snapshot, StateDict
from torchsnapshot.rss_profiler import measure_rss_deltas

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


TENSOR_DIMS = (50000, 50000)
MEMORY_BUDGET_BYTES = 100 * 1024**2


def benchmark_torchsnapshot(work_dir: str, gpu_tensor: torch.Tensor) -> None:
    app_state = {
        "state": StateDict(
            tensor=gpu_tensor,
        )
    }
    snapshot = Snapshot.take(path=f"{work_dir}/{uuid.uuid4()}", app_state=app_state)

    ts_begin = time.monotonic()
    rss_deltas = []
    logger.info("Loading the tensor with torchsnapshot (without memory budget)...")
    with measure_rss_deltas(rss_deltas=rss_deltas):
        snapshot.read_object(path="0/state/tensor", obj_out=gpu_tensor)
    logger.info(
        f"Took {time.monotonic() - ts_begin:.2f} seconds. "
        f"Peak RSS delta: {max(rss_deltas) // 1024**2}MB"
    )

    ts_begin = time.monotonic()
    rss_deltas = []
    logger.info(
        f"Loading the tensor with torchsnapshot "
        f"(with a {MEMORY_BUDGET_BYTES // 1024**2:.2f}MB memory budget)..."
    )
    with measure_rss_deltas(rss_deltas=rss_deltas):
        snapshot.read_object(
            path="0/state/tensor",
            obj_out=gpu_tensor,
            memory_budget_bytes=MEMORY_BUDGET_BYTES,
        )
    logger.info(
        f"Took {time.monotonic() - ts_begin:.2f}. "
        f"Peak RSS delta: {max(rss_deltas) // 1024**2}MB"
    )


def benchmark_torch_save_fsspec(work_dir: str, gpu_tensor: torch.Tensor) -> None:
    path = os.path.join(work_dir, str(uuid.uuid4()))
    with fsspec.open(path, "wb") as f:
        torch.save(gpu_tensor, f)

    ts_begin = time.monotonic()
    rss_deltas = []
    logger.info("Loading the tensor with torch.load()...")
    with measure_rss_deltas(rss_deltas=rss_deltas):
        with fsspec.open(path, "rb") as f:
            loaded = torch.load(f, map_location="cpu")
        gpu_tensor.copy_(loaded)

    logger.info(
        f"Took {time.monotonic() - ts_begin:.2f}. "
        f"Peak RSS delta: {max(rss_deltas) // 1024**2}MB"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as path:
        parser = argparse.ArgumentParser()
        parser.add_argument("--work-dir", default=str(path))
        args: argparse.Namespace = parser.parse_args()

        device = torch.device("cuda:0")
        gpu_tensor = torch.rand(*TENSOR_DIMS, device=device)
        benchmark_torch_save_fsspec(work_dir=args.work_dir, gpu_tensor=gpu_tensor)
        benchmark_torchsnapshot(work_dir=args.work_dir, gpu_tensor=gpu_tensor)
