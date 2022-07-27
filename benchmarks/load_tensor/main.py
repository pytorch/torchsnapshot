#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
import time

import torch
from torchsnapshot import Snapshot, StateDict
from torchsnapshot.rss_profiler import measure_rss_deltas

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


TENSOR_DIMS = (50000, 50000)
MEMORY_BUDGET_BYTES = 20 * 1024**2


def benchmark_torchsnapshot(path: str, gpu_tensor: torch.Tensor) -> None:
    app_state = {
        "state": StateDict(
            tensor=gpu_tensor,
        )
    }
    snapshot = Snapshot.take(path=path, app_state=app_state)

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


def benchmark_torch_save(path: str, gpu_tensor: torch.Tensor) -> None:
    tensor_path = os.path.join(path, "foo")
    torch.save(gpu_tensor, tensor_path)

    ts_begin = time.monotonic()
    rss_deltas = []
    logger.info("Loading the tensor with torch.load()...")
    with measure_rss_deltas(rss_deltas=rss_deltas):
        loaded = torch.load(tensor_path, map_location="cpu")
        gpu_tensor.copy_(loaded)

    logger.info(
        f"Took {time.monotonic() - ts_begin:.2f}. "
        f"Peak RSS delta: {max(rss_deltas) // 1024**2}MB"
    )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    gpu_tensor = torch.rand(*TENSOR_DIMS, device=device)
    with tempfile.TemporaryDirectory() as path:
        benchmark_torch_save(path=path, gpu_tensor=gpu_tensor)
    with tempfile.TemporaryDirectory() as path:
        benchmark_torchsnapshot(path=path, gpu_tensor=gpu_tensor)
