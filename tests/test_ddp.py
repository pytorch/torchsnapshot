#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Generator, List

import pytest

import torch
import torch.distributed as dist
import torchsnapshot
from _pytest.fixtures import SubRequest  # @manual
from torch.nn.parallel import DistributedDataParallel
from torchsnapshot.test_utils import check_state_dict_eq, run_with_pet


WORLD_SIZE = 4


@pytest.fixture
def layer_shapes() -> List[List[int]]:
    return [[128, 64], [64, 32], [32, 16]]


@pytest.fixture(params=[True, False])
def enable_chunking(
    layer_shapes: List[List[int]], request: SubRequest
) -> Generator[None, None, None]:
    if not request.param:
        yield
        return
    min_layer_size_bytes = min(reduce(mul, ls) for ls in layer_shapes) * 4

    os.environ["TORCHSNAPSHOT_MAX_CHUNK_SIZE_BYTES_OVERRIDE"] = str(
        min_layer_size_bytes // WORLD_SIZE
    )
    yield
    del os.environ["TORCHSNAPSHOT_MAX_CHUNK_SIZE_BYTES_OVERRIDE"]


@pytest.fixture(params=[True, False])
def enable_batcher(
    layer_shapes: List[List[int]], request: SubRequest
) -> Generator[None, None, None]:
    if not request.param:
        yield
        return
    total_layer_size_bytes = sum(reduce(mul, ls) for ls in layer_shapes) * 4
    os.environ["TORCHSNAPSHOT_SLAB_SIZE_THRESHOLD_BYTES_OVERRIDE"] = str(
        total_layer_size_bytes * 2
    )
    yield
    del os.environ["TORCHSNAPSHOT_SLAB_SIZE_THRESHOLD_BYTES_OVERRIDE"]


@pytest.mark.usefixtures("enable_chunking", "enable_batcher")
@run_with_pet(nproc=WORLD_SIZE)
def test_ddp_simple(layer_shapes: List[List[int]], tmp_path: Path) -> None:
    """
    Randomly initialize two DDP-wrapped models. Take the snapshot of one
    model and use the snapshot to restore the other model. Verify that the
    two model's state dicts are equal after restoration.
    """
    dist.init_process_group(backend="gloo")

    src = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    dst = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    src_ddp = DistributedDataParallel(src)
    dst_ddp = DistributedDataParallel(dst)

    assert not check_state_dict_eq(dst_ddp.state_dict(), src_ddp.state_dict())

    snapshot = torchsnapshot.Snapshot.take(
        path=str(tmp_path), app_state={"ddp": src_ddp}, replicated=["**"]
    )
    snapshot.restore(app_state={"ddp": dst_ddp})
    assert check_state_dict_eq(dst_ddp.state_dict(), src_ddp.state_dict())
