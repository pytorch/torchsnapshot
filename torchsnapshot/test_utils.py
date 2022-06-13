#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid
from contextlib import contextmanager

from typing import Any, Dict, Generator, Union
from unittest import mock

import torch
import torch.distributed.launcher as pet
from torch.distributed._shard.sharded_tensor import ShardedTensor


# pyre-ignore[2]: Parameter annotation cannot contain `Any`.
def _tensor_eq(lhs: Union[torch.Tensor, ShardedTensor], rhs: Any) -> bool:
    if type(lhs) != type(rhs):
        return False
    if isinstance(lhs, torch.Tensor):
        return torch.allclose(lhs, rhs)
    elif isinstance(lhs, ShardedTensor):
        for l_shard, r_shard in zip(lhs.local_shards(), rhs.local_shards()):
            if not torch.allclose(l_shard.tensor, r_shard.tensor):
                return False
        return True
    else:
        raise AssertionError("The lhs operand must be a Tensor or ShardedTensor.")


@contextmanager
def _patch_tensor_eq() -> Generator[None, None, None]:
    patchers = [
        mock.patch("torch.Tensor.__eq__", _tensor_eq),
        mock.patch(
            "torch.distributed._shard.sharded_tensor.ShardedTensor.__eq__", _tensor_eq
        ),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        yield
    finally:
        for patcher in patchers:
            patcher.stop()


def assert_state_dict_eq(
    tc: unittest.TestCase,
    # pyre-ignore[2]: Parameter annotation cannot contain `Any`.
    lhs: Dict[Any, Any],
    # pyre-ignore[2]: Parameter annotation cannot contain `Any`.
    rhs: Dict[Any, Any],
) -> None:
    """
    assertDictEqual except that it knows how to handle tensors.

    Args:
        tc: The test case.
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.
    """
    with _patch_tensor_eq():
        tc.assertDictEqual(lhs, rhs)


# pyre-ignore[2]: Parameter annotation cannot contain `Any`.
def check_state_dict_eq(lhs: Dict[Any, Any], rhs: Dict[Any, Any]) -> bool:
    """
    dict.__eq__ except that it knows how to handle tensors.

    Args:
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.

    Returns:
        Whether the two dictionaries are equal.
    """
    with _patch_tensor_eq():
        return lhs == rhs


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank tests.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank tests.
    """
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )
