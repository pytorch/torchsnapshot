#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

from typing import Any, Dict
from unittest import mock

import torch
import torch.distributed.launcher as pet


def assert_state_dict_eq(
    tc: unittest.TestCase,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    lhs: Dict[Any, Any],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    rhs: Dict[Any, Any],
) -> None:
    """
    assertDictEqual except that it knows how to handle tensors.

    Args:
        tc: The test case.
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, torch.Tensor):
            return False
        return torch.allclose(self, other)

    with mock.patch("torch.Tensor.__eq__", __eq__):
        tc.assertDictEqual(lhs, rhs)


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
def check_state_dict_eq(lhs: Dict[Any, Any], rhs: Dict[Any, Any]) -> bool:
    """
    dict.__eq__ except that it knows how to handle tensors.

    Args:
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.

    Returns:
        Whether the two dictionaries are equal.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, torch.Tensor):
            return False
        return torch.allclose(self, other)

    with mock.patch("torch.Tensor.__eq__", __eq__):
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
