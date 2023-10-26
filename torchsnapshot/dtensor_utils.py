#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate, Shard


def is_sharded(tensor: torch.Tensor) -> bool:
    """
    Returns true if tensor is a ShardedTensor or a DTensor that is partially
    or fully sharded
    """
    if isinstance(tensor, ShardedTensor):
        return True
    elif isinstance(tensor, DTensor):
        for placement in tensor.placements:
            if isinstance(placement, Shard):
                return True
    return False


def is_replicated_dtensor(dtensor: DTensor) -> bool:
    """
    Returns true if DTensor is fully or partially replicated, false if fully sharded.
    """
    for placement in dtensor.placements:
        if isinstance(placement, Replicate):
            return True
    return False
