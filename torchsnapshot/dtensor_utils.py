#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Iterator, List, Set

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


class _ReplicatedShards:
    """
    Utility class to collect ranks that each DTensor shard is replicated on
    in a convenient wrapper that allows efficient querying for all ranks
    that contain the same shard as a given rank.

    For example, a DTensor has a device mesh [[0,1,2],[3,4,5]]. It is replicated
    across mesh dim 1 and sharded across mesh dim 0. Thus, rank sets {0,1,2} and {3,4,5}
    would denote the two replicated shards. Then, the following queries would return:
        - 1 -> {0, 1, 2}
        - 5 -> {3, 4, 5}

    Attributes:
        replicated_ranks_for_shards (List[Set]): List of sets of ranks that each shard is
            replicated on. Length of list should be number of shards.
    """

    def __init__(self, replicated_ranks_for_shards: List[Set[int]]) -> None:
        self.repranks = replicated_ranks_for_shards
        self.lookup: Dict[int, Set[int]] = {}
        for rankset in self.repranks:
            for rank in rankset:
                self.lookup[rank] = rankset

    def get_all_replicated_ranks(self, rank: int) -> Set[int]:
        return self.lookup.get(rank, set())

    def __iter__(self) -> Iterator[Set[int]]:
        return iter(self.repranks)
