#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

import torch.distributed as dist
import torch.distributed.launcher as pet
from torch.distributed._shard.metadata import ShardMetadata

from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard

from torchsnapshot.io_preparer import ShardedTensorIOPreparer
from torchsnapshot.test_utils import get_pet_launch_config


class ShardedTensorIOPreparerTest(unittest.TestCase):
    @staticmethod
    def _worker() -> None:
        """
        Perform a series of sanity test against ShardedTensorIOPreparer.
        """
        dim_0: int = 128
        dim_1: int = 16

        dist.init_process_group(backend="gloo")
        torch.manual_seed(42)
        global_tensor = torch.rand((dim_0, dim_1))

        rank = dist.get_rank()
        world_sz = dist.get_world_size()
        chunk_sz = int(dim_0 / world_sz)
        begin = rank * chunk_sz

        shard_view = torch.narrow(global_tensor, 0, begin, chunk_sz)
        shard = Shard(
            tensor=shard_view,
            metadata=ShardMetadata(
                shard_offsets=[begin, 0],
                shard_sizes=[chunk_sz, dim_1],
                placement=f"rank:{rank}/cpu",
            ),
        )
        sharded_tensor = init_from_local_shards([shard], (dim_0, dim_1))

        entry, obj_write_req = ShardedTensorIOPreparer.prepare_write(
            "/foo", sharded_tensor
        )

        tc = unittest.TestCase()
        # For this sharded tensor, each rank writes 1 shard
        tc.assertEqual(len(obj_write_req.io_reqs), 1)
        # The path is determined by torch.distributed which is experiemental
        # and subject to change
        tc.assertEqual(obj_write_req.io_reqs[0].path, f"/foo_{begin}_0")

        obj_write_req.io_reqs[0].buf.seek(0)
        loaded = torch.load(obj_write_req.io_reqs[0].buf)
        # Make sure only the data described by the view gets persisted
        tc.assertEqual(loaded.nelement(), loaded.storage().size())
        tc.assertTrue(torch.allclose(loaded, shard_view))

    def test_sharded_tensor_io_preparer(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._worker)()
