#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import io
import unittest
from typing import List, Tuple

import torch

import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot.manifest as manifest
from torch.distributed._shard.metadata import ShardMetadata

from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard

from torchsnapshot.io_preparer import ShardedTensorIOPreparer
from torchsnapshot.serialization import string_to_dtype, tensor_from_memoryview
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
        shard_copy = copy.deepcopy(shard_view)
        shard = Shard(
            tensor=shard_view,
            metadata=ShardMetadata(
                shard_offsets=[begin, 0],
                shard_sizes=[chunk_sz, dim_1],
                placement=f"rank:{rank}/cpu",
            ),
        )
        sharded_tensor = init_from_local_shards([shard], (dim_0, dim_1))

        entry, write_reqs = ShardedTensorIOPreparer.prepare_write(
            "/foo", sharded_tensor
        )

        tc = unittest.TestCase()

        # The path is determined by torch.distributed which is experiemental
        # and subject to change
        tc.assertEqual(
            entry,
            manifest.ShardedTensorEntry(
                shards=[
                    manifest.Shard(
                        offsets=[begin, 0],
                        sizes=[chunk_sz, dim_1],
                        tensor=manifest.TensorEntry(
                            location=f"/foo_{begin}_0",
                            serializer="buffer_protocol",
                            dtype="torch.float32",
                            shape=[chunk_sz, dim_1],
                            replicated=False,
                        ),
                    )
                ]
            ),
        )

        # For this sharded tensor, each rank writes 1 shard
        tc.assertEqual(len(write_reqs), 1)

        # The path is determined by torch.distributed which is experiemental
        # and subject to change
        tc.assertEqual(write_reqs[0].path, f"/foo_{begin}_0")

        loop = asyncio.new_event_loop()
        buf = loop.run_until_complete(write_reqs[0].buffer_stager.stage_buffer())

        # Make sure only the data described by the view gets persisted
        loaded = tensor_from_memoryview(
            mv=memoryview(buf),
            dtype=string_to_dtype(entry.shards[0].tensor.dtype),
            shape=entry.shards[0].tensor.shape,
        )
        tc.assertEqual(loaded.nelement(), loaded.storage().size())

        # Randomize the original sharded tensor before restoring
        torch.nn.init.normal_(shard_view, mean=0, std=1.0)
        tc.assertFalse(torch.allclose(shard_view, shard_copy))

        read_reqs = ShardedTensorIOPreparer.prepare_read(entry, sharded_tensor)

        # For this sharded tensor, each rank writes 1 shard
        tc.assertEqual(len(read_reqs), 1)
        if isinstance(buf, memoryview):
            buf = buf.tobytes()
        loop.run_until_complete(read_reqs[0].buffer_consumer.consume_buffer(buf))

        # Verify that the original sharded tensor gets restored
        tc.assertTrue(torch.allclose(shard_view, shard_copy))

    def test_sharded_tensor_io_preparer(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._worker)()

    def _verify_subdivided_shards(
        self,
        subdivided: List[Tuple[torch.Tensor, List[int], List[int]]],
        dim: int,
        expected_num_sub_shards: int,
        expected_combined: torch.Tensor,
        expected_offsets: List[int],
        expected_sizes: List[int],
    ) -> None:
        """
        Combine the tensor, offsets, sizes of a subdivision result and verify
        the correctness along the way.
        """
        _, offsets, sizes = copy.deepcopy(subdivided[0])
        for _, sub_offsets, sub_sizes in subdivided[1:]:
            self.assertEqual(len(offsets), len(sub_offsets))
            self.assertEqual(len(sizes), len(sub_sizes))
            for i in range(len(offsets)):
                if i != dim:
                    self.assertEqual(sub_offsets[i], offsets[i])
                    self.assertEqual(sub_sizes[i], sizes[i])
            self.assertEqual(sub_offsets[dim], offsets[dim] + sizes[dim])
            sizes[dim] += sub_sizes[dim]

        sub_views = [sub_view for sub_view, _, _ in subdivided]
        combined = torch.concat(sub_views, dim)
        self.assertEqual(len(subdivided), expected_num_sub_shards)
        self.assertTrue(torch.allclose(combined, expected_combined))
        self.assertEqual(offsets, expected_offsets)
        self.assertEqual(sizes, expected_sizes)

    def test_subdivide_shard(self) -> None:
        tensor = torch.randn(256, 256)
        # max_shard_sz_bytes is smaller than the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=77,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=256,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is between 1x and 2x the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=1999,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=256,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than 2x of the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=4001,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=86,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than 2x of the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[0, 512],
            sizes=[256, 256],
            dim=1,
            max_shard_sz_bytes=4001,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=1,
            expected_num_sub_shards=86,
            expected_combined=tensor,
            expected_offsets=[0, 512],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than the shard size
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=300000,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=1,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )
