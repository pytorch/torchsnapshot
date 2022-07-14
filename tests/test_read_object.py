#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torchsnapshot.test_utils import get_pet_launch_config


class ReadObjectTest(unittest.TestCase):
    def test_read_object(self) -> None:
        state = torchsnapshot.StateDict(
            foo=42,
            bar=torch.randn(20, 20),
        )

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(
                path=path, app_state={"state": state}
            )

            self.assertEqual(snapshot.read_object("0/state/foo"), 42)
            self.assertEqual(snapshot.read_object("0/state/foo", 777), 42)

            baz = torch.randn(20, 20)
            self.assertFalse(torch.allclose(baz, state["bar"]))

            loaded_bar = snapshot.read_object("0/state/bar", baz)
            self.assertEqual(id(loaded_bar), id(baz))
            self.assertNotEqual(id(loaded_bar), id(state["bar"]))
            self.assertTrue(torch.allclose(baz, state["bar"]))

    @staticmethod
    def _test_read_sharded_tensor() -> None:
        tc = unittest.TestCase()
        dist.init_process_group(backend="gloo")
        torch.manual_seed(42 + dist.get_rank())

        # pyre-ignore [28]
        spec = ChunkShardingSpec(
            dim=0,
            placements=[f"rank:{rank}/cpu" for rank in range(dist.get_world_size())],
        )
        foo = sharded_tensor.empty(spec, 20_000, 128)
        for shard in foo.local_shards():
            torch.nn.init.uniform_(shard.tensor)

        bar = sharded_tensor.empty(spec, 20_000, 128)
        for shard in bar.local_shards():
            torch.nn.init.uniform_(shard.tensor)

        for foo_shard, bar_shard in zip(foo.local_shards(), bar.local_shards()):
            tc.assertFalse(torch.allclose(foo_shard.tensor, bar_shard.tensor))

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(
                path=path, app_state={"state": torchsnapshot.StateDict(foo=foo)}
            )
            snapshot.read_object("0/state/foo", obj_out=bar)

        for foo_shard, bar_shard in zip(foo.local_shards(), bar.local_shards()):
            tc.assertTrue(torch.allclose(foo_shard.tensor, bar_shard.tensor))

    def test_read_sharded_tensor(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_read_sharded_tensor)()
