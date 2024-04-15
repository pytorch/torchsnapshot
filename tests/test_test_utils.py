#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard as ShardedTensorShard,
    ShardedTensor,
)
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torchsnapshot.test_utils import (
    assert_state_dict_eq,
    check_state_dict_eq,
    get_pet_launch_config,
)


class TestUtilsTest(unittest.TestCase):
    """
    Watch the watchmen.
    """

    def test_assert_state_dict_eq(self) -> None:
        t0 = torch.rand(16, 16)
        t1 = torch.rand(16, 16)
        a = {"foo": t0, "bar": [t1], "baz": 42}
        b = {"foo": t0, "bar": [t1], "baz": 42}
        c = {"foo": t0, "bar": [t0], "baz": 42}
        d = {"foo": t1, "bar": [t1], "baz": 42}
        e = {"foo": t0, "bar": [t1], "baz": 43}

        assert_state_dict_eq(self, a, b)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, c)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, d)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, e)

    def test_check_state_dict_eq(self) -> None:
        t0 = torch.rand(16, 16)
        t1 = torch.rand(16, 16)
        a = {"foo": t0, "bar": [t1], "baz": 42}
        b = {"foo": t0, "bar": [t1], "baz": 42}
        c = {"foo": t0, "bar": [t0], "baz": 42}
        d = {"foo": t1, "bar": [t1], "baz": 42}
        e = {"foo": t0, "bar": [t1], "baz": 43}

        self.assertTrue(check_state_dict_eq(a, b))
        self.assertFalse(check_state_dict_eq(a, c))
        self.assertFalse(check_state_dict_eq(a, d))
        self.assertFalse(check_state_dict_eq(a, e))

    @staticmethod
    def _create_sharded_tensor() -> ShardedTensor:
        dim_0: int = 128
        dim_1: int = 16

        global_tensor = torch.rand((dim_0, dim_1))

        rank = dist.get_rank()
        world_sz = dist.get_world_size()
        chunk_sz = int(dim_0 / world_sz)
        begin = rank * chunk_sz

        shard_view = torch.narrow(global_tensor, 0, begin, chunk_sz)
        shard = ShardedTensorShard(
            tensor=shard_view,
            metadata=ShardMetadata(
                shard_offsets=[begin, 0],
                shard_sizes=[chunk_sz, dim_1],
                placement=f"rank:{rank}/cpu",
            ),
        )
        return init_from_local_shards([shard], (dim_0, dim_1))

    @classmethod
    def _worker_sharded_tensor(cls) -> None:
        dist.init_process_group(backend="gloo")

        torch.manual_seed(42)
        foo = {"": cls._create_sharded_tensor()}
        torch.manual_seed(42)
        bar = {"": cls._create_sharded_tensor()}
        torch.manual_seed(777)
        baz = {"": cls._create_sharded_tensor()}

        tc = unittest.TestCase()
        assert_state_dict_eq(tc, foo, foo)
        assert_state_dict_eq(tc, foo, bar)
        with tc.assertRaises(AssertionError):
            assert_state_dict_eq(tc, foo, baz)

        tc.assertTrue(check_state_dict_eq(foo, foo))
        tc.assertTrue(check_state_dict_eq(foo, bar))
        tc.assertFalse(check_state_dict_eq(foo, baz))

    def test_state_dict_eq_with_sharded_tensor(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._worker_sharded_tensor)()

    @staticmethod
    def _create_dtensor() -> DTensor:
        dim_0: int = 128
        dim_1: int = 16

        local_tensor = torch.rand((dim_0, dim_1))

        # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[typing.Any],...
        mesh = DeviceMesh("cpu", mesh=[[0, 1], [2, 3]])
        placements = [Replicate(), Shard(0)]
        dtensor = distribute_tensor(
            tensor=local_tensor, device_mesh=mesh, placements=placements
        )

        return dtensor

    @classmethod
    def _worker_dtensor(cls) -> None:
        dist.init_process_group(backend="gloo")

        torch.manual_seed(42)
        foo = {"": cls._create_dtensor()}
        torch.manual_seed(42)
        bar = {"": cls._create_dtensor()}
        torch.manual_seed(777)
        baz = {"": cls._create_dtensor()}

        tc = unittest.TestCase()
        assert_state_dict_eq(tc, foo, foo)
        assert_state_dict_eq(tc, foo, bar)
        with tc.assertRaises(AssertionError):
            assert_state_dict_eq(tc, foo, baz)

        tc.assertTrue(check_state_dict_eq(foo, foo))
        tc.assertTrue(check_state_dict_eq(foo, bar))
        tc.assertFalse(check_state_dict_eq(foo, baz))

    def test_state_dict_eq_with_dtensor(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._worker_dtensor)()
