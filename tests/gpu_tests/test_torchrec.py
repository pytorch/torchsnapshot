#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import logging
import os
import tempfile
import unittest
from typing import List

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet

import torchsnapshot

try:
    import torchrec
except Exception:
    raise unittest.SkipTest("torchrec not found")

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed import ModuleSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingPlan, ShardingType

from torchrec.models.dlrm import DLRM, DLRMTrain
from torchsnapshot.io_preparer import ShardedTensorIOPreparer
from torchsnapshot.test_utils import (
    _tensor_eq,
    assert_state_dict_eq,
    check_state_dict_eq,
    get_pet_launch_config,
)

# Each embedding table is about 100MB in size
_EMBEDDING_DIM = 128
_NUM_EMBEDDINGS = 200_000
_DENSE_IN_FEATURES = 128
_NUM_CLASSES = 8

_TABLES = [
    torchrec.EmbeddingBagConfig(
        name="t1",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_NUM_EMBEDDINGS,
        feature_names=["f1"],
        pooling=torchrec.PoolingType.SUM,
    ),
    torchrec.EmbeddingBagConfig(
        name="t2",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_NUM_EMBEDDINGS,
        feature_names=["f2"],
        pooling=torchrec.PoolingType.SUM,
    ),
    torchrec.EmbeddingBagConfig(
        name="t3",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_NUM_EMBEDDINGS,
        feature_names=["f3"],
        pooling=torchrec.PoolingType.SUM,
    ),
    torchrec.EmbeddingBagConfig(
        name="t4",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_NUM_EMBEDDINGS,
        feature_names=["f4"],
        pooling=torchrec.PoolingType.SUM,
    ),
]

_SHARDERS: List[ModuleSharder] = [
    EmbeddingBagCollectionSharder(
        fused_params={
            "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "learning_rate": 0.01,
            "eps": 0.01,
        }
    )
]


class TorchrecTest(unittest.TestCase):
    @staticmethod
    def _get_rowwise_sharding_plan(
        module: torch.nn.Module, device: torch.device
    ) -> ShardingPlan:
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=dist.get_world_size(), compute_device=device.type
            ),
            constraints={
                table.name: ParameterConstraints(
                    sharding_types=[ShardingType.ROW_WISE.value]
                )
                for table in _TABLES
            },
        )
        pg = dist.group.WORLD
        assert pg is not None
        return planner.collective_plan(
            module,
            _SHARDERS,
            # pyre-fixme[6]: For 3rd param expected `ProcessGroup` but got
            #  `ProcessGroup`.
            pg,
        )

    @classmethod
    def _initialize_dmp(
        cls, device: torch.device
    ) -> torchrec.distributed.DistributedModelParallel:
        dlrm_model = DLRM(
            embedding_bag_collection=torchrec.EmbeddingBagCollection(
                device=torch.device("meta"),
                tables=_TABLES,
            ),
            dense_in_features=_DENSE_IN_FEATURES,
            dense_arch_layer_sizes=[64, _EMBEDDING_DIM],
            over_arch_layer_sizes=[64, _NUM_CLASSES],
        )
        model = DLRMTrain(dlrm_model)
        return torchrec.distributed.DistributedModelParallel(
            module=model,
            device=device,
            plan=cls._get_rowwise_sharding_plan(model, device),
            sharders=_SHARDERS,
        )

    @classmethod
    def _test_take_restore(
        cls, path: str, max_shard_sz_bytes: int, use_async: bool
    ) -> None:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger = logging.getLogger("torchsnapshot.scheduler")
        logger.setLevel(logging.DEBUG)

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        ShardedTensorIOPreparer.DEFAULT_MAX_SHARD_SIZE_BYTES = max_shard_sz_bytes

        # First, initialize a dmp with a certain random seed
        # IMPORTANT: seed different rank differently
        torch.manual_seed(42 + dist.get_rank())
        dmp_0 = cls._initialize_dmp(device)

        # Take a snapshot of dmp_0
        if use_async:
            future = torchsnapshot.Snapshot.async_take(
                path=path, app_state={"dmp": dmp_0}
            )
            snapshot = future.wait()
        else:
            snapshot = torchsnapshot.Snapshot.take(path=path, app_state={"dmp": dmp_0})

        # Initialize another dmp with a different random seed
        torch.manual_seed(777 + dist.get_rank())
        dmp_1 = cls._initialize_dmp(device)

        tc = unittest.TestCase()
        tc.maxDiff = None

        # Sanity check that the state dicts of the two dmps are different
        tc.assertFalse(check_state_dict_eq(dmp_0.state_dict(), dmp_1.state_dict()))

        # Restore dmp_1 with dmp_0's snapshot, after which the state dicts of
        # the two dmps should be the same
        snapshot.restore(app_state={"dmp": dmp_1})
        assert_state_dict_eq(tc, dmp_0.state_dict(), dmp_1.state_dict())

        # Initialize another dmp to verify the behavior of snapshot.loadd_entry
        del dmp_1
        torch.manual_seed(420 + dist.get_rank())
        dmp_2 = cls._initialize_dmp(device)

        t1_weight_key = (
            "model.sparse_arch.embedding_bag_collection.embedding_bags.t1.weight"
        )
        dmp_0_t1_weight = dmp_0.state_dict()[t1_weight_key]
        dmp_2_t1_weight = dmp_2.state_dict()[t1_weight_key]

        # Since dmp_0 and dmp_2 were initialized with different random seeds,
        # their t1 weight should be different
        tc.assertFalse(_tensor_eq(dmp_2_t1_weight, dmp_0_t1_weight))

        # Load dmp_2's t1 weight from dmp_1's snapshot, after which the t1
        # weights of dmp_0 and dmp_2 should be the same
        t1_weight_entry_path = os.path.join("0/dmp", t1_weight_key)
        snapshot.read_object(path=t1_weight_entry_path, obj_out=dmp_2_t1_weight)
        tc.assertTrue(_tensor_eq(dmp_2_t1_weight, dmp_0_t1_weight))

    @classmethod
    def _test_resharding(cls, path: str) -> None:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        tc = unittest.TestCase()
        tc.maxDiff = None

        dmp_0 = cls._initialize_dmp(device)
        snapshot = torchsnapshot.Snapshot(path=path)
        snapshot.restore(app_state={"dmp": dmp_0})
        # TODO: verify weight

        torch.manual_seed(420 + dist.get_rank())
        dmp_1 = cls._initialize_dmp(device)

        t1_weight_key = (
            "model.sparse_arch.embedding_bag_collection.embedding_bags.t1.weight"
        )
        dmp_0_t1_weight = dmp_0.state_dict()[t1_weight_key]
        dmp_1_t1_weight = dmp_1.state_dict()[t1_weight_key]
        tc.assertFalse(_tensor_eq(dmp_1_t1_weight, dmp_0_t1_weight))

        t1_weight_entry_path = os.path.join("0/dmp", t1_weight_key)
        snapshot.read_object(path=t1_weight_entry_path, obj_out=dmp_1_t1_weight)
        tc.assertTrue(_tensor_eq(dmp_1_t1_weight, dmp_0_t1_weight))

    @unittest.skipUnless(torch.cuda.is_available(), "This test requires GPU to run.")
    def test_torchrec(self) -> None:
        for max_shard_sz_bytes in [16 * 1024 * 1024, 16 * 1024 * 1024 - 1]:
            with self.subTest(max_shard_sz_bytes=max_shard_sz_bytes):
                with tempfile.TemporaryDirectory() as path:
                    lc = get_pet_launch_config(nproc=4)
                    pet.elastic_launch(lc, entrypoint=self._test_take_restore)(
                        path, max_shard_sz_bytes, False
                    )
                    pet.elastic_launch(lc, entrypoint=self._test_resharding)(path)

    @unittest.skipUnless(torch.cuda.is_available(), "This test requires GPU to run.")
    def test_torchrec_async(self) -> None:
        for max_shard_sz_bytes in [16 * 1024 * 1024, 16 * 1024 * 1024 - 1]:
            with self.subTest(max_shard_sz_bytes=max_shard_sz_bytes):
                with tempfile.TemporaryDirectory() as path:
                    lc = get_pet_launch_config(nproc=4)
                    pet.elastic_launch(lc, entrypoint=self._test_take_restore)(
                        path, max_shard_sz_bytes, True
                    )
                    pet.elastic_launch(lc, entrypoint=self._test_resharding)(path)
