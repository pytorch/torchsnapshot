#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from typing import List

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet

import torchrec
import torchsnapshot
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed import ModuleSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingPlan, ShardingType

from torchrec.models.dlrm import DLRM, DLRMTrain
from torchsnapshot.test_utils import (
    assert_state_dict_eq,
    check_state_dict_eq,
    get_pet_launch_config,
)

# Each embedding table is about 1GB in size
_EMBEDDING_DIM = 128
_NUM_EMBEDDINGS = 2_000_000
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
        return planner.collective_plan(
            module,
            _SHARDERS,
            dist.group.WORLD,
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
    def _worker(cls, path: str) -> None:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # It's important seed different rank differently
        torch.manual_seed(42 + dist.get_rank())
        dmp_0 = cls._initialize_dmp(device)
        snapshot = torchsnapshot.Snapshot.take(path=path, app_state={"dmp": dmp_0})

        torch.manual_seed(777 + dist.get_rank())
        dmp_1 = cls._initialize_dmp(device)

        tc = unittest.TestCase()
        tc.maxDiff = None

        tc.assertFalse(check_state_dict_eq(dmp_0.state_dict(), dmp_1.state_dict()))

        snapshot.restore(app_state={"dmp": dmp_1})
        assert_state_dict_eq(tc, dmp_0.state_dict(), dmp_1.state_dict())

    @unittest.skipIf(not torch.cuda.is_available(), "This test requires GPU to run.")
    def test_torchrec(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        with tempfile.TemporaryDirectory() as path:
            pet.elastic_launch(lc, entrypoint=self._worker)(path)
