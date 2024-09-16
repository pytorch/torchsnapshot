#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import uuid
from typing import Optional

import torch
from torch import distributed as dist, nn
from torch.distributed import init_device_mesh
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torchsnapshot import Snapshot
from torchsnapshot.test_utils import check_state_dict_eq
from torchsnapshot.tricks.fsdp import FSDPOptimizerAdapter

logger: logging.Logger = logging.getLogger(__name__)


WORLD_SIZE: int = 4


class DummyModel(torch.nn.Module):
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    # pyre-fixme[3]: Return type must be annotated.
    def get_input(self):
        return torch.rand(4, 8, device="cuda")


# TODO: Test different world sizes (may require not using DTensorTestBase)
# TODO: Test FSDP + TP once dim_map is updated for [Shard(0), Shard(0)] cases
class TestSnapshotWithDTensor(DTensorTestBase):
    # pyre-fixme[3]: Return type must be annotated.
    def _create_model(
        self, seed: int, optim_lr: float, device_mesh: Optional[DeviceMesh] = None
    ):
        torch.manual_seed(seed)
        # Using HSDP model as an example model that uses DTensor
        # This should create model with placements
        # [Replicate(), Shard(0)]
        if device_mesh:
            model = FSDP(
                DummyModel().cuda(),
                device_mesh=device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        else:
            mesh_2d = init_device_mesh("cuda", (2, WORLD_SIZE // 2))
            intra_node_pg = mesh_2d.get_group(mesh_dim=1)
            inter_node_pg = mesh_2d.get_group(mesh_dim=0)
            model = FSDP(
                DummyModel().cuda(),
                process_group=(intra_node_pg, inter_node_pg),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(),
            optim_state_dict_config=ShardedOptimStateDictConfig(),
        )

        # Need to step and zero_grad in order to initialize all the optimizer parameters
        optim = torch.optim.Adam(model.parameters(), lr=optim_lr)
        optim.step(closure=None)
        optim.zero_grad(set_to_none=True)

        optim = FSDPOptimizerAdapter(model, optim)

        return model, optim

    @with_comms
    @skip_if_lt_x_gpu(WORLD_SIZE)
    # pyre-fixme[3]: Return type must be annotated.
    def test_save_and_load_same_world_size(self):
        mesh_2d = init_device_mesh("cuda", (2, WORLD_SIZE // 2))
        src_model, src_optim = self._create_model(
            seed=42, optim_lr=0.1, device_mesh=mesh_2d
        )
        dst_model, dst_optim = self._create_model(
            seed=24, optim_lr=0.2, device_mesh=mesh_2d
        )
        assert not check_state_dict_eq(src_model.state_dict(), dst_model.state_dict())
        assert not check_state_dict_eq(src_optim.state_dict(), dst_optim.state_dict())

        tmp_path = f"/tmp/{uuid.uuid4()}"
        if dist.get_rank() == 0:
            logger.info(f"Saving to {tmp_path}")

        snapshot = Snapshot.take(
            str(tmp_path), {"model": src_model, "optim": src_optim}
        )
        snapshot.restore({"model": dst_model, "optim": dst_optim})
        logging.info(f"{dst_model.state_dict()}")
        assert check_state_dict_eq(dst_model.state_dict(), src_model.state_dict())
        assert check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())
