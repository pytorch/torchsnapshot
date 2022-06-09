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
from torch.nn.parallel import DistributedDataParallel
from torchsnapshot.test_utils import (
    assert_state_dict_eq,
    check_state_dict_eq,
    get_pet_launch_config,
)


class DDPTest(unittest.TestCase):
    @staticmethod
    def _worker(path: str) -> None:
        """
        Randomly initialize two DDP-wrapped models. Take the snapshot of one
        model and use the snapshot to restore the other model. Verify that the
        two model's state dicts are equal after restoration.
        """
        tc = unittest.TestCase()

        dist.init_process_group(backend="gloo")
        foo = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        bar = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        ddp_foo = DistributedDataParallel(foo)
        ddp_bar = DistributedDataParallel(bar)

        tc.assertFalse(check_state_dict_eq(ddp_foo.state_dict(), ddp_bar.state_dict()))

        snapshot = torchsnapshot.Snapshot.take(
            path=path, app_state={"model": ddp_foo}, replicated=["**"]
        )
        snapshot.restore(app_state={"model": ddp_bar})

        assert_state_dict_eq(tc, ddp_foo.state_dict(), ddp_bar.state_dict())

    def test_ddp(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        with tempfile.TemporaryDirectory() as path:
            pet.elastic_launch(lc, entrypoint=self._worker)(path)
