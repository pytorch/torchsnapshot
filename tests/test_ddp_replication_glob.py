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
import torchsnapshot
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot.manifest import is_replicated, SnapshotMetadata
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchsnapshot.stateful import AppState
from torchsnapshot.test_utils import get_pet_launch_config


class ReplicationDDPGlobTest(unittest.TestCase):
    @staticmethod
    def _worker(path: str, replication_globs: List[List[str]]) -> None:
        dist.init_process_group(backend="gloo")
        app_state: AppState = {
            "ddp": DDP(torch.nn.Linear(4, 3)),
            "nonddp": torch.nn.Linear(3, 2),
        }
        torchsnapshot.Snapshot.take(
            path=path,
            app_state=app_state,
            replicated=replication_globs[dist.get_rank()],
        )

    def _test_helper(
        self,
        nproc: int,
        replication_globs: List[List[str]],
        expected_replicated_paths: List[str],
    ) -> None:
        """
        Verify whether the supplied replication globs result in expected
        replicated paths.
        """
        lc = get_pet_launch_config(nproc=nproc)
        with tempfile.TemporaryDirectory() as path:
            pet.elastic_launch(lc, entrypoint=self._worker)(path, replication_globs)
            with open(os.path.join(path, SNAPSHOT_METADATA_FNAME)) as f:
                metadata = SnapshotMetadata.from_yaml(f.read())
        replicated_paths = [
            path for path in metadata.manifest if is_replicated(metadata.manifest[path])
        ]
        self.assertSetEqual(set(replicated_paths), set(expected_replicated_paths))

    def test_only_ddp_replicated(self) -> None:
        replication_globs = [[], []]
        expected_replicated_paths = [
            "0/ddp/module.weight",
            "0/ddp/module.bias",
            "1/ddp/module.weight",
            "1/ddp/module.bias",
        ]

        self._test_helper(2, replication_globs, expected_replicated_paths)

    def test_all_replicated(self) -> None:
        replication_globs = [["**"], ["**"]]
        expected_replicated_paths = [
            "0/ddp/module.weight",
            "0/ddp/module.bias",
            "1/ddp/module.weight",
            "1/ddp/module.bias",
            "0/nonddp/weight",
            "0/nonddp/bias",
            "1/nonddp/weight",
            "1/nonddp/bias",
        ]
        self._test_helper(2, replication_globs, expected_replicated_paths)
