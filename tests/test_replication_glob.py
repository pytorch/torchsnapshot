#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot
from torchsnapshot.manifest import is_replicated, SnapshotMetadata
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchsnapshot.test_utils import get_pet_launch_config


class _TestStateful:
    def state_dict(self) -> Dict[str, Any]:
        return {
            "foo": torch.Tensor(1),
            "bar": torch.Tensor(1),
            "baz": [torch.Tensor(1), torch.Tensor(1)],
            "qux": {"quux": torch.Tensor(1), "quuz": torch.Tensor(1)},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError()


class ReplicationGlobTest(unittest.TestCase):
    @staticmethod
    def _worker(path: str, replication_globs: List[List[str]]) -> None:
        """
        Take a snapshot of a _TestStateful object with the given replication globs.
        """
        dist.init_process_group(backend="gloo")
        stateful = _TestStateful()
        app_state = {"my_stateful": stateful}
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

    def test_all_replicated(self) -> None:
        replication_globs = [["**"]] * 2
        expected_replicated_paths = [
            "0/my_stateful/foo",
            "0/my_stateful/bar",
            "0/my_stateful/baz/0",
            "0/my_stateful/baz/1",
            "0/my_stateful/qux/quux",
            "0/my_stateful/qux/quuz",
            "1/my_stateful/foo",
            "1/my_stateful/bar",
            "1/my_stateful/baz/0",
            "1/my_stateful/baz/1",
            "1/my_stateful/qux/quux",
            "1/my_stateful/qux/quuz",
        ]
        self._test_helper(2, replication_globs, expected_replicated_paths)

    def test_partially_replicated(self) -> None:
        replication_globs = [["my_stateful/baz/*", "my_stateful/qux/*"]] * 2
        expected_replicated_paths = [
            "0/my_stateful/baz/0",
            "0/my_stateful/baz/1",
            "0/my_stateful/qux/quux",
            "0/my_stateful/qux/quuz",
            "1/my_stateful/baz/0",
            "1/my_stateful/baz/1",
            "1/my_stateful/qux/quux",
            "1/my_stateful/qux/quuz",
        ]
        self._test_helper(2, replication_globs, expected_replicated_paths)

    def test_different_replication_globs_across_ranks(self) -> None:
        replication_globs = [
            ["my_stateful/foo", "my_stateful/qux/*"],
            ["my_stateful/foo", "my_stateful/baz/*"],
        ]
        expected_replicated_paths = [
            "0/my_stateful/foo",
            "1/my_stateful/foo",
        ]
        self._test_helper(2, replication_globs, expected_replicated_paths)
