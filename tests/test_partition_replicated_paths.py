#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import List

import torch
from torchsnapshot import Snapshot


class PartitionReplicatedPathsTest(unittest.TestCase):
    @staticmethod
    def _check_all_elements_of_list_equal(
        expected: List[List[str]], actual: List[List[str]]
    ) -> bool:
        return all(set(a) == set(b) for a, b in zip(expected, actual))

    def test_more_paths_than_ranks(self) -> None:
        replicated_paths = [
            "/tmp/foo",
            "/tmp/bar",
            "/tmp/quaz",
        ]
        flattened = {
            replicated_paths[0]: torch.zeros(1000),
            replicated_paths[1]: torch.zeros(500),
            replicated_paths[2]: torch.zeros(200),
        }
        world_size = 2

        paths_partition = Snapshot._partition_replicated_paths(
            replicated_paths,
            flattened,
            world_size,
        )

        expected_paths_partition = [
            [replicated_paths[0]],
            [replicated_paths[1], replicated_paths[2]],
        ]
        self.assertTrue(
            PartitionReplicatedPathsTest._check_all_elements_of_list_equal(
                expected=expected_paths_partition, actual=paths_partition
            )
        )

    def test_equal_number_of_paths_and_ranks(self) -> None:
        replicated_paths = [
            "/tmp/foo",
            "/tmp/bar",
            "/tmp/quaz",
            "/tmp/foofoo",
            "/tmp/barbar",
        ]

        flattened = {
            replicated_paths[0]: torch.zeros(9),
            replicated_paths[1]: torch.zeros(10),
            replicated_paths[2]: torch.zeros(995),
            replicated_paths[3]: torch.zeros(999),
            replicated_paths[4]: torch.zeros(1000),
        }

        world_size = 5

        paths_partition = Snapshot._partition_replicated_paths(
            replicated_paths,
            flattened,
            world_size,
        )
        expected_paths_partition = [
            [replicated_paths[4]],
            [replicated_paths[3]],
            [replicated_paths[2]],
            [replicated_paths[1]],
            [replicated_paths[0]],
        ]

        self.assertTrue(
            PartitionReplicatedPathsTest._check_all_elements_of_list_equal(
                expected=expected_paths_partition, actual=paths_partition
            )
        )

    def test_more_ranks_than_paths(self) -> None:
        replicated_paths = [
            "/tmp/foo",
            "/tmp/bar",
            "/tmp/quaz",
        ]

        flattened = {
            replicated_paths[0]: torch.zeros(100),
            replicated_paths[1]: torch.zeros(50),
            replicated_paths[2]: torch.zeros(1000),
        }

        world_size = 6

        paths_partition = Snapshot._partition_replicated_paths(
            replicated_paths,
            flattened,
            world_size,
        )
        expected_paths_partition = [
            [replicated_paths[2]],
            [replicated_paths[0]],
            [replicated_paths[1]],
            [],
            [],
            [],
        ]

        self.assertTrue(
            PartitionReplicatedPathsTest._check_all_elements_of_list_equal(
                expected=expected_paths_partition, actual=paths_partition
            )
        )
