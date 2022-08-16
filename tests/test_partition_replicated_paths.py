#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, List, Tuple

from torchsnapshot import Snapshot
from torchsnapshot.io_preparer import Chunk

_CHUNKING_INSTRUCTION_T = Dict[str, List[Chunk]]


class PartitionReplicatedPathsTest(unittest.TestCase):
    @staticmethod
    def _check_all_elements_of_partition_equal(
        expected: List[Tuple[_CHUNKING_INSTRUCTION_T, List[str]]],
        actual: List[Tuple[_CHUNKING_INSTRUCTION_T, List[str]]],
    ) -> None:
        tc = unittest.TestCase()
        tc.maxDiff = None
        tc.assertListEqual(expected, actual)

    def test_more_paths_and_ranks(self) -> None:
        chunking_instructions = {
            "/tmp/foo": [Chunk(offsets=[0], sizes=[9], dtype="torch.float32")],
            "/tmp/bar": [Chunk(offsets=[0], sizes=[10], dtype="torch.float32")],
            "/tmp/quaz": [Chunk(offsets=[0], sizes=[995], dtype="torch.float32")],
            "/tmp/foofoo": [Chunk(offsets=[0], sizes=[999], dtype="torch.float32")],
            "/tmp/barbar": [Chunk(offsets=[0], sizes=[1000], dtype="torch.float32")],
        }
        partition_results = Snapshot._partition_replicated_paths(
            list(chunking_instructions.keys()),
            chunking_instructions,
            world_size=3,
        )
        expected_chunked_results: List[_CHUNKING_INSTRUCTION_T] = [
            {"/tmp/barbar": [Chunk(offsets=[0], sizes=[1000], dtype="torch.float32")]},
            {
                "/tmp/foofoo": [Chunk(offsets=[0], sizes=[999], dtype="torch.float32")],
                "/tmp/foo": [Chunk(offsets=[0], sizes=[9], dtype="torch.float32")],
            },
            {
                "/tmp/quaz": [Chunk(offsets=[0], sizes=[995], dtype="torch.float32")],
                "/tmp/bar": [Chunk(offsets=[0], sizes=[10], dtype="torch.float32")],
            },
        ]
        expected_nonchunked_results: List[List[str]] = [[], [], []]
        PartitionReplicatedPathsTest._check_all_elements_of_partition_equal(
            list(zip(expected_chunked_results, expected_nonchunked_results)),
            partition_results,
        )

    def test_equal_number_of_paths_and_ranks(self) -> None:
        # check multiple chunks corresponding to a path
        chunking_instructions = {
            "/tmp/foo": [Chunk(offsets=[0], sizes=[500], dtype="torch.float32")],
            "/tmp/bar": [
                Chunk(offsets=[0], sizes=[1000], dtype="torch.float32"),
                Chunk(offsets=[1000], sizes=[200], dtype="torch.float32"),
            ],
        }
        partition_results = Snapshot._partition_replicated_paths(
            list(chunking_instructions.keys()),
            chunking_instructions,
            world_size=2,
        )
        expected_chunked_results: List[_CHUNKING_INSTRUCTION_T] = [
            {"/tmp/bar": [Chunk(offsets=[0], sizes=[1000], dtype="torch.float32")]},
            {
                "/tmp/foo": [Chunk(offsets=[0], sizes=[500], dtype="torch.float32")],
                "/tmp/bar": [Chunk(offsets=[1000], sizes=[200], dtype="torch.float32")],
            },
        ]
        expected_nonchunked_results: List[List[str]] = [[], [], []]
        PartitionReplicatedPathsTest._check_all_elements_of_partition_equal(
            list(zip(expected_chunked_results, expected_nonchunked_results)),
            partition_results,
        )

    def test_more_ranks_than_paths(self) -> None:
        # check combination of chunked and nonchunkable paths
        replicated_paths = [
            "/tmp/foo_chunked",
            "/tmp/bar_nonchunked",
            "/tmp/quaz_nonchunked",
        ]
        chunking_instructions = {
            "/tmp/foo_chunked": [
                Chunk(offsets=[0, 0], sizes=[10, 100], dtype="torch.float32"),
                Chunk(offsets=[10, 0], sizes=[5, 100], dtype="torch.float32"),
                Chunk(offsets=[15, 0], sizes=[2, 100], dtype="torch.float32"),
            ]
        }
        partition_results = Snapshot._partition_replicated_paths(
            replicated_paths,
            chunking_instructions,
            world_size=5,
        )
        expected_chunked_results: List[_CHUNKING_INSTRUCTION_T] = [
            {
                "/tmp/foo_chunked": [
                    Chunk(offsets=[0, 0], sizes=[10, 100], dtype="torch.float32")
                ]
            },
            {
                "/tmp/foo_chunked": [
                    Chunk(offsets=[10, 0], sizes=[5, 100], dtype="torch.float32")
                ]
            },
            {
                "/tmp/foo_chunked": [
                    Chunk(offsets=[15, 0], sizes=[2, 100], dtype="torch.float32")
                ]
            },
            {},
            {},
            {},
        ]
        expected_nonchunked_results = (
            ["/tmp/bar_nonchunked"],
            ["/tmp/quaz_nonchunked"],
            [],
            [],
            [],
        )

        PartitionReplicatedPathsTest._check_all_elements_of_partition_equal(
            list(zip(expected_chunked_results, expected_nonchunked_results)),
            partition_results,
        )
