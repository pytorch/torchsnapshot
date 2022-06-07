#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchsnapshot.manifest import (
    DictEntry,
    get_available_entries,
    ObjectEntry,
    Shard,
    ShardedTensorEntry,
)

_MANIFEST = {
    "0/foo": DictEntry(keys=["bar", "baz", "qux"]),
    "0/foo/bar": ObjectEntry(type="Bar", location="0/foo/bar", replicated=False),
    "0/foo/baz": ObjectEntry(
        type="Baz", location="replicated/foo/baz", replicated=True
    ),
    "0/foo/qux": ShardedTensorEntry(
        shards=[Shard(offsets=[0, 0], sizes=[4, 4], location="sharded/foo/qux.0")]
    ),
    "1/foo": DictEntry(keys=["bar", "baz", "qux"]),
    "1/foo/bar": ObjectEntry(type="Bar", location="1/foo/bar", replicated=False),
    "1/foo/baz": ObjectEntry(
        type="Baz", location="replicated/foo/baz", replicated=True
    ),
    "1/foo/qux": ShardedTensorEntry(
        shards=[Shard(offsets=[4, 0], sizes=[4, 4], location="sharded/foo/qux.1")]
    ),
}


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_load_with_same_world_size(self) -> None:
        available_entries = get_available_entries(_MANIFEST, 0)
        expected_available_entries = {
            "foo/bar": ObjectEntry(type="Bar", location="0/foo/bar", replicated=False),
            "foo/baz": ObjectEntry(
                type="Baz", location="replicated/foo/baz", replicated=True
            ),
            "foo/qux": ShardedTensorEntry(
                shards=[
                    Shard(offsets=[0, 0], sizes=[4, 4], location="sharded/foo/qux.0"),
                    Shard(offsets=[4, 0], sizes=[4, 4], location="sharded/foo/qux.1"),
                ]
            ),
        }
        self.assertDictEqual(available_entries, expected_available_entries)

    def test_load_with_larger_world_size(self) -> None:
        available_entries = get_available_entries(_MANIFEST, 42)
        expected_available_entries = {
            "foo/baz": ObjectEntry(
                type="Baz", location="replicated/foo/baz", replicated=True
            ),
            "foo/qux": ShardedTensorEntry(
                shards=[
                    Shard(offsets=[0, 0], sizes=[4, 4], location="sharded/foo/qux.0"),
                    Shard(offsets=[4, 0], sizes=[4, 4], location="sharded/foo/qux.1"),
                ]
            ),
        }
        self.assertDictEqual(available_entries, expected_available_entries)
