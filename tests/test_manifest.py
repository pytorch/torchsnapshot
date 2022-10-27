#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from typing import Dict, Generator
from unittest.mock import patch

import pytest

import yaml

from _pytest.fixtures import SubRequest  # @manual

from torchsnapshot.manifest import (
    ChunkedTensorEntry,
    DictEntry,
    Entry,
    is_replicated,
    ObjectEntry,
    Shard,
    ShardedTensorEntry,
    SnapshotMetadata,
    TensorEntry,
)
from torchsnapshot.manifest_ops import get_manifest_for_rank

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper


_WORLD_SIZE = 2
_MANIFEST_0: Dict[str, Entry] = {
    "0/foo": DictEntry(
        keys=["bar", "baz", "qux", "quux", "qux_chunked", "quux_chunked"]
    ),
    "0/foo/bar": ObjectEntry(
        location="0/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "0/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "0/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[0, 0],
                sizes=[4, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 8],
                    replicated=False,
                ),
            )
        ]
    ),
    "0/foo/quux": TensorEntry(
        location="0/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "0/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "0/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="0/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="0/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
    "1/foo": DictEntry(
        keys=["bar", "baz", "qux", "quux", "qux_chunked", "quux_chunked"]
    ),
    "1/foo/bar": ObjectEntry(
        location="1/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "1/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "1/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[4, 0],
                sizes=[4, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.1",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 8],
                    replicated=False,
                ),
            )
        ]
    ),
    "1/foo/quux": TensorEntry(
        location="1/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "1/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "1/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="1/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="1/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
}

# Same as _MANIFEST_0 expect that replicated entries only exist on rank 0
_MANIFEST_1: Dict[str, Entry] = {
    k: v
    for k, v in _MANIFEST_0.items()
    if not (k.startswith("1/") and is_replicated(v))
}


@pytest.fixture(params=[True, False])
def use_cyaml(request: SubRequest) -> Generator[None, None, None]:
    if request.param:
        from yaml import CSafeLoader

        with patch("torchsnapshot.manifest.Loader", CSafeLoader):
            yield
    else:
        from yaml import SafeLoader

        with patch("torchsnapshot.manifest.Loader", SafeLoader):
            yield


@pytest.mark.usefixtures("use_cyaml")
@pytest.mark.parametrize("manifest", [_MANIFEST_0, _MANIFEST_1])
def test_manifest_yaml_serialization(manifest: Dict[str, Entry]) -> None:
    metadata = SnapshotMetadata(
        version="0.0.0",
        world_size=_WORLD_SIZE,
        manifest=manifest,
    )
    yaml_str = metadata.to_yaml()
    loaded_metadata = SnapshotMetadata.from_yaml(yaml_str=yaml_str)
    assert metadata.manifest == loaded_metadata.manifest


@pytest.mark.usefixtures("use_cyaml")
@pytest.mark.parametrize("manifest", [_MANIFEST_0, _MANIFEST_1])
def test_manifest_yaml_dumper(manifest: Dict[str, Entry]) -> None:
    """
    :func:`SnapshotMetadata.to_yaml` switched to :func:`json.dumps`` to help
    with the serialization performance. This test verifies that old snapshot
    metadata serialized with :func:`yaml.dump` are still loadable.
    """
    metadata = SnapshotMetadata(
        version="0.0.0",
        world_size=_WORLD_SIZE,
        manifest=manifest,
    )
    yaml_str = yaml.dump(asdict(metadata), sort_keys=False, Dumper=Dumper)
    json_str = metadata.to_yaml()
    metadata_from_yaml = SnapshotMetadata.from_yaml(yaml_str=yaml_str)
    metadata_from_json = SnapshotMetadata.from_yaml(yaml_str=json_str)
    assert metadata_from_json == metadata_from_yaml


@pytest.mark.parametrize("manifest", [_MANIFEST_0, _MANIFEST_1])
@pytest.mark.parametrize("rank", range(_WORLD_SIZE * 2))
def test_get_local_manifest(manifest: Dict[str, Entry], rank: int) -> None:
    metadata = SnapshotMetadata(
        version="0.0.0",
        world_size=_WORLD_SIZE,
        manifest=manifest,
    )
    local_manifest, merged_sd_entries = get_manifest_for_rank(
        metadata=metadata, rank=rank
    )
    expected_local_manifest = {}
    for path, entry in manifest.items():
        local_path = "/".join(path.split("/")[1:])
        if path.startswith(f"{rank}/") or is_replicated(entry):
            expected_local_manifest[local_path] = entry

    if "foo/qux" in local_manifest:
        expected_local_manifest["foo/qux"] = ShardedTensorEntry(
            shards=[
                Shard(
                    offsets=[0, 0],
                    sizes=[4, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.0",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 8],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[4, 0],
                    sizes=[4, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.1",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 8],
                        replicated=False,
                    ),
                ),
            ]
        )
    if rank >= _WORLD_SIZE:
        expected_local_manifest["foo"] = DictEntry(keys=["baz", "qux_chunked"])
    assert local_manifest == expected_local_manifest


@pytest.mark.parametrize("rank", range(_WORLD_SIZE * 2))
def test_replicated_entries_only_on_rank_0(rank: int) -> None:
    """
    Previously, replicated entries were recorded under all ranks. Later, as an
    optimization, replicated entries were only recorded under rank 0. This test
    verifies that the optimization is backward compatible with the old format.
    """
    local_manifest_0 = get_manifest_for_rank(
        metadata=SnapshotMetadata(
            version="0.0.0",
            world_size=_WORLD_SIZE,
            manifest=_MANIFEST_0,
        ),
        rank=rank,
    )
    local_manifest_1 = get_manifest_for_rank(
        metadata=SnapshotMetadata(
            version="0.0.0",
            world_size=_WORLD_SIZE,
            manifest=_MANIFEST_0,
        ),
        rank=rank,
    )
    assert local_manifest_0 == local_manifest_1
