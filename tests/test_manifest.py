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
    DictEntry,
    DTensorEntry,
    Entry,
    Shard,
    ShardedTensorEntry,
    SnapshotMetadata,
    TensorEntry,
)
from torchsnapshot.manifest_ops import get_manifest_for_rank
from torchsnapshot.manifest_utils import is_fully_replicated_entry

from .assets.manifest import (
    _MANIFEST_0,
    _MANIFEST_1,
    _MANIFEST_2,
    _MANIFEST_3,
    _WORLD_SIZE,
)

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper


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
@pytest.mark.parametrize(
    "manifest", [_MANIFEST_0, _MANIFEST_1, _MANIFEST_2, _MANIFEST_3]
)
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
@pytest.mark.parametrize(
    "manifest", [_MANIFEST_0, _MANIFEST_1, _MANIFEST_2, _MANIFEST_3]
)
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


@pytest.mark.parametrize(
    "manifest", [_MANIFEST_0, _MANIFEST_1, _MANIFEST_2, _MANIFEST_3]
)
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
        if path.startswith(f"{rank}/") or is_fully_replicated_entry(entry):
            expected_local_manifest[local_path] = entry

    merged_local_manifest = _update_local_manifest_with_merged_entries(local_manifest)
    expected_local_manifest.update(merged_local_manifest)

    if rank >= _WORLD_SIZE:
        # Pure replicated entries only
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


def _update_local_manifest_with_merged_entries(
    local_manifest: Dict[str, Entry]
) -> None:
    """
    Update the expected local manifest with manually merged ShardedTensorEntries
    and DTensorEntries. See get_manifest_for_rank for more details.
    """
    merged_local_manifest = {}
    if "foo/qux" in local_manifest:
        merged_local_manifest["foo/qux"] = ShardedTensorEntry(
            shards=[
                Shard(
                    offsets=[0, 0],
                    sizes=[2, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.0",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 4],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[0, 4],
                    sizes=[2, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.1",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 4],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[2, 0],
                    sizes=[2, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.2",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 4],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[2, 4],
                    sizes=[2, 4],
                    tensor=TensorEntry(
                        location="sharded/foo/qux.3",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 4],
                        replicated=False,
                    ),
                ),
            ]
        )
    if "foo/corge" in local_manifest:
        merged_local_manifest["foo/corge"] = DTensorEntry(
            shards=[
                Shard(
                    offsets=[0, 0],
                    sizes=[5, 5],
                    tensor=TensorEntry(
                        location="sharded/foo/corge.0",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[5, 5],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[0, 5],
                    sizes=[5, 3],
                    tensor=TensorEntry(
                        location="sharded/foo/corge.1",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[5, 3],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[5, 0],
                    sizes=[2, 5],
                    tensor=TensorEntry(
                        location="sharded/foo/corge.2",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 5],
                        replicated=False,
                    ),
                ),
                Shard(
                    offsets=[5, 5],
                    sizes=[2, 3],
                    tensor=TensorEntry(
                        location="sharded/foo/corge.3",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[2, 3],
                        replicated=False,
                    ),
                ),
            ],
            mesh=[[0, 1], [2, 3]],
            dim_map=[[0], [1]],
        )
    if "foo/grault" in local_manifest:
        merged_local_manifest["foo/grault"] = DTensorEntry(
            shards=[
                Shard(
                    offsets=[0, 0],
                    sizes=[7, 5],
                    tensor=TensorEntry(
                        location="replicated_sharded/foo/grault.0",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[7, 5],
                        replicated=True,
                    ),
                ),
                Shard(
                    offsets=[0, 5],
                    sizes=[7, 3],
                    tensor=TensorEntry(
                        location="replicated_sharded/foo/grault.2",
                        serializer="torch_save",
                        dtype="float32",
                        shape=[7, 3],
                        replicated=True,
                    ),
                ),
            ],
            mesh=[[0, 1], [2, 3]],
            dim_map=[[-1], [0]],
        )
    return merged_local_manifest
