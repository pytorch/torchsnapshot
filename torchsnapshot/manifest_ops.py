#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Dict, List

from .manifest import (
    Entry,
    is_container_entry,
    is_dict_entry,
    is_replicated,
    Manifest,
    ShardedTensorEntry,
    SnapshotMetadata,
)


def get_manifest_for_rank(metadata: SnapshotMetadata, rank: int) -> Manifest:
    """
    Prepare the manifest for a rank according to the following rules:

        - Replicated entries are made available to all ranks.
        - Sharded entries are first merged across all ranks, then made available
              to all ranks.
        - Other entries are made available to a rank only if the rank produced
              the entry in the first place.

    Args:
        manifest: The global manifest.
        rank: The target rank.

    Returns:
        The local manifest for the rank.
    """
    rank_to_manifest: List[Dict[str, Entry]] = [{} for _ in range(metadata.world_size)]

    for path, entry in metadata.manifest.items():
        tokens = path.split("/")
        rnk = int(tokens.pop(0))
        logical_path = "/".join(tokens)
        rank_to_manifest[rnk][logical_path] = entry

    if rank < metadata.world_size:
        return _get_manifest_for_existing_rank(
            rank_to_manifest=rank_to_manifest, rank=rank
        )
    else:
        return _get_manifest_for_new_rank(rank_to_manifest=rank_to_manifest)


def _get_manifest_for_existing_rank(
    rank_to_manifest: List[Dict[str, Entry]], rank: int
) -> Manifest:
    local_manifest = copy.deepcopy(rank_to_manifest[rank])

    # Replicated entries are removed from the global manifest
    for logical_path, entry in rank_to_manifest[0].items():
        if is_replicated(entry):
            local_manifest[logical_path] = entry

    # Make all sharded tensor shards available to the local manifest
    for rnk, manifest in enumerate(rank_to_manifest):
        if rnk == rank:
            continue
        _copy_sharded_tensor_entries(
            dst_manifest=local_manifest,
            src_manifest=manifest,
        )
    return local_manifest


def _get_manifest_for_new_rank(rank_to_manifest: List[Dict[str, Entry]]) -> Manifest:
    # Use rank 0's manifest as the base
    local_manifest = copy.deepcopy(rank_to_manifest[0])

    # Remove non-replicated entries
    for logical_path in list(local_manifest.keys()):
        entry = local_manifest[logical_path]
        if (is_container_entry(entry) or is_replicated(entry)) or isinstance(
            entry, ShardedTensorEntry
        ):
            continue
        _remove_entry(manifest=local_manifest, logical_path=logical_path)

    # Make all sharded tensor shards available to the local manifest
    for manifest in rank_to_manifest[1:]:
        _copy_sharded_tensor_entries(
            dst_manifest=local_manifest,
            src_manifest=manifest,
        )
    return local_manifest


def _copy_sharded_tensor_entries(
    dst_manifest: Manifest, src_manifest: Manifest
) -> None:
    for logical_path, entry in src_manifest.items():
        if not isinstance(entry, ShardedTensorEntry):
            continue
        if logical_path not in dst_manifest:
            _insert_entry(dst_manifest, src_manifest, logical_path)
        else:
            dst_manifest[logical_path] = ShardedTensorEntry(
                shards=sorted(
                    dst_manifest[logical_path].shards + entry.shards,
                    key=lambda s: s.offsets,
                )
            )


def _insert_entry(
    dst_manifest: Manifest, src_manifest: Manifest, logical_path: str
) -> None:
    """
    Insert an entry from src_manifest and dst_manifest.

    Example:

        dst_manifest (before):
        {
            "foo": DictEntry(keys=["baz"]),
            "foo/baz: ...
        }

        src_manifest:
        {
            "foo": DictEntry(keys=["bar", "baz"]),
            "foo/bar": DictEntry(keys=["qux", "quux"]),
            "foo/bar/qux": ...
            "foo/bar/quux": ...
            "foo/baz": ...
        }

        logical_path: "foo/quux"

        dst_manifest (after):
        {
            "foo": DictEntry(keys=["baz", "bar"]),
            "foo/bar": DictEntry(keys=["quux"]),
            "foo/bar/quux": ...
            "foo/baz": ...
        }
    """
    if logical_path in dst_manifest:
        return

    dst_manifest[logical_path] = src_manifest[logical_path]

    # Find the first ancestor that exists and create missing
    # containers along the way.
    tokens = logical_path.split("/")
    anc_path = logical_path
    while True:
        key = tokens.pop()
        anc_path = "/".join(tokens)
        if len(anc_path) == 0 or anc_path in dst_manifest:
            break
        # anc_path must exist in a valid manifest
        container = copy.deepcopy(src_manifest[anc_path])
        if is_dict_entry(container):
            container.keys = [key]
        dst_manifest[anc_path] = container

    if anc_path not in dst_manifest:
        return

    if is_dict_entry(dst_manifest[anc_path]):
        dst_manifest[anc_path].keys.append(key)


def _remove_entry(manifest: Manifest, logical_path: str) -> None:
    """
    Remove an entry from a manifest.

    Example:

        manifest (before):
        {
            "foo": DictEntry(keys=["bar", "baz"]),
            "foo/bar": ...,
            "foo/baz": ...,
        }

        logical_path: "foo/bar",

        manifest (after):
        {
            "foo": DictEntry(keys=["baz"]),
            "foo/baz": ...,
        }
    """
    if logical_path not in manifest:
        return

    del manifest[logical_path]

    tokens = logical_path.split("/")
    key = tokens.pop()
    parent_path = "/".join(tokens)
    if len(parent_path) == 0:
        return

    parent = manifest[parent_path]
    if is_dict_entry(parent):
        if key in parent.keys:
            parent.keys.remove(key)
        else:
            parent.keys.remove(int(key))
