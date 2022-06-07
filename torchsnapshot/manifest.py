#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Dict, List

import yaml


@dataclass
class Entry:
    type: str


@dataclass
class ObjectEntry(Entry):
    location: str
    replicated: bool


@dataclass
class Shard:
    offsets: List[int]
    sizes: List[int]
    location: str


@dataclass
class ShardedTensorEntry(Entry):
    shards: List[Shard]

    def __init__(self, shards: List[Shard]) -> None:
        super().__init__(type="ShardedTensor")
        self.shards = shards


@dataclass
class ListEntry(Entry):
    def __init__(self) -> None:
        super().__init__(type="list")


@dataclass
class DictEntry(Entry):
    keys: List[str]

    def __init__(self, keys: List[str]) -> None:
        super().__init__(type="dict")
        self.keys = keys


@dataclass
class OrderedDictEntry(Entry):
    keys: List[str]

    def __init__(self, keys: List[str]) -> None:
        super().__init__(type="OrderedDict")
        self.keys = keys


Manifest = Dict[str, Entry]


@dataclass
class SnapshotMetadata:
    version: str
    world_size: int
    manifest: Manifest

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SnapshotMetadata":
        d = yaml.safe_load(yaml_str)
        manifest: Manifest = {}
        for path, entry in d["manifest"].items():
            type_name = entry["type"]
            del entry["type"]
            if type_name == "list":
                manifest[path] = ListEntry(**entry)
            elif type_name == "dict":
                manifest[path] = DictEntry(**entry)
            elif type_name == "OrderedDict":
                manifest[path] = OrderedDictEntry(**entry)
            elif type_name == "ShardedTensor":
                shards = [Shard(**shard) for shard in entry["shards"]]
                manifest[path] = ShardedTensorEntry(shards=shards)
            else:
                manifest[path] = ObjectEntry(type=type_name, **entry)
        d["manifest"] = manifest
        return cls(**d)


def get_available_entries(manifest: Manifest, rank: int) -> Manifest:
    """
    Prepare available entries to load from for the rank.

    Given a global manifest, prepare available entries to load from for the
    rank according to the following rules:

        per-rank: The entry is only made available to the rank saved it.
        replicated: The entry is made available to all ranks.
        sharded: Entries are first merged across all ranks then made available
            to all ranks.

    The function will not return any container entries which are only used to
    reconstruct the orginal state dict.

    Args:
        manifest: The global manifest.
        rank: The rank of the current process.

    Returns:
        The local manifest for the rank.
    """
    grouped = {}
    for path, entry in manifest.items():
        tokens = path.split("/")[0]
        entry_rank = int(tokens[0])
        local_path = "/".join(path.split("/")[1:])
        if local_path not in grouped:
            grouped[local_path] = {}
        grouped[local_path][entry_rank] = entry

    local_manifest = {}
    for local_path, group in grouped.items():
        entries = list(group.values())

        # If the entry is sharded, make all shards available to all ranks.
        if isinstance(entries[0], ShardedTensorEntry):
            local_manifest[local_path] = ShardedTensorEntry(
                shards=[shard for entry in entries for shard in entry.shards]
            )
        elif isinstance(entries[0], ObjectEntry):
            if rank in group:
                local_manifest[local_path] = group[rank]
            # The current rank did not save the entry. Only make the entry
            # available to the rank if the entry is replicated.
            elif entries[0].replicated:
                local_manifest[local_path] = entries[0]
        elif isinstance(entries[0], (ListEntry, DictEntry, OrderedDictEntry)):
            # Container entries are only used for reconstructing the original
            # state dicts.
            pass
        else:
            raise RuntimeError(
                f"Unknown entry type: {type(entries[0])} ({entries[0].type})."
            )

    return local_manifest


def is_replicated(entry: Entry) -> bool:
    return isinstance(entry, ObjectEntry) and entry.replicated
