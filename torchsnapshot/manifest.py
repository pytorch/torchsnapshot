#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import base64
import struct
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Union

import yaml

try:
    from yaml import CSafeDumper as Dumper, CSafeLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


@dataclass
class Entry:
    """
    An entry describes a persisted object/collection.

    In yaml, entries are tagged unions consisted of primitive yaml types.
    For backward compatibility purposes, only the yaml representation is
    considered. The Python dataclasses are only for type checking.
    """

    type: str


@dataclass
class TensorEntry(Entry):
    location: str
    serializer: str
    dtype: str
    shape: List[int]
    replicated: bool

    def __init__(
        self,
        location: str,
        serializer: str,
        dtype: str,
        shape: List[int],
        replicated: bool,
    ) -> None:
        super().__init__(type="Tensor")
        self.location = location
        self.serializer = serializer
        self.dtype = dtype
        self.shape = shape
        self.replicated = replicated


@dataclass
class Shard:
    offsets: List[int]
    sizes: List[int]
    tensor: TensorEntry


@dataclass
class ShardedTensorEntry(Entry):
    shards: List[Shard]

    def __init__(self, shards: List[Shard]) -> None:
        super().__init__(type="ShardedTensor")
        self.shards = shards


@dataclass
class ChunkedTensorEntry(Entry):
    dtype: str
    shape: List[int]
    chunks: List[Shard]
    replicated: bool

    def __init__(
        self, dtype: str, shape: List[int], chunks: List[Shard], replicated: bool
    ) -> None:
        super().__init__(type="ChunkedTensor")
        self.dtype = dtype
        self.shape = shape
        self.chunks = chunks
        self.replicated = replicated


@dataclass
class ObjectEntry(Entry):
    location: str
    serializer: str
    obj_type: str
    replicated: bool

    def __init__(
        self, location: str, serializer: str, obj_type: str, replicated: bool
    ) -> None:
        super().__init__(type="object")
        self.location = location
        self.serializer = serializer
        self.obj_type = obj_type
        self.replicated = replicated


@dataclass
class ListEntry(Entry):
    def __init__(self) -> None:
        super().__init__(type="list")


@dataclass
class DictEntry(Entry):
    keys: List[Union[str, int]]

    def __init__(self, keys: List[Union[str, int]]) -> None:
        super().__init__(type="dict")
        self.keys = keys


@dataclass
class OrderedDictEntry(Entry):
    keys: List[str]

    def __init__(self, keys: List[str]) -> None:
        super().__init__(type="OrderedDict")
        self.keys = keys


class PrimitiveType(Enum):
    INT = "int"
    STR = "str"
    BOOL = "bool"
    BYTES = "bytes"
    FLOAT = "float"


@dataclass
class PrimitiveEntry(Entry):
    """
    An Entry for certain primitive types that will be stored inline in metadata

    type: name of builtin type.
    serialized_value: value of the builtin type in serialized format.
    readable: for ease of inspection for certain types
    """

    serialized_value: str
    readable: Optional[str]
    replicated: bool

    def __init__(
        self,
        primitive_type: PrimitiveType,
        serialized_value: str,
        replicated: bool,
        readable_value: Optional[str] = None,
    ) -> None:
        super().__init__(type=primitive_type.value)
        self.serialized_value = serialized_value
        self.replicated = replicated
        self.readable = readable_value

    def get_value(self) -> Union[int, str, bool, bytes, float]:
        if self.type == "int":
            return int(self.serialized_value)
        elif self.type == "str":
            return self.serialized_value
        elif self.type == "bool":
            if self.serialized_value not in ["True", "False"]:
                raise RuntimeError(
                    f"Unexpected serialized_value for bool type: {self.serialized_value}"
                )
            return self.serialized_value == "True"
        elif self.type == "bytes":
            return base64.b64decode(bytes(self.serialized_value, "utf-8"))
        elif self.type == "float":
            packed_bytes = base64.b64decode(bytes(self.serialized_value, "utf-8"))
            return struct.unpack("d", packed_bytes)[0]
        raise ValueError(
            f"Unable to get deserialized value for {self.serialized_value}"
        )

    @classmethod
    def supported_types(cls) -> List[str]:
        return [t.value for t in PrimitiveType]

    @classmethod
    def _serialize(cls, type_name: str, obj: Any) -> str:
        if type_name == "int":
            return str(obj)
        elif type_name == "str":
            return str(obj)
        elif type_name == "bool":
            return str(obj)
        elif type_name == "bytes":
            return base64.b64encode(obj).decode("utf-8")
        elif type_name == "float":
            packed_bytes = struct.pack("d", float(obj))
            return cls._serialize("bytes", packed_bytes)
        else:
            raise TypeError(f"Unsupported primitive obj of type {type_name}")

    @classmethod
    def from_object(cls, obj: Any) -> "PrimitiveEntry":
        type_name = type(obj).__name__
        try:
            type_enum = PrimitiveType(type_name)
            serialized_value = cls._serialize(type_name, obj)
            return PrimitiveEntry(type_enum, serialized_value, False)
        except ValueError:
            raise TypeError(f"Unsupported primitive obj of type {type_name}")

    @classmethod
    def from_serialized(
        cls,
        type_name: str,
        serialized_value: str,
        replicated: bool,
        readable: Optional[str],
    ) -> "PrimitiveEntry":
        try:
            type_enum = PrimitiveType(type_name)
            return PrimitiveEntry(type_enum, serialized_value, replicated)
        except ValueError:
            raise TypeError(f"Unsupported primitive obj of type {type_name}")


T = TypeVar("T", bound=Entry)
Manifest = Dict[str, T]


@dataclass
class SnapshotMetadata:
    version: str
    world_size: int
    manifest: Manifest

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), sort_keys=False, Dumper=Dumper)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SnapshotMetadata":
        d = yaml.load(yaml_str, Loader=Loader)
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
            elif type_name in PrimitiveEntry.supported_types():
                manifest[path] = PrimitiveEntry.from_serialized(type_name, **entry)
            elif type_name == "Tensor":
                manifest[path] = TensorEntry(**entry)
            elif type_name == "ShardedTensor":
                shards = [
                    Shard(
                        offsets=shard["offsets"],
                        sizes=shard["sizes"],
                        tensor=TensorEntry(
                            location=shard["tensor"]["location"],
                            serializer=shard["tensor"]["serializer"],
                            dtype=shard["tensor"]["dtype"],
                            shape=shard["tensor"]["shape"],
                            replicated=shard["tensor"]["replicated"],
                        ),
                    )
                    for shard in entry["shards"]
                ]
                manifest[path] = ShardedTensorEntry(shards=shards)
            elif type_name == "ChunkedTensor":
                dtype = entry["dtype"]
                replicated = entry["replicated"]
                chunks = [
                    Shard(
                        offsets=chunk["offsets"],
                        sizes=chunk["sizes"],
                        tensor=TensorEntry(
                            location=chunk["tensor"]["location"],
                            serializer=chunk["tensor"]["serializer"],
                            dtype=chunk["tensor"]["dtype"],
                            shape=chunk["tensor"]["shape"],
                            replicated=chunk["tensor"]["replicated"],
                        ),
                    )
                    for chunk in entry["chunks"]
                ]
                shape = entry["shape"]
                manifest[path] = ChunkedTensorEntry(
                    dtype=dtype,
                    shape=shape,
                    chunks=chunks,
                    replicated=replicated,
                )

            elif type_name == "object":
                manifest[path] = ObjectEntry(**entry)
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
        elif isinstance(
            entries[0], (TensorEntry, ObjectEntry, ChunkedTensorEntry, PrimitiveEntry)
        ):
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
    return (
        isinstance(
            entry, (TensorEntry, ObjectEntry, ChunkedTensorEntry, PrimitiveEntry)
        )
        and entry.replicated
    )
