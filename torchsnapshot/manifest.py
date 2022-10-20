#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import base64
import logging
import struct
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import yaml

try:
    from yaml import CSafeDumper as Dumper, CSafeLoader as Loader
except ImportError:
    from yaml import SafeDumper as Dumper, SafeLoader as Loader

logger: logging.Logger = logging.getLogger(__name__)


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
    byte_range: Optional[List[int]]

    def __init__(
        self,
        location: str,
        serializer: str,
        dtype: str,
        shape: List[int],
        replicated: bool,
        byte_range: Optional[List[int]] = None,
    ) -> None:
        super().__init__(type="Tensor")
        self.location = location
        self.serializer = serializer
        self.dtype = dtype
        self.shape = shape
        self.replicated = replicated
        self.byte_range = byte_range

    @property
    def byte_range_tuple(self) -> Optional[Tuple[int, int]]:
        byte_range = self.byte_range
        if byte_range is None:
            return None
        else:
            return (byte_range[0], byte_range[1])


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

    @classmethod
    def from_yaml(cls, entry: Any) -> "ShardedTensorEntry":
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
                    byte_range=shard["tensor"].get("byte_range"),
                ),
            )
            for shard in entry["shards"]
        ]
        return cls(shards=shards)


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

    @classmethod
    def from_yaml(cls, entry: Any) -> "ChunkedTensorEntry":
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
                    byte_range=chunk["tensor"].get("byte_range"),
                ),
            )
            for chunk in entry["chunks"]
        ]
        shape = entry["shape"]
        return cls(
            dtype=dtype,
            shape=shape,
            chunks=chunks,
            replicated=replicated,
        )


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
    replicated: bool
    readable: Optional[str]

    def __init__(
        self,
        type: str,
        serialized_value: str,
        replicated: bool,
        readable_value: Optional[str] = None,
    ) -> None:
        super().__init__(type=type)
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
        if type_name not in cls.supported_types():
            raise TypeError(f"Unsupported primitive obj of type {type_name}")

        serialized_value = cls._serialize(type_name, obj)
        readable_value = str(obj) if type_name == "float" else None
        return PrimitiveEntry(type_name, serialized_value, False, readable_value)

    @classmethod
    def from_serialized(
        cls,
        type_name: str,
        serialized_value: str,
        replicated: bool,
        readable: Optional[str],
    ) -> "PrimitiveEntry":
        if type_name not in cls.supported_types():
            raise TypeError(f"Unsupported primitive obj of type {type_name}")

        return PrimitiveEntry(type_name, serialized_value, replicated)


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
                manifest[path] = ShardedTensorEntry.from_yaml(entry=entry)
            elif type_name == "ChunkedTensor":
                manifest[path] = ChunkedTensorEntry.from_yaml(entry=entry)
            elif type_name == "object":
                manifest[path] = ObjectEntry(**entry)
        d["manifest"] = manifest
        return cls(**d)


def get_manifest_for_rank(metadata: SnapshotMetadata, rank: int) -> Manifest:
    """
    Prepare available entries to load from for the rank.

    Given a global manifest, prepare available entries to load from for the
    rank according to the following rules:

        per-rank: The entry is only made available to the rank saved it.
        replicated: The entry is made available to all ranks.
        sharded: Entries are first merged across all ranks then made available
            to all ranks.

    Args:
        manifest: The global manifest.
        rank: The rank of the current process.

    Returns:
        The local manifest for the rank.
    """
    rank_to_manifest: Dict[int, Dict[str, Entry]] = {
        i: {} for i in range(metadata.world_size)
    }

    for path, entry in metadata.manifest.items():
        tokens = path.split("/")
        rnk = int(tokens.pop(0))
        logical_path = "/".join(tokens)
        rank_to_manifest[rnk][logical_path] = entry

    local_manifest = rank_to_manifest.get(rank, {})

    _copy_replicated_entries(
        dst_manifest=local_manifest, src_manifest=rank_to_manifest[0]
    )
    for rnk, manifest in rank_to_manifest.items():
        if rnk == rank:
            continue
        _copy_sharded_tensor_entries(dst_manifest=local_manifest, src_manifest=manifest)
    return local_manifest


def _copy_replicated_entries(dst_manifest: Manifest, src_manifest: Manifest) -> None:
    for logical_path, entry in src_manifest.items():
        if is_replicated(entry) and logical_path not in dst_manifest:
            _copy_entry(dst_manifest, src_manifest, logical_path)


def _copy_sharded_tensor_entries(
    dst_manifest: Manifest, src_manifest: Manifest
) -> None:
    for logical_path, entry in src_manifest.items():
        if not isinstance(entry, ShardedTensorEntry):
            continue
        if logical_path not in dst_manifest:
            _copy_entry(dst_manifest, src_manifest, logical_path)
        else:
            dst_manifest[logical_path] = ShardedTensorEntry(
                shards=sorted(
                    dst_manifest[logical_path].shards + entry.shards,
                    key=lambda s: s.offsets,
                )
            )


def _copy_entry(
    dst_manifest: Manifest, src_manifest: Manifest, logical_path: str
) -> None:
    if logical_path in dst_manifest:
        return

    dst_manifest[logical_path] = src_manifest[logical_path]

    tokens = logical_path.split("/")
    key = tokens.pop()
    path = "/".join(tokens)
    while path not in dst_manifest:
        entry = src_manifest[path]
        if is_dict_entry(entry):
            entry.keys = [key]
        dst_manifest[path] = entry
        key = tokens.pop()
        path = "/".join(tokens)
        if len(tokens) == 0:
            break

    if path in dst_manifest and is_dict_entry(dst_manifest[path]):
        dst_manifest[path].keys.append(key)


def is_dict_entry(entry: Entry) -> bool:
    return isinstance(entry, (DictEntry, OrderedDictEntry))


def is_replicated(entry: Entry) -> bool:
    if not hasattr(entry, "replicated"):
        return False
    # pyre-ignore
    return entry.replicated


def is_container_entry(entry: Entry) -> bool:
    return isinstance(entry, (ListEntry, DictEntry, OrderedDictEntry))
