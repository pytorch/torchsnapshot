#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import base64
import json
import logging
import struct
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypeVar, Union

import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

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

    @classmethod
    def from_yaml_obj(cls, yaml_obj: Any) -> "Entry":
        if "type" in yaml_obj:
            del yaml_obj["type"]
        return cls(**yaml_obj)


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

    @classmethod
    def from_yaml_obj(cls, yaml_obj: Any) -> "Shard":
        yaml_obj["tensor"] = TensorEntry.from_yaml_obj(yaml_obj["tensor"])
        return cls(**yaml_obj)


@dataclass
class ShardedTensorEntry(Entry):
    shards: List[Shard]

    def __init__(self, shards: List[Shard]) -> None:
        super().__init__(type="ShardedTensor")
        self.shards = shards

    @classmethod
    def from_yaml_obj(cls, yaml_obj: Any) -> "ShardedTensorEntry":
        if "type" in yaml_obj:
            del yaml_obj["type"]
        yaml_obj["shards"] = [
            Shard.from_yaml_obj(shard) for shard in yaml_obj["shards"]
        ]
        return cls(**yaml_obj)


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
    def from_yaml_obj(cls, yaml_obj: Any) -> "ChunkedTensorEntry":
        if "type" in yaml_obj:
            del yaml_obj["type"]
        yaml_obj["chunks"] = [
            Shard.from_yaml_obj(shard) for shard in yaml_obj["chunks"]
        ]
        return cls(**yaml_obj)


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
    keys: List[Union[str, int]]

    def __init__(self, keys: List[Union[str, int]]) -> None:
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

    supported_types: ClassVar[List[str]] = [t.value for t in PrimitiveType]

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
        if type_name not in cls.supported_types:
            raise TypeError(f"Unsupported primitive obj of type {type_name}")

        serialized_value = cls._serialize(type_name, obj)
        readable_value = str(obj) if type_name == "float" else None
        return PrimitiveEntry(type_name, serialized_value, False, readable_value)

    @classmethod
    def from_yaml_obj(
        cls,
        yaml_obj: Any,
    ) -> "PrimitiveEntry":
        type_name = yaml_obj["type"]
        if type_name not in cls.supported_types:
            raise TypeError(f"Unsupported primitive obj of type {type_name}")
        del yaml_obj["readable"]
        return cls(**yaml_obj)


T = TypeVar("T", bound=Entry)
Manifest = Dict[str, T]


@dataclass
class SnapshotMetadata:
    version: str
    world_size: int
    manifest: Manifest

    def to_yaml(self) -> str:
        # When the number of entries in the snapshot metadata is large, yaml
        # serialization becomes slow and there's little room for optimization.
        # Since the snapshot metadata can be dumped as json and json is a
        # subset of yaml, using json.dumps() here to help with the
        # serialization performance without needing to deprecate yaml.
        return json.dumps(asdict(self), sort_keys=False, indent=2)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SnapshotMetadata":
        d = yaml.load(yaml_str, Loader=Loader)
        manifest: Manifest = {}
        for path, yaml_obj in d["manifest"].items():
            type_name = yaml_obj["type"]
            if type_name == "list":
                manifest[path] = ListEntry.from_yaml_obj(yaml_obj)
            elif type_name == "dict":
                manifest[path] = DictEntry.from_yaml_obj(yaml_obj)
            elif type_name == "OrderedDict":
                manifest[path] = OrderedDictEntry.from_yaml_obj(yaml_obj)
            elif type_name in PrimitiveEntry.supported_types:
                manifest[path] = PrimitiveEntry.from_yaml_obj(yaml_obj)
            elif type_name == "Tensor":
                manifest[path] = TensorEntry.from_yaml_obj(yaml_obj)
            elif type_name == "ShardedTensor":
                manifest[path] = ShardedTensorEntry.from_yaml_obj(yaml_obj)
            elif type_name == "ChunkedTensor":
                manifest[path] = ChunkedTensorEntry.from_yaml_obj(yaml_obj)
            elif type_name == "object":
                manifest[path] = ObjectEntry.from_yaml_obj(yaml_obj)
        d["manifest"] = manifest
        return cls(**d)


def is_dict_entry(entry: Entry) -> bool:
    return isinstance(entry, (DictEntry, OrderedDictEntry))


def is_replicated(entry: Entry) -> bool:
    if not hasattr(entry, "replicated"):
        return False
    # pyre-ignore
    return entry.replicated


def is_container_entry(entry: Entry) -> bool:
    return isinstance(entry, (ListEntry, DictEntry, OrderedDictEntry))
