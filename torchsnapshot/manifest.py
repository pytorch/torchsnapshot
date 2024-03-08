#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
    """
    Entry dataclass for a torch.Tensor object.

    Attributes:
        location (str): the file storage location of the object in the saved snapshot
        serializer (str): the function that serialized the object (usually torch_save)
        dtype (str): torch datatype of the tensor
        shape (List[int]): shape of the tensor
        replicated (bool): whether the tensor is replicated on other ranks
        byte_range (Optional[List[int]]): indexable range of bytes in a continguous block of memory
    """

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
    """
    An individual shard dataclass for a ShardedTensorEntry.

    Attributes:
        offsets (List[int]): starting indices by dimension of the shard within the original tensor,
            should match n dims of tensor
        sizes (List[int]): shape of the shard, should be identical to shape of underlying TensorEntry
        tensor (TensorEntry): entry of actual tensor object in shard
    """

    offsets: List[int]
    sizes: List[int]
    tensor: TensorEntry

    @classmethod
    def from_yaml_obj(cls, yaml_obj: Any) -> "Shard":
        yaml_obj["tensor"] = TensorEntry.from_yaml_obj(yaml_obj["tensor"])
        return cls(**yaml_obj)


@dataclass
class ShardedTensorEntry(Entry):
    """
    Entry for overall ShardedTensor that can contain multiple shards.

    Attributes:
        shards (List[Shard]): list of individual Shard entries for each shard
    """

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

    def get_tensor_shape(self) -> List[int]:
        """
        Computes the shape of the entire tensor.

        Returns:
            List[int]: shape of the entire tensor

        .. note::
            The shape can be computed by finding the maximum (size + offset sum)
            tuple in all of the shards. The shard's size/offset are equal or
            increasing in each dimension as the shards progress in the list.
        """
        assert len(self.shards) > 0, "No shards found."

        first_shard = self.shards[0]
        shape = [
            size + offset
            for size, offset in zip(first_shard.sizes, first_shard.offsets)
        ]
        for shard in self.shards[1:]:
            sizes = shard.sizes
            offsets = shard.offsets
            # sum element-wise
            candidate_shape = [size + offset for size, offset in zip(sizes, offsets)]
            if all(x >= y for x, y in zip(candidate_shape, shape)):
                shape = candidate_shape
        return shape


@dataclass
class ChunkedTensorEntry(Entry):
    """
    Entry for Tensor object that has been chunked when saving for performance optimization.

    Attributes:
        dtype (str): dtype of all tensor chunks
        shape (List[int]): shape of overall unchunked tensor
        chunks (List[Shard]): individual entries for each chunk, represented as a Shard dataclass
        replicated (bool): whether the overall tensor is replicated across ranks
    """

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


# Define a type for n-dimensional list to represent n-dimensional device mesh of ranks
NestedList = Union[int, List["NestedList"]]


@dataclass
class DTensorEntry(Entry):
    """
    Entry class for recording a DTensor object in the manifest when saving a snapshot.

    This class should contain all information needed to save/load a DTensor object.

    DTensor can be sharded similarly to ShardedTensor. Unlike ShardedTensor, it can also be replicated
    across ranks, and can be BOTH sharded and replicated. In this combined case, individual shards may be replicated
    across different ranks. For this reason, we let the underlying TensorEntrys track replicated tensors.

    Attributes:
        shards (List[Shard]): list of shards the DTensor comprises of. If not sharded, then this is a single shard.
        mesh (NestedList): n-dimensional list representing the device mesh of the DTensor.
        dim_map (List[List[int]]): list indicating how each tensor dim is sharded or replicated on each device mesh dim.
            This is used as a condensed representation of the DTensor's Placements.

            Each element in the list indexed by i describes tensor dim i.
                - If tensor dim i is sharded, the list element will be a list of integers indicating the device mesh dims
                  tensor dim i is sharded on. The same tensor dim can be sharded across multiple device
                  mesh dims (e.g., [Shard(0), Shard(0)]).
                - If tensor dim i is replicated, the list element will be [-1].

            Example: If dim_map was [[0, 1], [-1]], this means the first tensor dim is sharded twice across device mesh dim 0
                     and device mesh dim 1, while the second tensor dim is replicated across all device mesh dims.

    """

    shards: List[Shard]
    mesh: NestedList
    dim_map: List[List[int]]

    def __init__(
        self,
        shards: List[Shard],
        mesh: NestedList,
        dim_map: List[List[int]],
    ) -> None:
        super().__init__(type="DTensor")
        self.shards = shards
        self.mesh = mesh
        self.dim_map = dim_map

    @classmethod
    def from_yaml_obj(cls, yaml_obj: Any) -> "DTensorEntry":
        if "type" in yaml_obj:
            del yaml_obj["type"]
        yaml_obj["shards"] = [
            Shard.from_yaml_obj(shard) for shard in yaml_obj["shards"]
        ]
        return cls(**yaml_obj)


@dataclass
class ObjectEntry(Entry):
    """
    Entry for a generic object.

    Attributes:
        location (str): the file storage location of the object in the saved snapshot
        serializer (str): the function that serialized the object (usually torch_save)
        obj_type (str): name of the object class
        replicated (bool): whether the object is replicated on other ranks
    """

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
    """
    Entry for a Python list primitive.
    """

    def __init__(self) -> None:
        super().__init__(type="list")


@dataclass
class DictEntry(Entry):
    """
    Entry for a Python dict primitive.
    """

    keys: List[Union[str, int]]

    def __init__(self, keys: List[Union[str, int]]) -> None:
        super().__init__(type="dict")
        self.keys = keys


@dataclass
class OrderedDictEntry(Entry):
    """
    Entry for a Python OrderedDict primitive.
    """

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
    """
    Overall manifest object that contains all the entries in a snapshot and
    the associated metadata. Converts all yaml objects into their appropriate
    Entry dataclass.

    Attributes:
        version (str): version of TorchSnapshot package
        world_size (str): total number of ranks in distributed environment where snapshot was taken
        manifest (Manifest): Manifest object containing all the entries in the snapshot
    """

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
            elif type_name == "DTensor":
                manifest[path] = DTensorEntry.from_yaml_obj(yaml_obj)
            elif type_name == "object":
                manifest[path] = ObjectEntry.from_yaml_obj(yaml_obj)
        d["manifest"] = manifest
        return cls(**d)
