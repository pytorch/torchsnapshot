#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import asyncio
import copy
import io
import math
import os
import sys
from concurrent.futures import Executor
from functools import reduce
from operator import mul
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec

from .io_types import BufferConsumer, BufferStager, BufferType, ReadReq, WriteReq
from .manifest import Entry, ObjectEntry, Shard, ShardedTensorEntry, TensorEntry
from .serialization import (
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_element_size,
    dtype_to_string,
    Serializer,
    string_to_dtype,
    tensor_as_memoryview,
    tensor_from_memoryview,
    torch_load_from_bytes,
    torch_save_as_bytes,
)
from .torch_dist_checkpoint.metadata import (
    ExtendedTensorMetadata,
    StorageMetadata,
    TensorReadRequest,
)
from .torch_dist_checkpoint.resharding import prepare_sharded_tensor_read


class ShardedTensorIOPreparer:
    DEFAULT_MAX_SHARD_SIZE_BYTES: int = 512 * 1024 * 1024

    @staticmethod
    def subdivide_shard(
        shard: torch.Tensor,
        offsets: List[int],
        sizes: List[int],
        dim: int,
        max_shard_sz_bytes: int,
    ) -> List[Tuple[torch.Tensor, List[int], List[int]]]:
        """
        Subdivide the shard along the sharding dim.
        """
        if max_shard_sz_bytes <= 0:
            raise ValueError(
                f"max_shard_sz_bytes must be a positive integer (got {max_shard_sz_bytes})."
            )
        slice_sz = (
            reduce(lambda x, y: x * y, sizes) // sizes[dim] * shard.element_size()
        )
        chunk_length = max(math.floor(max_shard_sz_bytes / slice_sz), 1)
        n_chunks = math.ceil(sizes[dim] / chunk_length)

        subdivided = []
        for i in range(n_chunks):
            start = i * chunk_length
            length = min((i + 1) * chunk_length, sizes[dim]) - i * chunk_length

            sub_offsets = copy.deepcopy(offsets)
            sub_offsets[dim] += start
            sub_sizes = copy.deepcopy(sizes)
            sub_sizes[dim] = length
            sub_view = torch.narrow(shard, dim, start, length)
            subdivided.append((sub_view, sub_offsets, sub_sizes))
        return subdivided

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        obj: ShardedTensor,
    ) -> Tuple[ShardedTensorEntry, List[WriteReq]]:
        shards = []
        write_reqs = []
        for shard in obj.local_shards():
            sharding_spec = obj.sharding_spec()
            if isinstance(sharding_spec, ChunkShardingSpec):
                sharding_dim = sharding_spec.dim
            else:
                sharding_dim = 0

            subdivided = cls.subdivide_shard(
                shard=shard.tensor,
                offsets=shard.metadata.shard_offsets,
                sizes=shard.metadata.shard_sizes,
                dim=sharding_dim,
                max_shard_sz_bytes=cls.DEFAULT_MAX_SHARD_SIZE_BYTES,
            )

            for tensor, offsets, sizes in subdivided:
                suffix = "_".join(str(i) for i in offsets)
                entry, tensor_write_reqs = TensorIOPreparer.prepare_write(
                    storage_path=f"{storage_path}_{suffix}", tensor=tensor
                )
                write_reqs += tensor_write_reqs

                shards.append(
                    Shard(
                        offsets=offsets,
                        sizes=sizes,
                        tensor=entry,
                    )
                )
        return ShardedTensorEntry(shards=shards), write_reqs

    @classmethod
    def prepare_read(
        cls,
        entry: ShardedTensorEntry,
        obj_out: Optional[ShardedTensor] = None,
    ) -> List[ReadReq]:
        if obj_out is None:
            # TODO: support loading sharded tensor without obj_out
            raise RuntimeError(
                "Reading a ShardedTensor without a runtime object is not yet supported."
            )

        metadata = ExtendedTensorMetadata(
            tensor_metadata=ShardedTensorMetadata(),  # Unused for load
            storage_metadata=[],
        )
        for shard in entry.shards:
            metadata.storage_metadata.append(
                StorageMetadata(
                    shard_metadata=ShardMetadata(
                        shard_offsets=shard.offsets,
                        shard_sizes=shard.sizes,
                        placement="cpu",  # Unused for load
                    ),
                    storage_key=shard.tensor.location,
                    length=0,  # Unused for load
                    offset=0,  # Unused for load
                )
            )
        tensor_read_reqs = prepare_sharded_tensor_read(
            metadata=metadata, sharded_tensor_out=obj_out
        )
        locations_to_load = {twr.storage_key for twr in tensor_read_reqs}

        read_reqs = []
        for shard in entry.shards:
            if shard.tensor.location not in locations_to_load:
                continue
            read_reqs.append(
                ReadReq(
                    path=shard.tensor.location,
                    buffer_consumer=ShardedTensorBufferConsumer(
                        tensor_read_reqs=[
                            trr
                            for trr in tensor_read_reqs
                            if trr.storage_key == shard.tensor.location
                        ],
                        entry=shard.tensor,
                    ),
                )
            )
        return read_reqs


@torch.jit.script
def tensor_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.copy_(src)


class ShardedTensorBufferConsumer(BufferConsumer):
    def __init__(
        self,
        tensor_read_reqs: List[TensorReadRequest],
        entry: TensorEntry,
    ) -> None:
        self.tensor_read_reqs = tensor_read_reqs
        self.entry = entry

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        view_to_copy = TensorBufferConsumer.deserialize_tensor(
            buf=buf, entry=self.entry
        )
        for req in self.tensor_read_reqs:
            for dim, (start, length) in enumerate(zip(req.offsets, req.lengths)):
                view_to_copy = torch.narrow(view_to_copy, dim, start, length)

            assert (
                view_to_copy.size() == req.tensor.size()
            ), f"The {req.storage_key} src/dst size does not match."

            if executor is not None:
                await asyncio.get_running_loop().run_in_executor(
                    executor, tensor_copy, req.tensor, view_to_copy
                )
            else:
                req.tensor.copy_(view_to_copy)

    def get_consuming_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")


@torch.jit.script
def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cpu")


class TensorBufferStager(BufferStager):
    def __init__(self, tensor: torch.Tensor, entry: TensorEntry) -> None:
        self.tensor = tensor
        self.entry = entry

    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        if self.tensor.is_cuda:
            # It would be nice to copy from GPU via DMA. However, it is very
            # difficult to figure out the safe amount of page-locked memory
            # that we can use. For now, we'll resort to a thread pool for
            # concurrent DtoH copy (with GIL released).
            if executor is not None:
                cpu_tensor = await asyncio.get_running_loop().run_in_executor(
                    executor, tensor_to_cpu, self.tensor.detach()
                )
            else:
                cpu_tensor = self.tensor.detach().to("cpu")
        elif (
            self.tensor.nelement() != self.tensor.storage().size()
            or self.entry.serializer == Serializer.BUFFER_PROTOCOL
        ):
            # Avoid saving the entire storage when saving a view
            cpu_tensor = self.tensor.detach().clone()
        else:
            cpu_tensor = self.tensor.detach()

        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            return torch_save_as_bytes(cpu_tensor)
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_as_memoryview(cpu_tensor)
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")

    def get_staging_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")


class TensorBufferConsumer(BufferConsumer):
    def __init__(
        self,
        tensor: torch.Tensor,
        entry: TensorEntry,
    ) -> None:
        self.tensor = tensor
        self.entry = entry

    @staticmethod
    def deserialize_tensor(buf: bytes, entry: TensorEntry) -> torch.Tensor:
        if entry.serializer == Serializer.TORCH_SAVE.value:
            return torch_load_from_bytes(buf)
        elif entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            dtype = string_to_dtype(entry.dtype)
            return tensor_from_memoryview(
                memoryview(buf), dtype=dtype, shape=entry.shape
            )
        else:
            raise ValueError(f"Unrecognized serializer: {entry.serializer}.")

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        loaded = self.deserialize_tensor(buf=buf, entry=self.entry)
        if executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                executor, tensor_copy, self.tensor, loaded
            )
        else:
            self.tensor.copy_(loaded)

    def get_consuming_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")


class TensorIOPreparer:
    @staticmethod
    def prepare_write(
        storage_path: str, tensor: torch.Tensor
    ) -> Tuple[TensorEntry, List[WriteReq]]:
        if tensor.dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
            serializer = Serializer.BUFFER_PROTOCOL.value
        else:
            serializer = Serializer.TORCH_SAVE.value

        entry = TensorEntry(
            location=storage_path,
            serializer=serializer,
            dtype=dtype_to_string(tensor.dtype),
            shape=list(tensor.shape),
            replicated=False,
        )
        buffer_stager = TensorBufferStager(tensor=tensor, entry=entry)
        return entry, [WriteReq(path=storage_path, buffer_stager=buffer_stager)]

    @classmethod
    def prepare_read(
        cls,
        entry: TensorEntry,
        tensor_out: Optional[torch.Tensor] = None,
    ) -> List[ReadReq]:
        if tensor_out is None:
            raise RuntimeError(
                "Reading a Tensor without a runtime object is not yet supported."
            )
        buffer_consumer = TensorBufferConsumer(
            tensor=tensor_out,
            entry=entry,
        )
        return [ReadReq(path=entry.location, buffer_consumer=buffer_consumer)]

    @staticmethod
    def get_tensor_size_from_entry(entry: TensorEntry) -> int:
        dtype = string_to_dtype(entry.dtype)
        element_size = dtype_to_element_size(dtype)
        n_element = reduce(mul, entry.shape, 1)
        return element_size * n_element


class ObjectBufferStager(BufferStager):
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        buf = io.BytesIO()
        torch.save(self.obj, buf)
        return buf.getvalue()

    def get_staging_cost_bytes(self) -> int:
        # TODO: this is not accurate
        return sys.getsizeof(self.obj)


T = TypeVar("T")


class ObjectBufferConsumer(BufferConsumer, Generic[T]):
    def __init__(self, obj_out: T) -> None:
        self.consuming_cost_bytes: int = sys.getsizeof(obj_out)
        self.callback: Optional[Callable[[T], None]] = None

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        obj: T = torch.load(io.BytesIO(buf))
        if self.callback is not None:
            self.callback(obj)

    def get_consuming_cost_bytes(self) -> int:
        return self.consuming_cost_bytes

    def set_consume_callback(self, callback: Callable[[T], None]) -> None:
        self.callback = callback


class ObjectIOPreparer(Generic[T]):
    @staticmethod
    def prepare_write(
        storage_path: str,
        obj: T,
    ) -> Tuple[ObjectEntry, List[WriteReq]]:
        buffer_stager = ObjectBufferStager(obj=obj)
        return (
            ObjectEntry(
                location=storage_path,
                serializer=Serializer.TORCH_SAVE.value,
                obj_type=type(obj).__module__ + "." + type(obj).__name__,
                replicated=False,
            ),
            [WriteReq(path=storage_path, buffer_stager=buffer_stager)],
        )

    @classmethod
    def prepare_read(cls, entry: ObjectEntry, obj_out: T) -> List[ReadReq]:
        buffer_consumer = ObjectBufferConsumer(obj_out=obj_out)
        return [
            ReadReq(
                path=entry.location,
                buffer_consumer=buffer_consumer,
            )
        ]


def get_storage_path(obj: Any, logical_path: str, rank: int, replicated: bool) -> str:
    if isinstance(obj, ShardedTensor):
        return os.path.join("sharded", logical_path)
    elif replicated:
        return os.path.join("replicated", logical_path)
    else:
        return os.path.join(str(rank), logical_path)


def prepare_write(
    obj: Any,
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[Entry, List[WriteReq]]:
    """
    Prepare write for an object.

    Args:
        obj: The object to save.
        logical_path: The logical path of the object.
        rank: The rank of the current process.
        replicated: Whether the object is replicated.

    Returns:
        The class::`Entry` describing the object, and a list of
        class::`WriteReq` for persisting the object.
    """
    storage_path = get_storage_path(obj, logical_path, rank, replicated)
    if isinstance(obj, ShardedTensor):
        return ShardedTensorIOPreparer.prepare_write(storage_path, obj)
    elif isinstance(obj, torch.Tensor):
        entry, obj_write_req = TensorIOPreparer.prepare_write(storage_path, obj)
    else:
        entry, obj_write_req = ObjectIOPreparer.prepare_write(storage_path, obj)

    entry.replicated = replicated
    return entry, obj_write_req


def prepare_read(entry: Entry, obj_out: Optional[Any] = None) -> List[ReadReq]:
    """
    Prepare read for an object.

    Args:
        entry: The entry describing the object.
        obj: The object to load.

    Returns:
        A list of class::`ReadReq` for loading the object.
    """
    if isinstance(entry, ShardedTensorEntry):
        if obj_out is None:
            # TODO: support loading sharded tensor without obj_out
            raise RuntimeError(
                "Reading a ShardedTensor without a runtime object is not yet supported."
            )
        return ShardedTensorIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, TensorEntry):
        return TensorIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, ObjectEntry):
        return ObjectIOPreparer.prepare_read(entry, obj_out)
    else:
        raise Exception(f"Unsupported entry type: {entry} ({entry.type}).")
