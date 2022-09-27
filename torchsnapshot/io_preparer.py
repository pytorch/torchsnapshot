#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import asyncio
import copy
import functools
import io
import itertools
import logging
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

import torch
from torch.distributed._shard.sharded_tensor import (
    Shard as ShardedTensorShard,
    ShardedTensor,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)

from .io_types import BufferConsumer, BufferStager, BufferType, ReadReq, WriteReq
from .manifest import (
    ChunkedTensorEntry,
    Entry,
    ObjectEntry,
    PrimitiveEntry,
    Shard,
    ShardedTensorEntry,
    TensorEntry,
)

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

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    offsets: List[int]
    sizes: List[int]
    dtype: str


class ChunkedTensorIOPreparer:
    DEFAULT_MAX_CHUNK_SIZE_BYTES: int = 512 * 1024 * 1024

    @staticmethod
    def chunk_tensor(
        tensor: torch.Tensor,
        chunking_dim: int = 0,
    ) -> List[Chunk]:
        # for 0-d case, reshape to 1-d
        if tensor.ndim == 0:
            tensor = tensor.view(-1)

        tensor_sz_bytes = tensor.numel() * tensor.element_size()
        n_chunks = math.ceil(
            tensor_sz_bytes / ChunkedTensorIOPreparer.DEFAULT_MAX_CHUNK_SIZE_BYTES
        )
        tensor_chunks = torch.chunk(tensor, chunks=n_chunks, dim=chunking_dim)

        curr_offsets = [0] * tensor.ndim
        chunking_instruction = []
        for i in range(len(tensor_chunks)):
            tensor_chunk_sizes = list(tensor_chunks[i].shape)
            chunking_instruction.append(
                Chunk(
                    offsets=curr_offsets[:],
                    sizes=tensor_chunk_sizes,
                    dtype=str(tensor.dtype),
                )
            )
            curr_offsets[chunking_dim] += tensor_chunk_sizes[chunking_dim]
        return chunking_instruction

    @staticmethod
    def _get_subtensor_view(
        tensor: torch.Tensor, chunk: Union[Shard, Chunk]
    ) -> torch.Tensor:
        # for 0-d case, reshape to 1-d
        result = tensor.view(-1) if tensor.ndim == 0 else tensor

        for d in range(len(chunk.sizes)):
            result = result.narrow(d, chunk.offsets[d], chunk.sizes[d])
        return result

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        tensor: torch.Tensor,
        chunking_instruction: List[Chunk],
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[ChunkedTensorEntry, List[WriteReq]]:
        write_reqs = []
        chunks = []
        for chunk in chunking_instruction:
            suffix = "_".join(str(x) for x in chunk.offsets)
            chunk_entry, chunk_write_reqs = TensorIOPreparer.prepare_write(
                f"{storage_path}_{suffix}",
                cls._get_subtensor_view(tensor, chunk),
            )
            chunks.append(
                Shard(offsets=chunk.offsets, sizes=chunk.sizes, tensor=chunk_entry)
            )
            write_reqs += chunk_write_reqs
        chunked_entry = ChunkedTensorEntry(
            dtype=dtype_to_string(tensor.dtype),
            shape=list(tensor.shape),
            chunks=chunks,
            replicated=False,
        )
        return chunked_entry, write_reqs

    @classmethod
    def prepare_read(
        cls,
        entry: ChunkedTensorEntry,
        tensor_out: Optional[torch.Tensor] = None,
        buffer_size_limit_bytes: Optional[int] = None,
    ) -> List[ReadReq]:
        if tensor_out is None:
            raise RuntimeError(
                "Reading a Tensor without a runtime object is not yet supported."
            )
        read_reqs = []
        for chunk in entry.chunks:
            tensor_out_chunk = cls._get_subtensor_view(tensor_out, chunk)
            chunk_read_reqs = TensorIOPreparer.prepare_read(
                chunk.tensor, tensor_out_chunk, buffer_size_limit_bytes
            )
            read_reqs += chunk_read_reqs
        return read_reqs


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
        slice_sz = reduce(mul, sizes) // sizes[dim] * shard.element_size()
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

    @staticmethod
    def _shards_get_overlap_region_wrt_saved_tensor(
        saved_shard: ShardMetadata, current_shard: ShardMetadata
    ) -> List[Tuple[int, int, int, int]]:
        """
        Return the overlapping region between saved_shard and current_shard.
        There returned list has the same number of elements as the tensor's dimension.
        For each element, we produce a tuple with the following contents:
            (dimension, `saved_shard` offset, `current_shard` offset, length)

        Offsets are relative to each shard.
        """
        # TODO: This is copied from
        # torch.distributed._shard.checkpoint.resharding._shards_get_overlap_region_wrt_saved_tensor
        # which is not in PyTorch 1.11 yet. Remove this method and import it
        # from PyTorch directly once we drop support for 1.11.
        narrows = []
        for dim, (
            saved_shard_offset,
            current_shard_offset,
            saved_shard_size,
            current_shard_size,
        ) in enumerate(
            zip(
                saved_shard.shard_offsets,
                current_shard.shard_offsets,
                saved_shard.shard_sizes,
                current_shard.shard_sizes,
            )
        ):
            min_range_end = min(
                saved_shard_offset + saved_shard_size,
                current_shard_offset + current_shard_size,
            )

            length = min_range_end - max(current_shard_offset, saved_shard_offset)

            if saved_shard_offset > current_shard_offset:
                offset_for_saved_tensor = 0
                offset_for_current_tensor = saved_shard_offset - current_shard_offset
            else:
                offset_for_saved_tensor = current_shard_offset - saved_shard_offset
                offset_for_current_tensor = 0

            narrows.append(
                (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
            )
        return narrows

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        obj: ShardedTensor,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
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
                    storage_path=f"{storage_path}_{suffix}",
                    tensor=tensor,
                    _tensor_prepare_func=_tensor_prepare_func,
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

    @staticmethod
    def _get_global_shape(entry: ShardedTensorEntry) -> List[int]:
        # TODO: the global dtype and shape should be tracked in the metadata
        global_shape = [0] * len(entry.shards[0].sizes)
        for shard in entry.shards:
            for dim in range(len(shard.offsets)):
                if shard.offsets[dim] + shard.sizes[dim] > global_shape[dim]:
                    global_shape[dim] = shard.offsets[dim] + shard.sizes[dim]
        return global_shape

    @staticmethod
    def _validate_shape(global_shape: List[int], obj_out: torch.Tensor) -> None:
        if isinstance(obj_out, ShardedTensor):
            out_shape = list(obj_out.metadata().size)
        else:
            out_shape = list(obj_out.shape)
        if out_shape != global_shape:
            logger.warn(
                "The shape of obj_out ({out_shape}) is different from the "
                "shape of the persisted sharded tensor ({global_shape}). "
                "Only the overlapping part will be loaded. "
            )

    @classmethod
    def prepare_read(
        cls,
        entry: ShardedTensorEntry,
        obj_out: Optional[torch.Tensor] = None,
    ) -> List[ReadReq]:
        if obj_out is None:
            # TODO: support loading sharded tensor without obj_out
            raise RuntimeError(
                "Reading a ShardedTensor without a runtime object is not yet supported."
            )

        global_shape = cls._get_global_shape(entry=entry)
        cls._validate_shape(global_shape=global_shape, obj_out=obj_out)

        if type(obj_out) == ShardedTensor:
            local_shards = obj_out.local_shards()
        elif type(obj_out) == torch.Tensor:
            local_shards = [
                ShardedTensorShard(
                    tensor=obj_out,
                    metadata=ShardMetadata(
                        shard_offsets=[0] * len(obj_out.shape),
                        shard_sizes=list(obj_out.shape),
                        placement="cpu",
                    ),
                )
            ]
        else:
            raise RuntimeError(
                f"obj_out must either be a Tensor or ShardedTensor (got {type(obj_out)})"
            )

        # For each persisted shard, find all its overlapping regions with the
        # local shards
        path_to_overlapping_regions = defaultdict(list)
        for local_shard, shard in itertools.product(local_shards, entry.shards):
            shard_md = ShardMetadata(
                shard_offsets=shard.offsets,
                shard_sizes=shard.sizes,
                placement="cpu",
            )
            if not _check_shard_metadata_pair_overlap(local_shard.metadata, shard_md):
                continue
            path_to_overlapping_regions[shard.tensor.location].append(
                _OverlappingRegion(
                    dst_tensor=local_shard.tensor,
                    overlap_region=cls._shards_get_overlap_region_wrt_saved_tensor(
                        saved_shard=shard_md,
                        current_shard=local_shard.metadata,
                    ),
                )
            )

        # Read each persisted shard once and use it to load all its overlapping
        # regions with the local shards
        read_reqs = []
        for shard in entry.shards:
            if shard.tensor.location not in path_to_overlapping_regions:
                continue
            read_reqs.append(
                ReadReq(
                    path=shard.tensor.location,
                    buffer_consumer=ShardedTensorBufferConsumer(
                        overlapping_regions=path_to_overlapping_regions[
                            shard.tensor.location
                        ],
                        entry=shard.tensor,
                    ),
                )
            )
        return read_reqs


@dataclass
class _OverlappingRegion:
    dst_tensor: torch.Tensor
    overlap_region: List[
        Tuple[int, int, int, int]
    ]  # (dim, src_offset, dst_offset, length)

    def get_views(self, src_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_view = src_tensor
        dst_view = self.dst_tensor
        for dim, src_offset, dst_offset, length in self.overlap_region:
            src_view = torch.narrow(src_view, dim, src_offset, length)
            dst_view = torch.narrow(dst_view, dim, dst_offset, length)
        return src_view, dst_view


def _q_params_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.qscheme() != rhs.qscheme():
        return False
    if lhs.qscheme() == torch.per_tensor_affine:
        return (
            lhs.q_scale() == rhs.q_scale() and lhs.q_zero_point() == rhs.q_zero_point()
        )
    elif lhs.qscheme() == torch.per_channel_affine:
        return torch.allclose(
            lhs.q_per_channel_scales(), lhs.q_per_channel_scales()
        ) and torch.allclose(
            lhs.q_per_channel_zero_points(), lhs.q_per_channel_zero_points()
        )
    else:
        raise RuntimeError(f"Unrecognized qscheme {lhs.qscheme()}")


@torch.jit.script
def _tensor_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.detach().copy_(src)  # pragma: no cover


@torch.jit.script
def _tensor_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.dequantize()  # pragma: no cover


def tensor_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    # When both dst and src are quantized tensors, .copy_() works as follows:
    # - src storage -> dst storage
    # - src qscheme -> dst qscheme
    #
    # We need to take special care when dst is a view of a larger tensor and
    # the quantization schemes of src and dst are different, because the new
    # qscheme will only be reflected on dest, which is a view object, but not
    # on the original tensor. Directly calling .copy_() in this case will cause
    # a region of the larger tensor's storage contain data that does not match
    # the larger tensor's qscheme.

    if src.is_quantized and (
        not dst.is_quantized  # Copying from quantized Tensor to non-quantized Tensor is not allowed
        or dst.qscheme() != src.qscheme()  # Quantized copy only works with same qscheme
        or dst.dtype != src.dtype  # Quantized copy requires matching dtypes
        or (dst._is_view() and not _q_params_equal(dst, src))  # See the top comment
    ):
        # TODO: tile the dequantize -> copy to reduce memory footprint
        src = _tensor_dequantize(src)
    _tensor_copy(dst, src)


class ShardedTensorBufferConsumer(BufferConsumer):
    def __init__(
        self,
        overlapping_regions: List[_OverlappingRegion],
        entry: TensorEntry,
    ) -> None:
        self.overlapping_regions = overlapping_regions
        self.entry = entry

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        deserialized = TensorBufferConsumer.deserialize_tensor(
            buf=buf, entry=self.entry
        )
        for overlapping_region in self.overlapping_regions:
            src_view, dst_view = overlapping_region.get_views(src_tensor=deserialized)
            if executor is not None:
                await asyncio.get_running_loop().run_in_executor(
                    executor, tensor_copy, dst_view, src_view
                )
            else:
                tensor_copy(dst_view, src_view)

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
    return tensor.to("cpu")  # pragma: no cover


class TensorBufferStager(BufferStager):
    def __init__(
        self,
        tensor: torch.Tensor,
        entry: TensorEntry,
        _tensor_prepare_func: Callable[[torch.Tensor, bool], torch.Tensor],
    ) -> None:
        self.tensor = tensor
        self.entry = entry
        self._tensor_prepare_func = _tensor_prepare_func

    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        # TODO: if the custom prepared tensor is different from the original
        # tensor and is a CPU tensor, don't copy it.
        self.tensor = self._tensor_prepare_func(self.tensor, False)  # tracing=False
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
            tensor_copy(self.tensor, loaded)

    def get_consuming_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")


def _identity_tensor_prepare_func(
    path: str, tensor: torch.Tensor, tracing: bool
) -> torch.Tensor:
    return tensor


class TensorIOPreparer:
    @staticmethod
    def prepare_write(
        storage_path: str,
        tensor: torch.Tensor,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[TensorEntry, List[WriteReq]]:
        if not _tensor_prepare_func:
            _tensor_prepare_func = functools.partial(_identity_tensor_prepare_func, "")

        proc_tensor = _tensor_prepare_func(tensor, True)  # tracing=True
        if proc_tensor.shape != tensor.shape:
            raise RuntimeError(
                "_tensor_prepare_func shouldn't change the tensor's shape "
                f"(changed from {tensor.shape} to {proc_tensor.shape})."
            )

        if proc_tensor.dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
            serializer = Serializer.BUFFER_PROTOCOL.value
        else:
            serializer = Serializer.TORCH_SAVE.value

        entry = TensorEntry(
            location=storage_path,
            serializer=serializer,
            dtype=dtype_to_string(proc_tensor.dtype),
            shape=list(proc_tensor.shape),
            replicated=False,
        )
        # stage the actual tensor, not processed tensor
        buffer_stager = TensorBufferStager(
            tensor=tensor,
            entry=entry,
            _tensor_prepare_func=_tensor_prepare_func,
        )
        return entry, [WriteReq(path=storage_path, buffer_stager=buffer_stager)]

    @classmethod
    def prepare_read(
        cls,
        entry: TensorEntry,
        tensor_out: Optional[torch.Tensor] = None,
        buffer_size_limit_bytes: Optional[int] = None,
    ) -> List[ReadReq]:
        # TODO: When the output tensor is a CPU tensor, we should directly load
        # into its storage buffer. This is an important optimization because:
        # - We eliminate an allocation and a copy
        # - With the extra allocation, the I/O concurrency will be severely
        # reduced by the scheduler in a memory constrained environment
        if tensor_out is None:
            raise RuntimeError(
                "Reading a Tensor without a runtime object is not yet supported."
            )

        if (
            buffer_size_limit_bytes is None
            or entry.serializer != Serializer.BUFFER_PROTOCOL.value
        ):
            buffer_consumer = TensorBufferConsumer(
                tensor=tensor_out,
                entry=entry,
            )
            return [ReadReq(path=entry.location, buffer_consumer=buffer_consumer)]

        num_chunks = math.ceil(
            cls.get_tensor_size_from_entry(entry) / buffer_size_limit_bytes
        )
        # Try to flatten the tensor without copying to achieve better chunking granularity.
        # This is only possible if the tensor satisfies the contiguity condition described in:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view
        try:
            tensor_out = tensor_out.view(-1)
        except RuntimeError:
            pass
        chunks = torch.chunk(tensor_out, chunks=num_chunks, dim=0)
        element_size = dtype_to_element_size(string_to_dtype(entry.dtype))

        read_reqs = []
        offset = 0
        for chunk in chunks:
            chunk_sz_bytes = chunk.nelement() * element_size
            buffer_consumer = TensorBufferConsumer(
                tensor=chunk,
                entry=TensorEntry(
                    location=entry.location,
                    serializer=entry.serializer,
                    dtype=entry.dtype,
                    shape=list(chunk.shape),
                    replicated=entry.replicated,
                ),
            )
            read_reqs.append(
                ReadReq(
                    path=entry.location,
                    byte_range=(offset, offset + chunk_sz_bytes),
                    buffer_consumer=buffer_consumer,
                )
            )
            offset += chunk_sz_bytes
        return read_reqs

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


class PrimitivePreparer:
    @staticmethod
    def should_inline(obj: Any) -> bool:
        type_name = type(obj).__name__
        if type_name not in PrimitiveEntry.supported_types():
            return False
        # TODO for long str/bytes, return False to fall back to ObjectEntry
        return True

    @staticmethod
    def prepare_write(obj: Any) -> PrimitiveEntry:
        return PrimitiveEntry.from_object(obj)


def prepare_write(
    obj: Any,
    logical_path: str,
    rank: int,
    replicated: bool,
    _tensor_prepare_func: Optional[Callable[[torch.Tensor, bool], torch.Tensor]] = None,
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
    if PrimitivePreparer.should_inline(obj):
        entry = PrimitivePreparer.prepare_write(obj)
        entry.replicated = replicated
        return entry, []

    storage_path = get_storage_path(obj, logical_path, rank, replicated)
    if isinstance(obj, ShardedTensor):
        return ShardedTensorIOPreparer.prepare_write(
            storage_path, obj, _tensor_prepare_func
        )
    elif isinstance(obj, torch.Tensor):
        entry, obj_write_req = TensorIOPreparer.prepare_write(
            storage_path, obj, _tensor_prepare_func
        )
    else:
        entry, obj_write_req = ObjectIOPreparer.prepare_write(storage_path, obj)

    entry.replicated = replicated
    return entry, obj_write_req


def prepare_read(
    entry: Entry,
    obj_out: Optional[Any] = None,
    buffer_size_limit_bytes: Optional[int] = None,
) -> List[ReadReq]:
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
    elif isinstance(entry, ChunkedTensorEntry):
        return ChunkedTensorIOPreparer.prepare_read(
            entry, obj_out, buffer_size_limit_bytes=buffer_size_limit_bytes
        )
    elif isinstance(entry, TensorEntry):
        return TensorIOPreparer.prepare_read(
            entry, obj_out, buffer_size_limit_bytes=buffer_size_limit_bytes
        )
    elif isinstance(entry, ObjectEntry):
        return ObjectIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, PrimitiveEntry):
        # primitive types are stored inline in snapshot metadata
        return []
    else:
        raise Exception(f"Unsupported entry type: {entry} ({entry.type}).")
