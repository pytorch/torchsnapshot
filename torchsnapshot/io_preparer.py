#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import functools
import os
from typing import Any, List, Optional, Tuple

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

from .io_types import IOReq, ObjReadReq, ObjWriteReq
from .manifest import Entry, ObjectEntry, Shard, ShardedTensorEntry
from .torch_dist_checkpoint.metadata import (
    ExtendedTensorMetadata,
    StorageMetadata,
    TensorReadRequest,
)
from .torch_dist_checkpoint.resharding import (
    prepare_sharded_tensor_read,
    prepare_sharded_tensor_write,
)


class ShardedTensorIOPreparer:
    @staticmethod
    def prepare_write(
        storage_path: str, obj: ShardedTensor
    ) -> Tuple[ShardedTensorEntry, ObjWriteReq]:
        tensor_write_reqs, _ = prepare_sharded_tensor_write(obj, storage_path)
        shards = []
        io_reqs = []
        for shard, twr in zip(obj.local_shards(), tensor_write_reqs):
            shards.append(
                Shard(
                    offsets=shard.metadata.shard_offsets,
                    sizes=shard.metadata.shard_sizes,
                    location=twr.storage_key,
                )
            )
            io_reqs.append(IOReq(path=twr.storage_key))
            # Avoid saving the entire storage when saving a view
            if twr.tensor.is_cuda:
                # TODO: use non-blocking DtoH copy
                torch.save(twr.tensor.detach().cpu(), io_reqs[-1].buf)
            elif twr.tensor.nelement() != twr.tensor.storage().size():
                # TODO: avoid the additional data copy when saving the view
                # only without saving the entire storage
                # TODO: non-blocking HtoD copy
                torch.save(twr.tensor.detach().clone(), io_reqs[-1].buf)
            else:
                torch.save(twr.tensor, io_reqs[-1].buf)
        return ShardedTensorEntry(shards=shards), ObjWriteReq(io_reqs)

    @classmethod
    def prepare_read(
        cls,
        entry: ShardedTensorEntry,
        obj_out: Optional[ShardedTensor] = None,
    ) -> ObjReadReq[ShardedTensor]:
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
                    storage_key=shard.location,
                    length=0,  # Unused for load
                    offset=0,  # Unused for load
                )
            )

        tensor_read_reqs = prepare_sharded_tensor_read(
            metadata=metadata, sharded_tensor_out=obj_out
        )
        io_reqs = [
            IOReq(
                path=os.path.join(trr.storage_key),
            )
            for trr in tensor_read_reqs
        ]
        return ObjReadReq(
            io_reqs=io_reqs,
            on_read_complete=functools.partial(
                cls._on_read_complete, obj_out, tensor_read_reqs, io_reqs
            ),
        )

    @staticmethod
    def _on_read_complete(
        obj: ShardedTensor,
        tensor_read_reqs: List[TensorReadRequest],
        io_reqs: List[IOReq],
    ) -> ShardedTensor:
        views = {}
        for trr, io_req in zip(tensor_read_reqs, io_reqs):
            views[trr.storage_key] = torch.load(io_req.buf)
            # Reclaim buffer memory
            io_req.buf.close()

        for req in tensor_read_reqs:
            view_to_copy = views[req.storage_key]
            for dim, (start, length) in enumerate(zip(req.offsets, req.lengths)):
                view_to_copy = torch.narrow(view_to_copy, dim, start, length)

            assert (
                view_to_copy.size() == req.tensor.size()
            ), f"The {req.storage_key} src/dst size does not match."

            req.tensor.copy_(view_to_copy)
        return obj


class ObjectIOPreparer:
    @staticmethod
    # pyre-ignore[2]: obj can have arbitrary type
    def prepare_write(storage_path: str, obj: Any) -> Tuple[ObjectEntry, ObjWriteReq]:
        io_req = IOReq(path=storage_path)
        torch.save(obj, io_req.buf)
        return (
            ObjectEntry(
                type=type(obj).__module__ + "." + type(obj).__name__,
                location=storage_path,
                replicated=False,
            ),
            ObjWriteReq(io_reqs=[io_req]),
        )

    @classmethod
    def prepare_read(
        cls,
        entry: ObjectEntry,
        obj_out: Optional[torch.Tensor] = None,
    ) -> ObjReadReq[torch.Tensor]:
        io_req = IOReq(path=entry.location)
        return ObjReadReq(
            io_reqs=[io_req],
            on_read_complete=functools.partial(cls._on_read_complete, io_req, obj_out),
        )

    @staticmethod
    def _on_read_complete(
        io_req: IOReq, obj_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        obj = torch.load(io_req.buf)
        # Reclaim buffer memory
        io_req.buf.close()
        if obj_out is not None and isinstance(obj_out, torch.Tensor):
            # Is there a way to directly load to page-locked memory?
            # TODO: use non-blocking copy and yield control
            obj_out.copy_(obj)
        return obj


# pyre-ignore[2]: obj can have arbitrary type
def get_storage_path(obj: Any, logical_path: str, rank: int, replicated: bool) -> str:
    if isinstance(obj, ShardedTensor):
        return os.path.join("sharded", logical_path)
    elif replicated:
        return os.path.join("replicated", logical_path)
    else:
        return os.path.join(str(rank), logical_path)


async def prepare_gpu_tensor_write(
    storage_path: str, tensor: torch.Tensor
) -> Tuple[ObjectEntry, ObjWriteReq]:
    # Saving a GPU tensor involves 3 stages with different critical resources:
    # 1. DtoH copy (PCIe)
    # 2. Serialization (CPU)
    # 3. I/O (storage)
    # Pipelining these stages across different tensors allows us to achieve
    # better utilization of these critical resources which leads to speedup.
    cpu_tensor = tensor.to("cpu", non_blocking=True)
    copy_done = torch.cuda.Event()
    copy_done.record()

    # Use asyncio.sleep(0) to force a coroutine context
    # switch when the DtoH copy is not yet ready.
    await asyncio.sleep(0)
    while not copy_done.query():
        await asyncio.sleep(0)

    io_req = IOReq(path=storage_path)
    torch.save(cpu_tensor, io_req.buf)
    return (
        ObjectEntry(type="Tensor", location=storage_path, replicated=False),
        ObjWriteReq(io_reqs=[io_req]),
    )


async def prepare_write(
    # pyre-ignore[2]: obj can have arbitrary type
    obj: Any,
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[Entry, ObjWriteReq]:
    """
    Prepare write for an object.

    Args:
        obj: The object to save.
        logical_path: The logical path of the object.
        rank: The rank of the current process.
        replicated: Whether the object is replicated.

    Returns:
        The class::`Entry` describing the object, and the class::`ObjWriteReq`
        for persisting the object.
    """
    storage_path = get_storage_path(obj, logical_path, rank, replicated)
    if isinstance(obj, ShardedTensor):
        return ShardedTensorIOPreparer.prepare_write(storage_path, obj)
    elif isinstance(obj, torch.Tensor) and obj.is_cuda:
        entry, obj_write_req = await prepare_gpu_tensor_write(storage_path, obj)
    else:
        entry, obj_write_req = ObjectIOPreparer.prepare_write(storage_path, obj)

    entry.replicated = replicated
    return entry, obj_write_req


# pyre-ignore[2, 3]: obj can have arbitrary type
def prepare_read(entry: Entry, obj_out: Optional[Any] = None) -> ObjReadReq[Any]:
    """
    Prepare read for an object.

    Args:
        entry: The entry describing the object.
        obj: The object to load.

    Returns:
        The class::`ObjReadReq` for loading the object.
    """
    if isinstance(entry, ShardedTensorEntry):
        if obj_out is None:
            # TODO: support loading sharded tensor without obj_out
            raise RuntimeError(
                "Reading a ShardedTensor without a runtime object is not yet supported."
            )
        return ShardedTensorIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, ObjectEntry):
        return ObjectIOPreparer.prepare_read(entry)
    else:
        raise Exception(f"Unsupported entry type: {entry} ({entry.type}).")
