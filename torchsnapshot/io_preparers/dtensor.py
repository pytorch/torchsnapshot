#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch._prims_common import ShapeType

from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Placement,
    Replicate,
    Shard as ShardPlacement,
)
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from torchsnapshot.io_preparers.sharded_tensor import ShardedTensorIOPreparer

from torchsnapshot.io_preparers.tensor import TensorIOPreparer

from torchsnapshot.io_types import WriteReq
from torchsnapshot.knobs import get_max_shard_size_bytes
from torchsnapshot.manifest import DTensorEntry, Shard as ShardEntry

logger: logging.Logger = logging.getLogger(__name__)


class DTensorIOPreparer:
    @staticmethod
    def _get_largest_shard_dim(
        local_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> int:
        """
        Given a local shard, identify which dims have been sharded and chunk across the largest one.
        If it is not sharded, chunk over the largest dim.

        Args:
            shard (Tensor): local tensor on this rank
            mesh (DeviceMesh): device mesh associated with the DTensor the shard is from
            placements (Tuple[Placement,...]): placements associated with the DTensor the shard is from

        Returns:
            largest_shard_dim (int): index of largest dim that is sharded
        """
        # Find which dims of device mesh have Shard placements,
        # then get the tensor dim we are sharding across
        shard_dims = [
            placement.dim
            for placement in placements
            if isinstance(placement, ShardPlacement)
        ]

        # Find the largest sharding dim for the local tensor, so we can chunk
        # for parallelized saving
        if len(shard_dims) > 0:
            largest_shard_dim = max(shard_dims, key=lambda dim: local_shape[dim])
        else:
            # If tensor is not sharded then it is replicated across the device mesh.
            # Just chunk over largest dimension
            largest_shard_dim = torch.argmax(torch.tensor(local_shape)).item()

        return largest_shard_dim

    @staticmethod
    def _get_dim_map(obj: DTensor) -> List[List[int]]:
        dim_map = [[] for _ in range(len(obj.size()))]
        for mesh_dim, pm in enumerate(obj.placements):
            if isinstance(pm, ShardPlacement):
                dim_map[pm.dim].append(mesh_dim)
            elif not isinstance(pm, Replicate):
                raise ValueError("Unsupported placement type")
        # Add [-1] for mesh dims that aren't sharded
        dim_map = [dims if len(dims) != 0 else [-1] for dims in dim_map]
        return dim_map

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        obj: DTensor,
        is_async_snapshot: bool = False,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[DTensorEntry, List[WriteReq]]:
        """
        Compile list of write requests and the corresponding DTensorEntry given the provided
        DTensor object.

        Args:
            storage_path (str): device location of the DTensor. Because the DTensor is sharded and/or replicated
                the path will usually start with "sharded/" or "replicated/" instead of the rank number.
            obj (DTensor): DTensor object to be saved.
            is_async_snapshot (bool): whether or not the write request is from async_take
            _tensor_prepare_func (Optional[Callable[[torch.Tensor, bool], torch.Tensor]]): custom transform to apply
                to tensor before staging it in buffer.

        Returns:
            (DTensorEntry, List[WriteReq]): list of write requests and corresponding DTensorEntry
        """
        mesh = obj.device_mesh
        placements = obj.placements

        # Get shape of local shard on this rank and the offsets w.r.t. the global tensor
        local_shape, shard_offsets = compute_local_shape_and_global_offset(
            obj.size(), mesh, placements
        )

        # Determine which sharding dim to chunk
        largest_shard_dim = cls._get_largest_shard_dim(local_shape, mesh, placements)

        # Chunk the shard across the largest dim to improve save performance
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=obj.to_local(),
            offsets=list(shard_offsets),
            sizes=list(local_shape),
            dim=largest_shard_dim,
            max_shard_sz_bytes=get_max_shard_size_bytes(),
        )

        # Create write reqs for each chunk of the local shard for this rank
        # Each chunk is represented as a Shard in the manifest, although
        # the actual shard is the tensor pre-chunking. Chunking is simply
        # done for optimization purposes.
        write_reqs = []
        shards = []
        for tensor, offsets, sizes in subdivided:
            suffix = "_".join(str(i) for i in offsets)
            entry, tensor_write_reqs = TensorIOPreparer.prepare_write(
                storage_path=f"{storage_path}_{suffix}",
                tensor=tensor,
                is_async_snapshot=is_async_snapshot,
                _tensor_prepare_func=_tensor_prepare_func,
            )
            write_reqs += tensor_write_reqs

            shards.append(
                ShardEntry(
                    offsets=offsets,
                    sizes=sizes,
                    tensor=entry,
                )
            )

        dtensor_entry = DTensorEntry(
            shards=shards,
            mesh=mesh.mesh.cpu().numpy().tolist(),
            dim_map=cls._get_dim_map(obj),
        )

        return dtensor_entry, write_reqs
