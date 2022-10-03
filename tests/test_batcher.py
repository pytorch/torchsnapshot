#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[21, 56]: ignore pytest undefine import and invalid decoration
import random
import sys
from typing import Generator, List, Optional, Tuple

import torch
from torchsnapshot.batcher import (
    batch_read_requests,
    batch_write_requests,
    is_batchable,
)
from torchsnapshot.io_preparer import ObjectIOPreparer, TensorIOPreparer
from torchsnapshot.manifest import Entry
from torchsnapshot.serialization import ALL_SUPPORTED_DTYPES
from torchsnapshot.test_utils import rand_tensor

NUM_TENSORS = 50
TENSOR_SHAPE = (64, 64)


import uuid

import pytest
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torchsnapshot.io_preparer import (
    ChunkedTensorIOPreparer,
    prepare_read,
    ShardedTensorIOPreparer,
)
from torchsnapshot.io_types import WriteReq
from torchsnapshot.test_utils import tensor_eq, tensor_local_sz_bytes


@pytest.fixture
def dummy_pg() -> Generator[None, None, None]:
    """
    Fixture for initializing a single process pg.
    """
    dist.init_process_group(
        backend="gloo", init_method=f"file:///tmp/{uuid.uuid4()}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


@pytest.fixture
def sharded_tensor_test_cases(
    dummy_pg: None,
) -> Tuple[List[ShardedTensor], List[Entry], List[WriteReq], List[ShardedTensor]]:
    """
    Fixture for preparing sharded tensor test cases.

    Returns:
        - src tensors
        - Entries produced by ShardedTensorIOPreparer
        - Write requests produced by ShardedTensorIOPreparer
        - dst tensors whose values are different from the src tensors
    """
    spec = ChunkShardingSpec(  # pyre-ignore
        dim=0,
        placements=[
            "rank:0/cpu",
        ]
        * 4,
    )
    srcs = [sharded_tensor.empty(spec, TENSOR_SHAPE) for _ in range(NUM_TENSORS)]
    dsts = [sharded_tensor.empty(spec, TENSOR_SHAPE) for _ in range(NUM_TENSORS)]
    for tensor in srcs + dsts:
        for shard in tensor.local_shards():
            shard.tensor.random_()

    entries = []
    write_reqs = []
    for idx, tensor in enumerate(srcs):
        entry, wrs = ShardedTensorIOPreparer.prepare_write(
            storage_path=f"sharded_tensor_{idx}", obj=tensor
        )
        entries.append(entry)
        write_reqs.extend(wrs)
    return srcs, entries, write_reqs, dsts


@pytest.fixture
def tensor_test_cases(
    dtype: torch.dtype, enable_chunking: bool
) -> Tuple[List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]]:
    """
    Fixture for preparing tensor test cases.

    Returns:
        - src tensors
        - Entries produced by TensorIOPreparer/ChunkedTensorIOPreparer
        - Write requests produced by TensorIOPreparer/ChunkedTensorIOPreparer
        - dst tensors whose values are different from the src tensors
    """
    if dtype is not None:
        dtypes = [dtype] * NUM_TENSORS
    else:
        dtypes = [
            ALL_SUPPORTED_DTYPES[i % len(ALL_SUPPORTED_DTYPES)]
            for i in range(NUM_TENSORS)
        ]
    srcs = [rand_tensor(TENSOR_SHAPE, dtype=dtypes[i]) for i in range(NUM_TENSORS)]
    dsts = [rand_tensor(TENSOR_SHAPE, dtype=dtypes[i]) for i in range(NUM_TENSORS)]

    entries = []
    write_reqs = []
    for idx, tensor in enumerate(srcs):
        if enable_chunking:
            tensor_sz_bytes = tensor.nelement() * tensor.element_size()
            chunking_instruction = ChunkedTensorIOPreparer.chunk_tensor(
                tensor=tensor, chunk_sz_bytes=tensor_sz_bytes // 4
            )
            entry, wrs = ChunkedTensorIOPreparer.prepare_write(
                storage_path=f"{idx}",
                tensor=tensor,
                chunking_instruction=chunking_instruction,
            )
        else:
            entry, wrs = TensorIOPreparer.prepare_write(
                storage_path=f"{idx}", tensor=tensor
            )
        entries.append(entry)
        write_reqs.extend(wrs)
    return srcs, entries, write_reqs, dsts


@pytest.fixture
def slab_size_bytes(
    tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
    sharded_tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
) -> int:
    """
    Fixture for determining the slab size.

    Returns:
        Slab size inferred from tensor_test_cases and sharded_tensor_test_cases
        that makes sure multiple slabs are needed.
    """
    total_tensor_sz_bytes = 0
    for tensor in tensor_test_cases[0] + sharded_tensor_test_cases[0]:
        total_tensor_sz_bytes += tensor_local_sz_bytes(tensor=tensor)
    # We want to test multiple slabs
    return total_tensor_sz_bytes // 8


@pytest.fixture
def read_chunk_size_bytes(
    tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
    sharded_tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
) -> int:
    """
    Fixture for determining the read chunk size.

    Returns:
        Read chunk size inferred from tensor_test_cases and
        sharded_tensor_test_cases that makes sure each tensor is read as
        multiple chunks.
    """
    min_tensor_sz_bytes = sys.maxsize
    for tensor in tensor_test_cases[0] + sharded_tensor_test_cases[0]:
        min_tensor_sz_bytes = min(
            min_tensor_sz_bytes, tensor_local_sz_bytes(tensor=tensor)
        )
    # We want to test multiple slabs
    return min_tensor_sz_bytes // 4


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", ALL_SUPPORTED_DTYPES)
@pytest.mark.parametrize("enable_chunking", [True, False])
@pytest.mark.parametrize("enable_batched_read", [True, False])
@pytest.mark.parametrize("enable_chunked_read", [True, False])
async def test_batcher(
    tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
    sharded_tensor_test_cases: Tuple[
        List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]
    ],
    slab_size_bytes: int,
    read_chunk_size_bytes: Optional[int],
    enable_batched_read: bool,
    enable_chunked_read: bool,
) -> None:
    """
    Verify the behavior of the batcher.
    """
    src_tensors, tensor_entries, tensor_write_reqs, dst_tensors = tensor_test_cases
    (
        src_sharded_tensors,
        sharded_tensor_entries,
        sharded_tensor_write_reqs,
        dst_sharded_tensors,
    ) = sharded_tensor_test_cases

    src_tensors.extend(src_sharded_tensors)
    dst_tensors.extend(dst_sharded_tensors)
    entries = tensor_entries + sharded_tensor_entries
    write_reqs = tensor_write_reqs + sharded_tensor_write_reqs

    # Mix in some object write requests
    for idx in range(NUM_TENSORS):
        entry, wrs = ObjectIOPreparer.prepare_write(
            storage_path=f"object_{idx}", obj=object()
        )
        entries.append(entry)
        write_reqs.extend(wrs)

    # The order of the write request shouldn't matter so we shuffle it
    random.shuffle(write_reqs)

    # The src tensors and dst tensors should have different value now
    for src, dst in zip(src_tensors, dst_tensors):
        assert not tensor_eq(src, dst)

    # Expect the batcher to fail if it did not receive all affected entries
    # NOTE: don't shuffle the entries as they have to be aligned with
    # src_tensors and dst_tensors.
    if is_batchable(entries[0]):
        with pytest.raises(RuntimeError):
            batch_write_requests(
                entries=entries[1:],
                write_reqs=write_reqs,
            )

    # Batch the write requests
    entries, batched_write_reqs = batch_write_requests(
        entries=entries, write_reqs=write_reqs
    )
    assert len(batched_write_reqs) < len(write_reqs)
    write_reqs = batched_write_reqs

    # Prepare read requests for the dst tensors
    read_reqs = []
    for entry, obj_out in zip(entries, dst_tensors + [None] * NUM_TENSORS):
        rrs = prepare_read(
            entry=entry,
            obj_out=obj_out,
            buffer_size_limit_bytes=read_chunk_size_bytes
            if enable_chunked_read
            else None,
        )
        read_reqs.extend(rrs)

    # Read should work regardless of whether write was batched or not
    if enable_batched_read:
        batched_read_reqs = batch_read_requests(read_reqs=read_reqs)
        assert len(batched_read_reqs) < len(read_reqs)
        read_reqs = batched_read_reqs

    # Fulfill read requests with write requests
    location_to_buf = {
        wr.path: await wr.buffer_stager.stage_buffer(executor=None) for wr in write_reqs
    }
    for rr in read_reqs:
        buf = location_to_buf[rr.path]
        byte_range = rr.byte_range
        if byte_range is not None:
            buf = buf[byte_range[0] : byte_range[1]]
        else:
            buf = buf
        await rr.buffer_consumer.consume_buffer(buf, executor=None)

    # The src tensors and dst tensors should have the same values now
    for src, dst in zip(src_tensors, dst_tensors):
        assert tensor_eq(src, dst)
