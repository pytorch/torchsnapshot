#!/usr/bin/env python3

import asyncio
import io
import logging
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import cast, List, Optional

import psutil

from .io_types import BufferType, IOReq, ReadReq, StoragePlugin, WriteReq
from .pg_wrapper import PGWrapper

logger: logging.Logger = logging.getLogger(__name__)


_MAX_PER_RANK_MEMORY_BUDGET_BYTES: int = 32 * 1024 * 1024 * 1024
_AVAILABLE_MEMORY_MULTIPLIER: float = 0.8
_MAX_PER_RANK_STAGING_CONCURRENCY: int = 16
_MAX_PER_RANK_IO_CONCURRENCY: int = 16


def get_local_world_size(pg: PGWrapper) -> int:
    hostname = socket.gethostname()
    obj_list = [None] * pg.get_world_size()
    pg.all_gather_object(obj_list=obj_list, obj=hostname)

    hostname_world_size = defaultdict(int)
    for hostname in obj_list:
        hostname_world_size[hostname] += 1

    return hostname_world_size[hostname]


def get_process_memory_budget_bytes(pg: PGWrapper) -> int:
    if "TORCHSNAPSHOT_PER_RANK_MEMORY_BUDGET_BYTES" in os.environ:
        try:
            memory_budget_bytes = int(
                os.environ["TORCHSNAPSHOT_PER_RANK_MEMORY_BUDGET_BYTES"]
            )
            logger.info(
                "Manually set process memory budget to {memory_budget_bytes} bytes."
            )
            return memory_budget_bytes
        except Exception as e:
            logger.warning(f"Failed to override memory budget: {e}.")
    available_mem_bytes = int(
        psutil.virtual_memory().available * _AVAILABLE_MEMORY_MULTIPLIER
    )
    local_world_size = get_local_world_size(pg)
    memory_budget_bytes = min(
        available_mem_bytes // local_world_size, _MAX_PER_RANK_MEMORY_BUDGET_BYTES
    )
    logger.info(f"Set process memory budget to {memory_budget_bytes} bytes.")
    return memory_budget_bytes


class _WritePipeline:
    def __init__(self, write_req: WriteReq, storage: StoragePlugin) -> None:
        self.write_req = write_req
        self.staging_cost_bytes: int = write_req.buffer_stager.get_staging_cost_bytes()
        self.storage = storage
        self.buf: Optional[BufferType] = None
        self.buf_sz_bytes: Optional[int] = None

    async def stage_buffer(self, executor: Executor) -> "_WritePipeline":
        self.buf = await self.write_req.buffer_stager.stage_buffer(executor)
        self.buf_sz_bytes = len(self.buf)
        return self

    async def write_buffer(self) -> "_WritePipeline":
        if self.buf is None:
            raise AssertionError("self.buf can not be None.")
        io_req = IOReq(path=self.write_req.path, buf=io.BytesIO(self.buf))
        await self.storage.write(io_req)

        # Reclaim buffer memory
        del io_req
        self.buf = None
        return self


async def execute_write_reqs(
    write_reqs: List[WriteReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
) -> None:
    write_pipelines = [_WritePipeline(write_req, storage) for write_req in write_reqs]
    pending_ids = set(range(len(write_pipelines)))
    staging_tasks = set()
    io_tasks = set()
    executor = ThreadPoolExecutor(max_workers=_MAX_PER_RANK_STAGING_CONCURRENCY)

    bytes_written = 0
    begin_ts = time.monotonic()

    while len(pending_ids) != 0 or len(staging_tasks) != 0 or len(io_tasks) != 0:
        # Dispatch as many staging tasks as the memory budget allows
        dispatched_ids = set()
        for i in pending_ids:
            write_pipeline = write_pipelines[i]
            staging_cost_bytes = write_pipeline.staging_cost_bytes
            if len(staging_tasks) == 0 or staging_cost_bytes < memory_budget_bytes:
                memory_budget_bytes -= staging_cost_bytes
                staging_task = asyncio.create_task(
                    write_pipeline.stage_buffer(executor)
                )
                staging_tasks.add(staging_task)
                dispatched_ids.add(i)
        pending_ids -= dispatched_ids

        logger.debug(
            f"Rank {rank}\t"
            f"pending: {len(pending_ids)}\t"
            f"staing_tasks: {len(staging_tasks)}\t"
            f"io_tasks: {len(io_tasks)}\t"
            f"memory_budget_bytes: {memory_budget_bytes}"
        )

        # At this point, we can not dispatch more staging tasks
        done, _ = await asyncio.wait(
            staging_tasks | io_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for d in done:
            if d in staging_tasks:
                if len(io_tasks) >= _MAX_PER_RANK_IO_CONCURRENCY:
                    await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)
                    continue
                staging_tasks.remove(d)
                write_pipeline: _WritePipeline = d.result()
                # Update memory budget: the staging cost can be different from
                # the buffer size. For example, when serializing a tensor with
                # torch.save, the staging cost is 2x the buffer size.
                memory_budget_bytes += write_pipeline.staging_cost_bytes
                memory_budget_bytes -= cast(int, write_pipeline.buf_sz_bytes)
                io_task = asyncio.create_task(write_pipeline.write_buffer())
                io_tasks.add(io_task)
            if d in io_tasks:
                io_tasks.remove(d)
                write_pipeline: _WritePipeline = d.result()
                memory_budget_bytes += cast(int, write_pipeline.buf_sz_bytes)
                bytes_written += cast(int, write_pipeline.buf_sz_bytes)

    mbps = (bytes_written / 1e6) / (time.monotonic() - begin_ts)
    logger.info(f"Rank {rank} finished saving. Throughput: {mbps:.2f}MB/s")

    executor.shutdown()


def sync_execute_write_reqs(
    write_reqs: List[WriteReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    if event_loop is None:
        event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(
        execute_write_reqs(
            write_reqs=write_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=rank,
        )
    )


class _ReadPipeline:
    def __init__(self, read_req: ReadReq, storage: StoragePlugin) -> None:
        self.read_req = read_req
        self.consuming_cost_bytes: int = (
            read_req.buffer_consumer.get_consuming_cost_bytes()
        )
        self.storage = storage
        self.buf: Optional[BufferType] = None
        self.buf_sz_bytes: Optional[int] = None

    async def read_buffer(self) -> "_ReadPipeline":
        io_req = IOReq(path=self.read_req.path)
        await self.storage.read(io_req=io_req)
        self.buf = io_req.buf.getvalue()
        self.buf_sz_bytes = len(self.buf)
        return self

    async def consume_buffer(self) -> "_ReadPipeline":
        if self.buf is None:
            raise AssertionError("self.buf can not be None.")
        await self.read_req.buffer_consumer.consume_buffer(self.buf)

        # Reclaim buffer memory
        self.buf = None
        return self


async def execute_read_reqs(
    read_reqs: List[ReadReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
) -> None:
    read_pipelines = [_ReadPipeline(read_req, storage) for read_req in read_reqs]
    pending_ids = set(range(len(read_pipelines)))
    io_tasks = set()
    consuming_tasks = set()

    bytes_read = 0
    begin_ts = time.monotonic()

    while len(pending_ids) != 0 or len(io_tasks) != 0 or len(consuming_tasks) != 0:
        dispatched_ids = set()
        for i in pending_ids:
            if len(io_tasks) >= _MAX_PER_RANK_IO_CONCURRENCY:
                await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)
                break
            read_pipeline = read_pipelines[i]
            if (
                len(io_tasks) == 0
                or read_pipeline.consuming_cost_bytes < memory_budget_bytes
            ):
                memory_budget_bytes += read_pipeline.consuming_cost_bytes
                io_task = asyncio.create_task(read_pipeline.read_buffer())
                io_tasks.add(io_task)
                dispatched_ids.add(i)
        pending_ids -= dispatched_ids

        logger.debug(
            f"Rank {rank}\t"
            f"pending: {len(pending_ids)}\t"
            f"io_tasks: {len(io_tasks)}\t"
            f"consuming_tasks: {len(consuming_tasks)}\t"
            f"memory_budget_bytes: {memory_budget_bytes}"
        )

        done, _ = await asyncio.wait(
            io_tasks | consuming_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for d in done:
            if d in io_tasks:
                io_tasks.remove(d)
                read_pipeline: _ReadPipeline = d.result()
                consuming_task = asyncio.create_task(read_pipeline.consume_buffer())
                consuming_tasks.add(consuming_task)
            if d in consuming_tasks:
                consuming_tasks.remove(d)
                read_pipeline: _ReadPipeline = d.result()
                memory_budget_bytes -= read_pipeline.consuming_cost_bytes
                bytes_read += cast(int, read_pipeline.buf_sz_bytes)

    mbps = (bytes_read / 1e6) / (time.monotonic() - begin_ts)
    logger.info(f"Rank {rank} finished loading. Throughput: {mbps:.2f}MB/s")


def sync_execute_read_reqs(
    read_reqs: List[ReadReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    if event_loop is None:
        event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(
        execute_read_reqs(
            read_reqs=read_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=rank,
        )
    )
