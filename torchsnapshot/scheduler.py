#!/usr/bin/env python3

import asyncio
import io
import logging
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import cast, List, Optional, Set

import psutil

from .io_types import BufferType, IOReq, ReadReq, StoragePlugin, WriteReq
from .pg_wrapper import PGWrapper

logger: logging.Logger = logging.getLogger(__name__)


_MAX_PER_RANK_MEMORY_BUDGET_BYTES: int = 32 * 1024 * 1024 * 1024
_AVAILABLE_MEMORY_MULTIPLIER: float = 0.8
_MAX_PER_RANK_CPU_CONCURRENCY: int = 4
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
        if isinstance(self.buf, bytes):
            self.buf_sz_bytes = len(self.buf)
        else:
            # self.buf is memoryview
            self.buf_sz_bytes = self.buf.nbytes
        return self

    async def write_buffer(self) -> "_WritePipeline":
        if self.buf is None:
            raise AssertionError("self.buf can not be None.")
        # pyre-ignore[6]: it's valid to initialize BytesIO with memoryview
        # according to: https://docs.python.org/3/library/io.html#io.BytesIO
        io_req = IOReq(path=self.write_req.path, buf=io.BytesIO(self.buf))
        await self.storage.write(io_req)

        # Reclaim buffer memory
        del io_req
        self.buf = None
        return self


class PendingIOWork:
    def __init__(
        self,
        ready_for_io: Set[_WritePipeline],
        io_tasks: Set[asyncio.Task],
        rank: int,
        memory_budget_bytes: int,
        begin_ts: float,
        bytes_written: int,
    ) -> None:
        self.ready_for_io = ready_for_io
        self.io_tasks = io_tasks
        self.rank = rank
        self.memory_budget_bytes = memory_budget_bytes
        self.begin_ts = begin_ts
        self.bytes_written = bytes_written

    async def complete(self) -> None:
        while len(self.ready_for_io) + len(self.io_tasks) != 0:
            done, _ = await asyncio.wait(
                self.io_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                write_pipeline: _WritePipeline = d.result()
                self.memory_budget_bytes += cast(int, write_pipeline.buf_sz_bytes)
                self.bytes_written += cast(int, write_pipeline.buf_sz_bytes)
                self.io_tasks.remove(d)
                for p in set(self.ready_for_io):
                    if len(self.io_tasks) >= _MAX_PER_RANK_IO_CONCURRENCY:
                        break
                    io_task = asyncio.create_task(p.write_buffer())
                    self.io_tasks.add(io_task)
                    self.ready_for_io.remove(p)
            logger.debug(
                f"Rank {self.rank}\t"
                f"ready_for_io: {len(self.ready_for_io)}\t"
                f"io_tasks: {len(self.io_tasks)}\t"
                f"memory_budget_bytes: {self.memory_budget_bytes}"
            )
        mbps = (self.bytes_written / 1024**2) / (time.monotonic() - self.begin_ts)
        logger.info(f"Rank {self.rank} finished saving. Throughput: {mbps:.2f}MB/s")

    def sync_complete(self, event_loop: asyncio.AbstractEventLoop) -> None:
        event_loop.run_until_complete(self.complete())


async def execute_write_reqs(
    write_reqs: List[WriteReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
) -> PendingIOWork:
    # This function fulfills the write requests by moving them through the
    # following stages with the specified memory budget:
    #
    # ready_for_staging - The is the initial state.
    # staging - Performing DtoH copy and serialization.
    # ready_for_io - A buffer is ready for the I/O stage.
    # io - Writing the buffer to the storage.
    # done (implicit) - The write request has been fulfilled.
    #
    # It returns as soon as all write requests are moved past the staging
    # stage.
    ready_for_staging = {_WritePipeline(write_req, storage) for write_req in write_reqs}
    staging_tasks = set()
    ready_for_io = set()
    io_tasks = set()
    executor = ThreadPoolExecutor(max_workers=_MAX_PER_RANK_CPU_CONCURRENCY)

    bytes_written = 0
    begin_ts = time.monotonic()

    def dispatch_staging(
        ready_for_staging: Set[_WritePipeline],
        staging_tasks: Set[asyncio.Task],
        memory_budget_bytes: int,
        executor: Executor,
    ) -> int:
        """
        Dispatch as many staging tasks as the memory budget allows.
        """
        for p in set(ready_for_staging):
            if len(staging_tasks) == 0 or p.staging_cost_bytes < memory_budget_bytes:
                memory_budget_bytes -= p.staging_cost_bytes
                staging_task = asyncio.create_task(p.stage_buffer(executor))
                staging_tasks.add(staging_task)
                ready_for_staging.remove(p)
        return memory_budget_bytes

    def dispatch_io(
        ready_for_io: Set[_WritePipeline], io_tasks: Set[asyncio.Task]
    ) -> None:
        """
        Dispatch as many I/O tasks as the I/O concurrency allows.
        """
        for p in set(ready_for_io):
            if len(io_tasks) >= _MAX_PER_RANK_IO_CONCURRENCY:
                break
            io_task = asyncio.create_task(p.write_buffer())
            io_tasks.add(io_task)
            ready_for_io.remove(p)

    memory_budget_bytes = dispatch_staging(
        ready_for_staging=ready_for_staging,
        staging_tasks=staging_tasks,
        memory_budget_bytes=memory_budget_bytes,
        executor=executor,
    )

    while len(ready_for_staging) + len(staging_tasks) != 0:
        done, _ = await asyncio.wait(
            staging_tasks | io_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for d in done:
            if d in staging_tasks:
                staging_tasks.remove(d)
                write_pipeline: _WritePipeline = d.result()
                ready_for_io.add(write_pipeline)
                # Update memory budget: the staging cost can be different from
                # the buffer size. For example, when serializing a tensor with
                # torch.save, the staging cost is 2x the buffer size.
                memory_budget_bytes += write_pipeline.staging_cost_bytes
                memory_budget_bytes -= cast(int, write_pipeline.buf_sz_bytes)
            elif d in io_tasks:
                io_tasks.remove(d)
                write_pipeline: _WritePipeline = d.result()
                memory_budget_bytes += cast(int, write_pipeline.buf_sz_bytes)
                bytes_written += cast(int, write_pipeline.buf_sz_bytes)
            else:
                raise AssertionError(
                    "The completed task must be in either staging_tasks or io_tasks."
                )
            dispatch_io(ready_for_io=ready_for_io, io_tasks=io_tasks)
            memory_budget_bytes = dispatch_staging(
                ready_for_staging=ready_for_staging,
                staging_tasks=staging_tasks,
                memory_budget_bytes=memory_budget_bytes,
                executor=executor,
            )
        logger.debug(
            f"Rank {rank}\t"
            f"ready_for_staging: {len(ready_for_staging)}\t"
            f"staging_tasks: {len(staging_tasks)}\t"
            f"ready_for_io: {len(ready_for_io)}\t"
            f"io_tasks: {len(io_tasks)}\t"
            f"memory_budget_bytes: {memory_budget_bytes}"
        )

    logger.debug(
        f"Rank {rank} finished staging in " f"{time.monotonic() - begin_ts:.2f} seconds"
    )
    executor.shutdown()
    return PendingIOWork(
        ready_for_io=ready_for_io,
        io_tasks=io_tasks,
        rank=rank,
        memory_budget_bytes=memory_budget_bytes,
        begin_ts=begin_ts,
        bytes_written=bytes_written,
    )


def sync_execute_write_reqs(
    write_reqs: List[WriteReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> PendingIOWork:
    if event_loop is None:
        event_loop = asyncio.new_event_loop()
    return event_loop.run_until_complete(
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
        self.buf: Optional[bytes] = None
        self.buf_sz_bytes: Optional[int] = None

    async def read_buffer(self) -> "_ReadPipeline":
        io_req = IOReq(path=self.read_req.path)
        await self.storage.read(io_req=io_req)
        self.buf = io_req.buf.getvalue()
        self.buf_sz_bytes = len(self.buf)
        return self

    async def consume_buffer(self, executor: Optional[Executor]) -> "_ReadPipeline":
        if self.buf is None:
            raise AssertionError("self.buf can not be None.")
        await self.read_req.buffer_consumer.consume_buffer(self.buf, executor)

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
    executor = ThreadPoolExecutor(max_workers=_MAX_PER_RANK_CPU_CONCURRENCY)

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
                consuming_task = asyncio.create_task(
                    read_pipeline.consume_buffer(executor)
                )
                consuming_tasks.add(consuming_task)
            if d in consuming_tasks:
                consuming_tasks.remove(d)
                read_pipeline: _ReadPipeline = d.result()
                memory_budget_bytes -= read_pipeline.consuming_cost_bytes
                bytes_read += cast(int, read_pipeline.buf_sz_bytes)

    mbps = (bytes_read / 1e6) / (time.monotonic() - begin_ts)
    logger.info(f"Rank {rank} finished loading. Throughput: {mbps:.2f}MB/s")

    executor.shutdown()


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
