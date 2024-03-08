#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import math
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import cast, ClassVar, List, Optional, Set

import psutil

from .io_types import BufferType, ReadIO, ReadReq, StoragePlugin, WriteIO, WriteReq
from .knobs import get_max_per_rank_io_concurrency
from .pg_wrapper import PGWrapper

logger: logging.Logger = logging.getLogger(__name__)


_MAX_PER_RANK_MEMORY_BUDGET_BYTES: int = 32 * 1024 * 1024 * 1024
_AVAILABLE_MEMORY_MULTIPLIER: float = 0.6
_MAX_PER_RANK_CPU_CONCURRENCY: int = 4


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
                f"Manually set process memory budget to {memory_budget_bytes} bytes."
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
        write_io = WriteIO(path=self.write_req.path, buf=self.buf)
        await self.storage.write(write_io=write_io)

        # Reclaim buffer memory
        del write_io
        self.buf = None
        return self


_LOG_LINE_LIMIT = 8


class _WriteReporter:
    _WRITE_LOG_TEMPLATE: ClassVar[str] = (
        "{rank:>4} {stageable:>12} {staging:>12} {writable:>12} "
        "{writing:>12} {rss_delta:>16} {memory_budget:>20} {data_written:>18}"
    )

    def __init__(
        self,
        ready_for_staging: Set[_WritePipeline],
        staging_tasks: Set[asyncio.Task],
        ready_for_io: Set[_WritePipeline],
        io_tasks: Set[asyncio.Task],
        rank: int,
        total_memory_budget_bytes: int,
    ) -> None:
        self.ready_for_staging = ready_for_staging
        self.staging_tasks = staging_tasks
        self.ready_for_io = ready_for_io
        self.io_tasks = io_tasks
        self.rank = rank
        self.total_memory_budget_bytes = total_memory_budget_bytes

        self._process = psutil.Process()
        self.baseline_rss_bytes: int = self._process.memory_info().rss
        self.begin_ts: float = time.monotonic()
        self.bytes_written = 0

        self._header: str = self._WRITE_LOG_TEMPLATE.format(
            rank="Rank",
            stageable="Stage-able",
            staging="Staging",
            writable="Writable",
            writing="Writing",
            rss_delta="RSS Delta (GB)",
            memory_budget="Memory Budget (GB)",
            data_written="Data Written (GB)",
        )

    def print_header(self) -> None:
        if self.rank == 0:
            logger.info(self._header)
            logger.info("-" * len(self._header))

    def report(self, memory_budget_bytes: int) -> None:
        rss_bytes = self._process.memory_info().rss
        rss_delta_gb = (rss_bytes - self.baseline_rss_bytes) / 1024**3
        msg = self._WRITE_LOG_TEMPLATE.format(
            rank=self.rank,
            stageable=len(self.ready_for_staging),
            staging=len(self.staging_tasks),
            writable=len(self.ready_for_io),
            writing=len(self.io_tasks),
            rss_delta=f"{rss_delta_gb:.2f}",
            memory_budget=(
                f"{memory_budget_bytes / 1024**3:.2f}/"
                f"{self.total_memory_budget_bytes / 1024**3:.2f}"
            ),
            data_written=f"{self.bytes_written / 1024**3:.2f}",
        )
        logger.info(msg)

    def report_staging_done(self) -> None:
        msg = (
            f"Rank {self.rank} completed staging in "
            f"{time.monotonic() - self.begin_ts:.2f} seconds"
        )
        logger.info(self._pad_msg(msg=msg))

    def report_writing_done(self) -> None:
        elapsed_secs = time.monotonic() - self.begin_ts
        mbps = self.bytes_written / 1024**2 / (elapsed_secs)
        msg = (
            f"Rank {self.rank} completed writing in "
            f"{elapsed_secs:.2f} seconds (throughput {mbps:.2f}MB/s)"
        )
        logger.info(self._pad_msg(msg=msg))

    def _pad_msg(self, msg: str) -> str:
        padding = (len(self._header) - len(msg) - 2) / 2
        return f"{'-' * math.ceil(padding)} {msg} {'-' * math.floor(padding)}"


class PendingIOWork:
    def __init__(
        self,
        ready_for_io: Set[_WritePipeline],
        io_tasks: Set[asyncio.Task],
        memory_budget_bytes: int,
        write_reporter: _WriteReporter,
    ) -> None:
        self.ready_for_io = ready_for_io
        self.io_tasks = io_tasks
        self.memory_budget_bytes = memory_budget_bytes
        self.write_reporter = write_reporter
        self.logging_freq: int = math.ceil(
            (len(self.ready_for_io) + len(self.io_tasks)) / _LOG_LINE_LIMIT
        )

    async def complete(self) -> None:
        while len(self.ready_for_io) + len(self.io_tasks) != 0:
            done, _ = await asyncio.wait(
                self.io_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                write_pipeline: _WritePipeline = d.result()
                self.memory_budget_bytes += cast(int, write_pipeline.buf_sz_bytes)
                self.write_reporter.bytes_written += cast(
                    int, write_pipeline.buf_sz_bytes
                )
                self.io_tasks.remove(d)
                for p in set(self.ready_for_io):
                    if len(self.io_tasks) >= get_max_per_rank_io_concurrency():
                        break
                    io_task = asyncio.create_task(p.write_buffer())
                    self.io_tasks.add(io_task)
                    self.ready_for_io.remove(p)
            if (len(self.ready_for_io) + len(self.io_tasks)) % self.logging_freq == 0:
                self.write_reporter.report(memory_budget_bytes=self.memory_budget_bytes)
        self.write_reporter.report_writing_done()

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
    ready_for_io: Set[_WritePipeline] = set()
    io_tasks: Set[asyncio.Task] = set()

    write_reporter = _WriteReporter(
        ready_for_staging=ready_for_staging,
        staging_tasks=staging_tasks,
        ready_for_io=ready_for_io,
        io_tasks=io_tasks,
        rank=rank,
        total_memory_budget_bytes=memory_budget_bytes,
    )
    logging_freq = math.ceil(len(ready_for_staging) / _LOG_LINE_LIMIT)
    write_reporter.print_header()

    executor = ThreadPoolExecutor(max_workers=_MAX_PER_RANK_CPU_CONCURRENCY)

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
            # Allow staging tasks whose cost exceeds the memory budget only
            # when there is no inflight write pipeline.
            if (
                len(staging_tasks) + len(ready_for_io) + len(io_tasks) == 0
                or p.staging_cost_bytes < memory_budget_bytes
            ):
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
            if len(io_tasks) >= get_max_per_rank_io_concurrency():
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
                if len(staging_tasks) % logging_freq == 0:
                    write_reporter.report(memory_budget_bytes=memory_budget_bytes)

            elif d in io_tasks:
                io_tasks.remove(d)
                write_pipeline: _WritePipeline = d.result()
                memory_budget_bytes += cast(int, write_pipeline.buf_sz_bytes)
                write_reporter.bytes_written += cast(int, write_pipeline.buf_sz_bytes)
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
    write_reporter.report_staging_done()
    executor.shutdown()
    return PendingIOWork(
        ready_for_io=ready_for_io,
        io_tasks=io_tasks,
        memory_budget_bytes=memory_budget_bytes,
        write_reporter=write_reporter,
    )


def sync_execute_write_reqs(
    write_reqs: List[WriteReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
    event_loop: asyncio.AbstractEventLoop,
) -> PendingIOWork:
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
        read_io = ReadIO(path=self.read_req.path, byte_range=self.read_req.byte_range)
        await self.storage.read(read_io=read_io)
        self.buf = read_io.buf.getvalue()
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
            if len(io_tasks) >= get_max_per_rank_io_concurrency():
                await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)
                break
            read_pipeline = read_pipelines[i]
            if (
                len(io_tasks) == 0
                or read_pipeline.consuming_cost_bytes < memory_budget_bytes
            ):
                memory_budget_bytes -= read_pipeline.consuming_cost_bytes
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
                memory_budget_bytes += read_pipeline.consuming_cost_bytes
                bytes_read += cast(int, read_pipeline.buf_sz_bytes)

    mbps = (bytes_read / 1024**2) / (time.monotonic() - begin_ts)
    logger.info(f"Rank {rank} finished loading. Throughput: {mbps:.2f}MB/s")

    executor.shutdown()


def sync_execute_read_reqs(
    read_reqs: List[ReadReq],
    storage: StoragePlugin,
    memory_budget_bytes: int,
    rank: int,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    event_loop.run_until_complete(
        execute_read_reqs(
            read_reqs=read_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=rank,
        )
    )
