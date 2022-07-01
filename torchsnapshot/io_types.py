#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import io
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Optional, Union


BufferType = Union[bytes, memoryview]


class BufferStager:
    @abc.abstractmethod
    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        pass

    @abc.abstractmethod
    def get_staging_cost_bytes(self) -> int:
        pass


@dataclass
class WriteReq:
    path: str
    buffer_stager: BufferStager


class BufferConsumer:
    @abc.abstractmethod
    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def get_consuming_cost_bytes(self) -> int:
        pass


@dataclass
class ReadReq:
    path: str
    buffer_consumer: BufferConsumer


@dataclass
class IOReq:
    path: str
    buf: io.BytesIO = field(default_factory=io.BytesIO)


class StoragePlugin(abc.ABC):
    @abc.abstractmethod
    async def write(self, io_req: IOReq) -> None:
        pass

    @abc.abstractmethod
    async def read(self, io_req: IOReq) -> None:
        pass

    @abc.abstractmethod
    async def delete(self, path: str) -> None:
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        pass

    def sync_write(
        self, io_req: IOReq, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = asyncio.new_event_loop()
        event_loop.run_until_complete(self.write(io_req=io_req))

    def sync_read(
        self, io_req: IOReq, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = asyncio.new_event_loop()
        event_loop.run_until_complete(self.read(io_req=io_req))

    def sync_close(
        self, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = asyncio.new_event_loop()
        event_loop.run_until_complete(self.close())
