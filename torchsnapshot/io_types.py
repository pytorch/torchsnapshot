#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import io
from dataclasses import dataclass, field
from typing import Callable, Generic, List, TypeVar


T = TypeVar("T")


@dataclass
class IOReq:
    path: str
    buf: io.BytesIO = field(default_factory=io.BytesIO)


@dataclass
class ObjWriteReq:
    io_reqs: List[IOReq]


@dataclass
class ObjReadReq(Generic[T]):
    io_reqs: List[IOReq]
    on_read_complete: Callable[[], T]


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
    def close(self) -> None:
        pass

    async def write_obj(self, obj_write_req: ObjWriteReq) -> None:
        for io_req in obj_write_req.io_reqs:
            await self.write(io_req)
            # Reclaim buffer memory
            io_req.buf.close()

    async def read_obj(self, obj_read_req: ObjReadReq[T]) -> T:
        await asyncio.gather(*[self.read(io_req) for io_req in obj_read_req.io_reqs])
        return obj_read_req.on_read_complete()
