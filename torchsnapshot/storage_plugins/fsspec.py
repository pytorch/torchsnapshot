#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import io
import os

import fsspec

from torchsnapshot.io_types import StoragePlugin, ReadIO, WriteIO


class FSSpecPlugin(StoragePlugin):
    def __init__(self, root: str, **storage_options) -> None:
        protocol, self.root = root.split("://")
        if not protocol.startswith("fsspec-"):
            raise ValueError(f"Invalid protocol: {protocol}, Only fsspec-* protocols are supported")
        self.fs = fsspec.filesystem(protocol=protocol.removeprefix("fsspec-"), **storage_options)
        self._session = None
        self._lock = asyncio.Lock()

    async def _init_session(self) -> None:
        async with self._lock:
            if self._session is None:
                self._session = await self.fs.set_session()

    async def write(self, write_io: WriteIO) -> None:
        await self._init_session()
        path = os.path.join(self.root, write_io.path)
        await self.fs._pipe_file(path, bytes(write_io.buf))

    async def read(self, read_io: ReadIO) -> None:
        await self._init_session()
        path = os.path.join(self.root, read_io.path)
        result = await self.fs._cat_file(path)
        read_io.buf = io.BytesIO(result)

    async def delete(self, path: str) -> None:
        await self._init_session()
        path = os.path.join(self.root, path)
        await self.fs._rm_file(path)

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None:
                await self._session.close()
                self._session = None
