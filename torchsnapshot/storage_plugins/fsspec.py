#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import io
import os
from typing import Any, Dict, Optional

import fsspec

from torchsnapshot.io_types import ReadIO, StoragePlugin, WriteIO

__all__ = ["FSSpecStoragePlugin"]


class FSSpecStoragePlugin(StoragePlugin):
    def __init__(self, root: str, storage_options: Optional[Dict[str, Any]]) -> None:
        root_items = root.split("://")
        if len(root_items) != 2:
            raise ValueError("only protocol://path is supported by fsspec plugin")
        protocol, self.root = root_items
        if not protocol.startswith("fsspec-"):
            raise ValueError(
                f"Invalid protocol: {protocol}, Only fsspec-* protocols are supported"
            )
        self._protocol = protocol[len("fsspec-") :]
        self._fs = None
        self._session = None
        self._lock = asyncio.Lock()
        self._storage_options = storage_options

    async def _init_session(self) -> None:
        async with self._lock:
            if self._session is None:
                self._fs = fsspec.filesystem(
                    protocol=self._protocol, **self._storage_options
                )
                self._session = await self._fs.set_session(refresh=True)

    async def write(self, write_io: WriteIO) -> None:
        await self._init_session()
        path = os.path.join(self.root, write_io.path)
        splits = path.split("/")
        for i in range(len(splits)):
            dir_path = "/".join(splits[:i])
            if dir_path and not await self._fs._exists(dir_path):
                try:
                    await self._fs._mkdir(dir_path)
                except AttributeError:
                    break
        await self._fs._pipe_file(path, bytes(write_io.buf))

    async def read(self, read_io: ReadIO) -> None:
        await self._init_session()
        path = os.path.join(self.root, read_io.path)
        data = await self._fs._cat_file(path)
        if read_io.byte_range is None:
            read_io.buf = io.BytesIO(data)
        else:
            start, end = read_io.byte_range
            read_io.buf = io.BytesIO(data[start:end])

    async def delete(self, path: str) -> None:
        await self._init_session()
        path = os.path.join(self.root, path)
        await self._fs._rm(path, recursive=True)

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None:
                await self._session.close()
                self._session = None
                self._fs = None
