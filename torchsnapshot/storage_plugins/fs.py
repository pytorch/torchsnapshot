#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pathlib
from typing import Set

import aiofiles
import aiofiles.os
from torchsnapshot.io_types import IOReq, StoragePlugin


class FSStoragePlugin(StoragePlugin):
    def __init__(self, root: str) -> None:
        self.root = root
        self._dir_cache: Set[pathlib.Path] = set()

    async def write(self, io_req: IOReq) -> None:
        path = os.path.join(self.root, io_req.path)

        dir_path = pathlib.Path(path).parent
        if dir_path not in self._dir_cache:
            dir_path.mkdir(parents=True, exist_ok=True)
            self._dir_cache.add(dir_path)

        async with aiofiles.open(path, "wb+") as f:
            await f.write(io_req.buf.getvalue())

    async def read(self, io_req: IOReq) -> None:
        path = os.path.join(self.root, io_req.path)

        async with aiofiles.open(path, "rb") as f:
            io_req.buf = io.BytesIO(await f.read())

    async def delete(self, path: str) -> None:
        await aiofiles.os.remove(path)

    def close(self) -> None:
        pass
