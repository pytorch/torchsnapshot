#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import os

import fsspec

from torchsnapshot.io_types import StoragePlugin, ReadIO, WriteIO


class FSSpecPlugin(StoragePlugin):
    def __init__(self, root: str, protocol: str, **storage_options) -> None:
        self.root = root
        if protocol not in ["http", "s3"]:
            raise ValueError(f"Protocol {protocol} does not support async")
        self.fs = fsspec.filesystem(protocol, asynchronous=True, **storage_options)

    async def write(self, write_io: WriteIO) -> None:
        path = os.path.join(self.root, write_io.path)
        session = await self.fs.set_session()
        with self.fs.open(path, 'wb') as f:
            await f.write(write_io.buf)
        await session.close()

    async def read(self, read_io: ReadIO) -> None:
        path = os.path.join(self.root, read_io.path)
        byte_range = read_io.byte_range

        session = await self.fs.set_session()
        with self.fs.open(path, 'rb') as f:
            if byte_range is None:
                read_io.buf = io.BytesIO(await f.read())
            else:
                offset = byte_range[0]
                size = byte_range[1] - byte_range[0]
                await f.seek(offset)
                read_io.buf = io.BytesIO(await f.read(size))
        await session.close()

    async def delete(self, path: str) -> None:
        path = os.path.join(self.root, path)
        session = await self.fs.set_session()
        await self.fs.delete(path)
        await session.close()

    async def close(self) -> None:
        pass
