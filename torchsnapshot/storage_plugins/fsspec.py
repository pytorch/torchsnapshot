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
        self.fs = fsspec.filesystem(protocol, **storage_options)

    async def write(self, write_io: WriteIO) -> None:
        path = os.path.join(self.root, write_io.path)
        with self.fs.open(path, 'wb+') as f:
            f.write(write_io.buf)

    async def read(self, read_io: ReadIO) -> None:
        path = os.path.join(self.root, read_io.path)
        byte_range = read_io.byte_range

        with self.fs.open(path, 'rb') as f:
            if byte_range is None:
                read_io.buf = io.BytesIO(f.read())
            else:
                offset = byte_range[0]
                size = byte_range[1] - byte_range[0]
                await f.seek(offset)
                read_io.buf = io.BytesIO(f.read(size))

    async def delete(self, path: str) -> None:
        path = os.path.join(self.root, path)
        self.fs.delete(path)

    async def close(self) -> None:
        pass
