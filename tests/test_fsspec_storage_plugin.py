#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
import tempfile
import unittest

import torch

from torchsnapshot.io_types import ReadIO, WriteIO
from torchsnapshot.storage_plugins.fsspec import FSSpecPlugin
from torchsnapshot.test_utils import async_test

logger: logging.Logger = logging.getLogger(__name__)

_TENSOR_SZ = int(100_000_000 / 4)


class FSSpecStoragePluginTest(unittest.TestCase):
    @async_test
    async def test_write_read_delete(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            logger.info(path)
            plugin = FSSpecPlugin(root=path, protocol="file")

            tensor = torch.rand((_TENSOR_SZ,))
            tensor_path = os.path.join(path, "tensor")
            buf = io.BytesIO()
            torch.save(tensor, buf)
            write_io = WriteIO(path="tensor", buf=memoryview(buf.getvalue()))

            await plugin.write(write_io=write_io)
            self.assertTrue(os.path.exists(tensor_path))

            read_io = ReadIO(path="tensor")
            await plugin.read(read_io=read_io)
            loaded = torch.load(read_io.buf)
            self.assertTrue(torch.allclose(tensor, loaded))

            await plugin.delete(path="tensor")
            self.assertFalse(os.path.exists(tensor_path))
            await plugin.close()
