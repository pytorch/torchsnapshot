#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
import unittest

import torch
import torchsnapshot
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import async_test

logger = logging.getLogger(__name__)

_TENSOR_SZ = int(100_000_000 / 4)


class FSStoragePluginTest(unittest.TestCase):
    @async_test
    async def test_write_read_delete(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            logger.info(path)
            plugin = FSStoragePlugin(root=path)

            tensor = torch.rand((_TENSOR_SZ,))
            tensor_path = os.path.join(path, "tensor")
            write_req = torchsnapshot.io_types.IOReq(path="tensor")
            torch.save(tensor, write_req.buf)

            await plugin.write(io_req=write_req)
            self.assertTrue(os.path.exists(tensor_path))

            read_req = torchsnapshot.io_types.IOReq(path="tensor")
            await plugin.read(io_req=read_req)
            loaded = torch.load(read_req.buf)
            self.assertTrue(torch.allclose(tensor, loaded))

            await plugin.delete(path="tensor")
            self.assertFalse(os.path.exists(tensor_path))
            await plugin.close()
