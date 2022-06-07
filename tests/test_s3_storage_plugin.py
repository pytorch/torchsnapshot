#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import unittest
import uuid

import torch
import torchsnapshot
from torchsnapshot.storage_plugins.s3 import S3StoragePlugin

logger = logging.getLogger(__name__)

_TEST_BUCKET = "torchsnapshot-test"
_TENSOR_SZ = int(100_000_000 / 4)


class S3StoragePluginTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_AWS_TEST") is None, "")
    def test_read_write_via_snapshot(self) -> None:
        path = f"s3://{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)

        tensor = torch.rand((_TENSOR_SZ,))
        app_state = {"state": torchsnapshot.StateDict(tensor=tensor)}
        snapshot = torchsnapshot.Snapshot.take(path=path, app_state=app_state)

        app_state["state"]["tensor"] = torch.rand((_TENSOR_SZ,))
        self.assertFalse(torch.allclose(tensor, app_state["state"]["tensor"]))

        snapshot.restore(app_state)
        self.assertTrue(torch.allclose(tensor, app_state["state"]["tensor"]))

    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_AWS_TEST") is None, "")
    def test_write_read(self) -> None:
        path = f"{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)
        plugin = S3StoragePlugin(root=path)

        tensor = torch.rand((_TENSOR_SZ,))
        write_req = torchsnapshot.io_types.IOReq(path=os.path.join(path, "tensor"))
        torch.save(tensor, write_req.buf)
        asyncio.run(plugin.write(io_req=write_req))

        read_req = torchsnapshot.io_types.IOReq(path=os.path.join(path, "tensor"))
        asyncio.run(plugin.read(io_req=read_req))
        loaded = torch.load(read_req.buf)

        self.assertTrue(torch.allclose(tensor, loaded))

        plugin.close()
