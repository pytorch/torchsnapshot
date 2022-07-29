#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import io
import logging
import os
import unittest
import uuid

import torch
import torchsnapshot
from torchsnapshot.io_types import ReadIO, WriteIO
from torchsnapshot.test_utils import async_test

logger: logging.Logger = logging.getLogger(__name__)

_TEST_BUCKET = "torchsnapshot-benchmark"
_TENSOR_SZ = int(100_000_000 / 4)


class GCSStoragePluginTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, "")
    def test_read_write_via_snapshot(self) -> None:
        path = f"gs://{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)

        tensor = torch.rand((_TENSOR_SZ,))
        app_state = {"state": torchsnapshot.StateDict(tensor=tensor)}
        snapshot = torchsnapshot.Snapshot.take(path=path, app_state=app_state)

        app_state["state"]["tensor"] = torch.rand((_TENSOR_SZ,))
        self.assertFalse(torch.allclose(tensor, app_state["state"]["tensor"]))

        snapshot.restore(app_state)
        self.assertTrue(torch.allclose(tensor, app_state["state"]["tensor"]))

    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, "")
    @async_test
    async def test_write_read_delete(self) -> None:
        path = f"{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)

        from torchsnapshot.storage_plugins.gcs import GCSStoragePlugin

        plugin = GCSStoragePlugin(root=path)

        tensor = torch.rand((_TENSOR_SZ,))
        buf = io.BytesIO()
        torch.save(tensor, buf)
        write_io = WriteIO(path="tensor", buf=memoryview(buf.getvalue()))
        await plugin.write(write_io=write_io)

        read_io = ReadIO(path="tensor")
        await plugin.read(read_io=read_io)
        loaded = torch.load(read_io.buf)
        self.assertTrue(torch.allclose(tensor, loaded))

        # TODO: bring this back
        # await plugin.delete(path="tensor")
        await plugin.close()
