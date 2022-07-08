#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest
import uuid

import torch
import torchsnapshot
from torchsnapshot.test_utils import async_test

logger: logging.Logger = logging.getLogger(__name__)

_TEST_BUCKET = "torchsnapshot-test"
_TENSOR_SZ = int(100_000_000 / 4)


class GCSStoragePluginTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None` to decorator factory
    #  `unittest.skipIf`.
    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, "")
    def test_read_write_via_snapshot(self) -> None:
        path = f"gs://{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)

        tensor = torch.rand((_TENSOR_SZ,))
        app_state = {"state": torchsnapshot.StateDict(tensor=tensor)}
        # pyre-fixme[6]: For 2nd param expected `Dict[str, Stateful]` but got
        #  `Dict[str, StateDict]`.
        snapshot = torchsnapshot.Snapshot.take(path=path, app_state=app_state)

        app_state["state"]["tensor"] = torch.rand((_TENSOR_SZ,))
        self.assertFalse(torch.allclose(tensor, app_state["state"]["tensor"]))

        # pyre-fixme[6]: For 1st param expected `Dict[str, Stateful]` but got
        #  `Dict[str, StateDict]`.
        snapshot.restore(app_state)
        self.assertTrue(torch.allclose(tensor, app_state["state"]["tensor"]))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None` to decorator factory
    #  `unittest.skipIf`.
    @unittest.skipIf(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, "")
    @async_test
    async def test_write_read_delete(self) -> None:
        path = f"{_TEST_BUCKET}/{uuid.uuid4()}"
        logger.info(path)

        from torchsnapshot.storage_plugins.gcs import GCSStoragePlugin

        plugin = GCSStoragePlugin(root=path)

        tensor = torch.rand((_TENSOR_SZ,))
        write_req = torchsnapshot.io_types.IOReq(path="tensor")
        torch.save(tensor, write_req.buf)

        await plugin.write(io_req=write_req)

        read_req = torchsnapshot.io_types.IOReq(path="tensor")
        await plugin.read(io_req=read_req)
        loaded = torch.load(read_req.buf)
        self.assertTrue(torch.allclose(tensor, loaded))

        await plugin.delete(path="tensor")
        await plugin.close()
