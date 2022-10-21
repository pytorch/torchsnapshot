#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import logging
import uuid

import pytest
import torch

from torchsnapshot.io_types import WriteIO, ReadIO
from torchsnapshot.storage_plugins.fsspec import FSSpecPlugin

logger: logging.Logger = logging.getLogger(__name__)

# _TEST_BUCKET = "torchsnapshot-test"
_TEST_BUCKET = "chengcshi"
# _TENSOR_SZ = int(100_000_000 / 4)
_TENSOR_SZ = 10


# @pytest.mark.s3_integration_test
# @pytest.mark.skipif(os.environ.get("TORCHSNAPSHOT_ENABLE_AWS_TEST") is None, reason="")
# @pytest.mark.usefixtures("s3_health_check")
@pytest.mark.asyncio
async def test_fsspec_s3_write_read_delete() -> None:
    path = f"fsspec-s3://{_TEST_BUCKET}/{uuid.uuid4()}"
    logger.info(path)
    plugin = FSSpecPlugin(root=path, key="AKIA34KDUMSNPNTFPTSA",
                          secret="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")

    tensor = torch.rand((_TENSOR_SZ,))
    buf = io.BytesIO()
    torch.save(tensor, buf)
    write_io = WriteIO(path="tensor", buf=memoryview(buf.getvalue()))

    await plugin.write(write_io=write_io)

    read_io = ReadIO(path="tensor")
    await plugin.read(read_io=read_io)
    loaded = torch.load(read_io.buf)
    assert torch.allclose(tensor, loaded)

    await plugin.delete(path="tensor")
    await plugin.close()
