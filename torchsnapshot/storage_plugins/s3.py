#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os

from torchsnapshot.io_types import IOReq, StoragePlugin


class S3StoragePlugin(StoragePlugin):
    def __init__(self, root: str) -> None:
        try:
            from aiobotocore.session import get_session  # @manual
        except ImportError:
            raise RuntimeError(
                "S3 support requires aiobotocore. "
                "Please make sure aiobotocore is installed."
            )
        components = root.split("/")
        if len(components) < 2:
            raise RuntimeError(
                "The S3 root path must follow the following pattern: "
                f"[BUCKET]/[PATH] (got {root})"
            )
        self.bucket: str = components[0]
        self.root: str = "/".join(components[1:])
        # pyre-ignore
        self.session = get_session()

    async def write(self, io_req: IOReq) -> None:
        async with self.session.create_client("s3") as client:
            key = os.path.join(self.root, io_req.path)
            io_req.buf.seek(0)
            await client.put_object(Bucket=self.bucket, Key=key, Body=io_req.buf)

    async def read(self, io_req: IOReq) -> None:
        async with self.session.create_client("s3") as client:
            key = os.path.join(self.root, io_req.path)
            response = await client.get_object(Bucket=self.bucket, Key=key)
            async with response["Body"] as stream:
                io_req.buf = io.BytesIO(await stream.read())

    async def delete(self, path: str) -> None:
        async with self.session.create_client("s3") as client:
            key = os.path.join(self.root, path)
            await client.delete_object(Bucket=self.bucket, Key=key)

    def close(self) -> None:
        pass
