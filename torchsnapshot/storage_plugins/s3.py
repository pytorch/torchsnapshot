#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import os
from typing import Any, Dict, Optional

from torchsnapshot.io_types import ReadIO, StoragePlugin, WriteIO
from torchsnapshot.memoryview_stream import MemoryviewStream

from aioboto3 import Session
from boto3.s3.transfer import TransferConfig

class S3StoragePlugin(StoragePlugin):
    def __init__(
        self, root: str, storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        components = root.split("/")
        if len(components) < 2:
            raise RuntimeError(
                "The S3 root path must follow the following pattern: "
                f"[BUCKET]/[PATH] (got {root})"
            )
        self.bucket: str = components[0]
        self.root: str = "/".join(components[1:])
        self.session = Session()

    async def write(self, write_io: WriteIO) -> None:
        if isinstance(write_io.buf, bytes):
            stream = io.BytesIO(write_io.buf)
        elif isinstance(write_io.buf, memoryview):
            stream = MemoryviewStream(write_io.buf)
        else:
            raise TypeError(f"Unrecognized buffer type: {type(write_io.buf)}")
        async with self.session.client("s3") as client:
            key = os.path.join(self.root, write_io.path)
            config = TransferConfig(
                multipart_threshold=8388608,
                multipart_chunksize=8388608,
                max_concurrency=128,
                use_threads=True
            )
            stream.seek(0)
            await client.upload_fileobj( 
                Fileobj=stream,
                Bucket=self.bucket,
                Key=key,
                Config=config,
            )

    async def read(self, read_io: ReadIO) -> None:
        async with self.session.client("s3") as client:
            key = os.path.join(self.root, read_io.path)
            byte_range = read_io.byte_range
            if byte_range is None:
                response = await client.get_object(Bucket=self.bucket, Key=key)
            else:
                response = await client.get_object(
                    Bucket=self.bucket,
                    Key=key,
                    # HTTP Byte Range is inclusive:
                    # https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.35
                    Range=f"bytes={byte_range[0]}-{byte_range[1] - 1}",
                )
            async with response["Body"] as stream:
                read_io.buf = io.BytesIO(await stream.read())

    async def delete(self, path: str) -> None:
        async with self.session.client("s3") as client:
            key = os.path.join(self.root, path)
            await client.delete_object(Bucket=self.bucket, Key=key)

    async def delete_dir(self, path: str) -> None:
        raise NotImplementedError()

    async def close(self) -> None:
        pass
