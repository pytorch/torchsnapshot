#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from torchsnapshot.io_types import IOReq, StoragePlugin


_DEFAULT_CONCURRENCY = 4


class GCSStoragePlugin(StoragePlugin):
    def __init__(self, root: str) -> None:
        try:
            # pyre-ignore
            from google.cloud import storage  # @manual
        except ImportError:
            raise RuntimeError(
                "GCS support requires google-cloud-storage. "
                "Please make sure google-cloud-storage is installed."
            )
        components = root.split("/")
        if len(components) < 2:
            raise RuntimeError(
                "The GCS root path must follow the following pattern: "
                f"[BUCKET]/[PATH] (got {root})"
            )
        self.bucket_name: str = components[0]
        self.root: str = "/".join(components[1:])
        # pyre-ignore
        self.client = storage.Client()
        # pyre-ignore
        self.bucket = self.client.bucket(self.bucket_name)
        self.executor = ThreadPoolExecutor(max_workers=_DEFAULT_CONCURRENCY)

    async def write(self, io_req: IOReq) -> None:
        loop = asyncio.get_running_loop()
        key = os.path.join(self.root, io_req.path)
        blob = self.bucket.blob(key)
        io_req.buf.seek(0)
        await loop.run_in_executor(
            self.executor, partial(blob.upload_from_file, io_req.buf)
        )

    async def read(self, io_req: IOReq) -> None:
        loop = asyncio.get_running_loop()
        key = os.path.join("gs://", self.bucket_name, self.root, io_req.path)
        await loop.run_in_executor(
            self.executor, partial(self.client.download_blob_to_file, key, io_req.buf)
        )
        io_req.buf.seek(0)

    async def delete(self, path: str) -> None:
        loop = asyncio.get_running_loop()
        key = os.path.join(self.root, path)
        blob = self.bucket.blob(key)
        await loop.run_in_executor(self.executor, blob.delete)

    def close(self) -> None:
        self.executor.shutdown()
