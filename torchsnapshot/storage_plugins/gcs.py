#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import functools
import io
import logging
import os
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar
from urllib.parse import quote

import aiohttp  # @manual

# pyre-ignore-all-errors[21]: Undefined import
import google.auth.exceptions  # @manual
from google._async_resumable_media.requests import (  # @manual
    ChunkedDownload,
    ResumableUpload,
)
from google.auth._default_async import default_async  # @manual
from google.auth.transport._aiohttp_requests import AuthorizedSession  # @manual

from torchsnapshot.io_types import IOReq, StoragePlugin

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_DEADLINE_SEC: int = 180
_DEFAULT_CHUNK_SIZE_BYTE: int = 100 * 1024 * 1024


class GCSStoragePlugin(StoragePlugin):
    SCOPES = (
        "https://www.googleapis.com/auth/devstorage.full_control",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/devstorage.read_write",
    )
    UPLOAD_URL_TEMPLATE = (
        "https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?"
        "uploadType=resumable"
    )
    DOWNLOAD_URL_TEMPLATE = (
        "https://www.googleapis.com/download/storage/v1/b/"
        "{bucket}/o/{blob_name}?alt=media"
    )

    def __init__(self, root: str) -> None:
        components = root.split("/")
        if len(components) < 2:
            raise RuntimeError(
                "The GCS root path must follow the following pattern: "
                f"[BUCKET]/[PATH] (got {root})"
            )
        self.bucket_name: str = components[0]
        self.root: str = "/".join(components[1:])

        # pyre-ignore
        credentials, _ = default_async(scopes=self.SCOPES)
        # pyre-ignore
        self.authed_session = AuthorizedSession(credentials)
        self.retry_strategy = _RetryStrategy(deadline_sec=_DEFAULT_DEADLINE_SEC)

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        return isinstance(
            e,
            (
                ConnectionError,
                aiohttp.ClientConnectionError,
                # pyre-ignore
                google.auth.exceptions.TransportError,
            ),
        )

    @staticmethod
    async def _recover_resumable_upload(
        # pyre-ignore
        upload: ResumableUpload,
        stream: io.BytesIO,
    ) -> None:
        if upload.invalid:
            await upload.recover()
        # When ResumableUpload becomes invalid, its .recover() method rewinds
        # the stream. However, certain failures can cause the cursor position
        # to be different from ResumableUpload.bytes_uploaded without rendering
        # the ResumableUpload invalid. In such cases, we need to rewind the
        # stream explicitly.
        stream.seek(upload.bytes_uploaded)

    async def write(self, io_req: IOReq) -> None:
        # pyre-ignore
        upload = ResumableUpload(
            upload_url=self.UPLOAD_URL_TEMPLATE.format(bucket=self.bucket_name),
            chunk_size=_DEFAULT_CHUNK_SIZE_BYTE,
        )
        await self.retry_strategy.await_with_retry(
            func=functools.partial(
                upload.initiate,
                transport=self.authed_session,
                stream=io_req.buf,
                metadata={"name": os.path.join(self.root, io_req.path)},
                content_type="application/octet-stream",
            ),
            is_transient_error=self._is_transient_error,
        )
        while not upload.finished:
            await self.retry_strategy.await_with_retry(
                func=functools.partial(
                    upload.transmit_next_chunk, transport=self.authed_session
                ),
                is_transient_error=self._is_transient_error,
                before_retry=functools.partial(
                    self._recover_resumable_upload, upload=upload, stream=io_req.buf
                ),
            )

    async def read(self, io_req: IOReq) -> None:
        blob_name = quote(
            os.path.join(self.root, io_req.path).encode("utf-8"), safe=b"~"
        )
        # pyre-ignore
        download = ChunkedDownload(
            media_url=self.DOWNLOAD_URL_TEMPLATE.format(
                bucket=self.bucket_name,
                blob_name=blob_name,
            ),
            chunk_size=_DEFAULT_CHUNK_SIZE_BYTE,
            stream=io_req.buf,
        )
        while not download.finished:
            await self.retry_strategy.await_with_retry(
                func=functools.partial(
                    download.consume_next_chunk, transport=self.authed_session
                ),
                is_transient_error=self._is_transient_error,
            )

    async def delete(self, path: str) -> None:
        raise NotImplementedError()

    async def close(self) -> None:
        await self.authed_session.close()


T = TypeVar("T")


class _RetryStrategy:
    """
    A retry strategy that takes the collective progress of concurrent
    coroutines into consideration.

    All concurrent coroutines share the same deadline, which is refreshed when
    a new concurrent coroutine is kicked off or when a concurrent coroutine
    completes. The targeted effect is that all coroutines can retry on
    transient error as long as some concurrent coroutines are making progress.
    The rationale behind the retry strategy is that we favor progress over
    efficiency when faced with congestion.

    NOTE: this class is not thread-safe.
    """

    INITIAL_BACKOFF = 1
    BACKOFF_MULTIPLIER = 2

    def __init__(self, deadline_sec: int) -> None:
        self.deadline_sec = deadline_sec
        self.countdown_begin_ts = 0

    async def await_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        is_transient_error: Callable[[Exception], bool],
        before_retry: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> T:
        self.countdown_begin_ts = int(time.monotonic())
        backoff_sec = self.INITIAL_BACKOFF
        while True:
            try:
                ret = await func()
            except Exception as e:
                if not is_transient_error(e):
                    raise e
                if time.monotonic() >= self.countdown_begin_ts + self.deadline_sec:
                    logger.warn(
                        f"Encountered a retryable error: {e}\n"
                        "Not retrying since no concurrent requests were "
                        f"completed within {self.deadline_sec} seconds."
                    )
                    raise e
                else:
                    backoff_sec *= self.BACKOFF_MULTIPLIER
                    backoff_sec += random.randint(0, 1000) * 0.001  # jitter
                    logger.warn(
                        f"Encountered a retryable error: {e}\n"
                        f"Retrying after {backoff_sec} seconds."
                    )
                    await asyncio.sleep(backoff_sec)
                    if before_retry is not None:
                        await before_retry()
                    continue
            else:
                self.countdown_begin_ts = int(time.monotonic())
                return ret
