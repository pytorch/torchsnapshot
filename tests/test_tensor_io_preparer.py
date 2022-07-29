#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import tempfile

import unittest
from typing import cast, List

import torch

from torchsnapshot.io_preparer import TensorIOPreparer
from torchsnapshot.io_types import ReadReq, WriteReq
from torchsnapshot.scheduler import execute_read_reqs, execute_write_reqs
from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    Serializer,
)
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import async_test


class TensorIOPreparerTest(unittest.TestCase):
    TENSOR_DIM = (1000, 1000)

    @staticmethod
    def _fulfill_read_reqs_with_write_reqs(
        read_reqs: List[ReadReq], write_reqs: List[WriteReq]
    ) -> None:
        """
        Fulfill read requests with write requests. This allows us to test I/O
        preparers without depending on the scheduler and storage plugins.
        """
        if len(read_reqs) != len(write_reqs):
            raise AssertionError(
                "The size of read_reqs and write reqs must be the same."
            )
        path_to_buffer_stager = {
            write_req.path: write_req.buffer_stager for write_req in write_reqs
        }
        event_loop = asyncio.new_event_loop()
        for read_req in read_reqs:
            buffer_stager = path_to_buffer_stager[read_req.path]
            buffer_stager.get_staging_cost_bytes()
            read_req.buffer_consumer.get_consuming_cost_bytes()
            buf = event_loop.run_until_complete(buffer_stager.stage_buffer())
            event_loop.run_until_complete(
                read_req.buffer_consumer.consume_buffer(buf=cast(bytes, buf))
            )
        event_loop.close()

    @classmethod
    def _rand(cls, dtype: torch.dtype) -> torch.Tensor:
        if dtype in (torch.qint32, torch.qint8, torch.quint8):
            return torch.quantize_per_tensor(
                torch.rand(cls.TENSOR_DIM), 0.1, 10, dtype=dtype
            )
        elif dtype.is_floating_point or dtype.is_complex:
            return torch.randn(cls.TENSOR_DIM, dtype=dtype)
        elif dtype == torch.bool:
            return torch.randint(2, cls.TENSOR_DIM, dtype=dtype)
        else:
            return torch.randint(torch.iinfo(dtype).max, cls.TENSOR_DIM, dtype=dtype)

    def _test_tensor_io_preparer_helper(self, dtype: torch.dtype) -> None:
        """
        Use TensorIOPreparer to transfer the data of one tensor to a different
        tensor and verify that their data are the same.
        """
        foo = self._rand(dtype=dtype)
        bar = self._rand(dtype=dtype)

        # TODO: test memory estimation
        if foo.is_quantized:
            self.assertFalse(torch.allclose(foo.dequantize(), bar.dequantize()))
        else:
            self.assertFalse(torch.allclose(foo, bar))

        entry, write_reqs = TensorIOPreparer.prepare_write(
            storage_path="/foo", tensor=foo
        )
        # IMPORTANT: we should only serialize a tensor with BUFFER_PROTOCOL if
        # the tensor's dtype is in BUFFER_PROTOCOL_SUPPORTED_DTYPES.
        self.assertEqual(
            entry.serializer == Serializer.BUFFER_PROTOCOL.value,
            dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES,
        )

        read_reqs = TensorIOPreparer.prepare_read(entry=entry, tensor_out=bar)
        self._fulfill_read_reqs_with_write_reqs(
            read_reqs=read_reqs, write_reqs=write_reqs
        )
        if foo.is_quantized:
            self.assertTrue(torch.allclose(foo.dequantize(), bar.dequantize()))
        else:
            self.assertTrue(torch.allclose(foo, bar))

    def test_tensor_io_preparer(self) -> None:
        for dtype in ALL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_tensor_io_preparer_helper(dtype=dtype)

    def test_get_tensor_size_from_entry(self) -> None:
        for dtype in ALL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                tensor = self._rand(dtype=dtype)
                entry, _ = TensorIOPreparer.prepare_write("/foo", tensor)
                self.assertTrue(
                    TensorIOPreparer.get_tensor_size_from_entry(entry),
                    tensor.element_size() * tensor.nelement(),
                )

    async def _test_chunked_read_helper(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> None:
        """
        First save src tensor. Then load the persisted src tensor into dst
        tensor using a buffer_size_limit_bytes that would lead to chunked read.
        Finally verify that src tensor equals to dst tensor.
        """
        self.assertFalse(torch.allclose(src, dst))

        with tempfile.TemporaryDirectory() as path:
            storage = FSStoragePlugin(root=path)
            entry, write_reqs = TensorIOPreparer.prepare_write(
                storage_path="src", tensor=src
            )
            pending_io_work = await execute_write_reqs(
                write_reqs=write_reqs,
                storage=storage,
                memory_budget_bytes=32 * 1024**3,
                rank=0,
            )
            await pending_io_work.complete()

            buffer_size_limit_bytes = src.nelement() * src.element_size() // 4
            read_reqs = TensorIOPreparer.prepare_read(
                entry=entry,
                tensor_out=dst,
                buffer_size_limit_bytes=buffer_size_limit_bytes,
            )
            self.assertEqual(len(read_reqs), 4)
            await execute_read_reqs(
                read_reqs=read_reqs,
                storage=storage,
                memory_budget_bytes=32 * 1024**3,
                rank=0,
            )
        self.assertTrue(torch.allclose(src, dst))

    @async_test
    async def test_chunked_read(self) -> None:
        foo = torch.rand(2000, 2000)
        bar = torch.rand(2000, 2000)
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # strided
        foo = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2))
        bar = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2))
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # strided with storage_offset
        foo = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2), 1009)
        bar = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2), 1009)
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # non-contiguous view
        foo = torch.rand(4000, 4000)[1000:3000, 1000:3000]
        bar = torch.rand(4000, 4000)[1000:3000, 1000:3000]
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # 1009 is a prime number
        foo = torch.rand(1009, 1009)
        bar = torch.rand(1009, 1009)
        await self._test_chunked_read_helper(src=foo, dst=bar)
