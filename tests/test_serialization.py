#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_string,
    string_to_dtype,
    tensor_as_memoryview,
    tensor_from_memoryview,
)


class SerializationTest(unittest.TestCase):
    TENSOR_DIM = (1000, 1000)

    def _test_buffer_protocol_helper(self, dtype: torch.dtype) -> None:
        if dtype.is_floating_point:
            foo = torch.randn(self.TENSOR_DIM, dtype=dtype)
        elif dtype == torch.bool:
            foo = torch.randint(1, self.TENSOR_DIM, dtype=dtype)
        else:
            foo = torch.randint(torch.iinfo(dtype).max, self.TENSOR_DIM, dtype=dtype)

        serialized = tensor_as_memoryview(foo).tobytes()
        dtype_str = dtype_to_string(foo.dtype)
        shape = list(foo.shape)

        bar = tensor_from_memoryview(
            memoryview(serialized),
            dtype=string_to_dtype(dtype_str),
            shape=shape,
        )
        self.assertTrue(torch.allclose(foo, bar))

    def test_buffer_protocol(self) -> None:
        for dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_buffer_protocol_helper(dtype)

    def test_string_dtype_conversion(self) -> None:
        for dtype in ALL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                dtype_str = dtype_to_string(dtype)
                restored = string_to_dtype(dtype_str)
                self.assertEqual(restored, dtype)
