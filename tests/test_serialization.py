#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch

from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_string,
    per_tensor_affine_qtensor_as_bytes,
    per_tensor_affine_qtensor_from_bytes,
    string_to_dtype,
    tensor_as_memoryview,
    tensor_from_memoryview,
)


class SerializationTest(unittest.TestCase):
    def _test_buffer_protocol_helper(self, dtype: torch.dtype) -> None:
        if dtype.is_floating_point:
            foo = torch.randn((1000, 1000), dtype=dtype)
        elif dtype == torch.bool:
            foo = torch.randint(1, (1000, 1000), dtype=dtype)
        else:
            foo = torch.randint(torch.iinfo(dtype).max, (1000, 1000), dtype=dtype)

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

    def test_per_tensor_affine_qtensor(self) -> None:
        for shape, scale, zero_point, dtype in itertools.product(
            [(100, 100), (10, 11, 12), (10, 11, 12, 13)],
            [0.1, 0.2],
            [1, 10],
            [torch.qint8, torch.quint8],
        ):
            qtensor = torch.quantize_per_tensor(
                torch.rand(shape), scale, zero_point, dtype=dtype
            )
            buf = per_tensor_affine_qtensor_as_bytes(qtensor)
            deserialized = per_tensor_affine_qtensor_from_bytes(
                buf, dtype=dtype, shape=list(shape)
            )
            self.assertEqual(qtensor.dtype, deserialized.dtype)
            self.assertTrue(qtensor.is_quantized)
            self.assertTrue(deserialized.is_quantized)
            self.assertEqual(qtensor.qscheme(), deserialized.qscheme())
            self.assertEqual(qtensor.q_scale(), deserialized.q_scale())
            self.assertEqual(qtensor.q_zero_point(), deserialized.q_zero_point())
            self.assertEqual(qtensor.stride(), deserialized.stride())
            self.assertTrue(
                torch.allclose(qtensor.dequantize(), deserialized.dequantize())
            )
