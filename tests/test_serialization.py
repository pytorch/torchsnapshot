#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import pytest

import torch

from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_string,
    per_channel_affine_qtensor_as_bytes,
    per_channel_affine_qtensor_from_bytes,
    per_tensor_affine_qtensor_as_bytes,
    per_tensor_affine_qtensor_from_bytes,
    string_to_dtype,
    tensor_as_memoryview,
    tensor_from_memoryview,
)
from torchsnapshot.test_utils import rand_tensor, tensor_eq


class SerializationTest(unittest.TestCase):
    def _test_buffer_protocol_helper(self, dtype: torch.dtype) -> None:
        foo = rand_tensor(shape=(1000, 1000), dtype=dtype)

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
            [torch.qint32, torch.qint8, torch.quint8],
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

    def test_per_channel_affine_qtensor(self) -> None:
        for shape, dtype in itertools.product(
            [(100, 100), (10, 11, 12), (10, 11, 12, 13)],
            [torch.qint32, torch.qint8, torch.quint8],
        ):
            for axis in range(len(shape)):
                qtensor = torch.quantize_per_channel(
                    torch.rand(shape),
                    torch.rand(shape[axis]),
                    torch.randint(128, (shape[axis],)),
                    axis=axis,
                    dtype=dtype,
                )
                buf = per_channel_affine_qtensor_as_bytes(qtensor)
                deserialized = per_channel_affine_qtensor_from_bytes(
                    buf, dtype=dtype, shape=list(shape)
                )
                self.assertEqual(qtensor.dtype, deserialized.dtype)
                self.assertTrue(qtensor.is_quantized)
                self.assertTrue(deserialized.is_quantized)
                self.assertEqual(qtensor.qscheme(), deserialized.qscheme())
                self.assertTrue(
                    torch.allclose(
                        qtensor.q_per_channel_scales(),
                        deserialized.q_per_channel_scales(),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        qtensor.q_per_channel_zero_points(),
                        deserialized.q_per_channel_zero_points(),
                    )
                )
                self.assertEqual(qtensor.stride(), deserialized.stride())
                self.assertTrue(
                    torch.allclose(qtensor.dequantize(), deserialized.dequantize())
                )


@pytest.mark.parametrize("dtype", BUFFER_PROTOCOL_SUPPORTED_DTYPES)
def test_tensor_as_memoryview_for_continuous_view(dtype: torch.dtype) -> None:
    """
    Verify that tensor_as_memoryview() behaves correctly for continuous views.
    """
    tensor = rand_tensor((64, 64), dtype=dtype)
    cont_view = tensor[32:, :]
    assert cont_view.is_contiguous()

    mv = tensor_as_memoryview(cont_view)
    assert len(mv) == cont_view.nelement() * cont_view.element_size()

    deserialized_view = tensor_from_memoryview(mv=mv, dtype=dtype, shape=[32, 64])
    assert tensor_eq(deserialized_view, cont_view)
