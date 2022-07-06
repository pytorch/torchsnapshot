#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from enum import Enum
from typing import Dict, List

import torch

logger: logging.Logger = logging.getLogger(__name__)


# https://pytorch.org/docs/stable/tensors.html#data-types
ALL_SUPPORTED_DTYPES: List[torch.dtype] = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.complex128,
    torch.complex64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.qint32,
    torch.qint8,
    torch.quint8,
]

# The approach is dumb. But we want to be 100% certain we can recognize the
# dtype strings we persist.
_DTYPE_TO_STRING: Dict[torch.dtype, str] = {
    torch.float64: "torch.float64",
    torch.float32: "torch.float32",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.complex128: "torch.complex128",
    torch.complex64: "torch.complex64",
    torch.int64: "torch.int64",
    torch.int32: "torch.int32",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
    torch.qint32: "torch.qint32",
    torch.qint8: "torch.qint8",
    torch.quint8: "torch.quint8",
}

# This could be figured out by creating an empty tensor with the dtype and
# checking its .element_size(). However, the approach may not work for
# quantized dtypes if the appropriate backend is not available. Since special
# casing is inevitable, we might as well enumerate all dtypes.
_DTYPE_TO_ELEMENT_SIZE: Dict[torch.dtype, int] = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.complex128: 16,
    torch.complex64: 8,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
    torch.qint32: 4,
    torch.qint8: 1,
    torch.quint8: 1,
}


# The approach is dumb. But we want to be 100% certain we can recognize the
# dtype strings we persist.
_STRING_TO_DTYPE: Dict[str, torch.dtype] = {
    val: key for key, val in _DTYPE_TO_STRING.items()
}


def dtype_to_string(dtype: torch.dtype) -> str:
    """
    Converty a class::`torch.dtype` to class::`str`.
    """
    if dtype in _DTYPE_TO_STRING:
        return _DTYPE_TO_STRING[dtype]
    else:
        raise ValueError(
            f"Unsupported dtype {dtype}. "
            f"(Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


def dtype_to_element_size(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_TO_ELEMENT_SIZE:
        return _DTYPE_TO_ELEMENT_SIZE[dtype]
    else:
        raise ValueError(
            f"Unsupported dtype {dtype}. "
            f"(Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


def string_to_dtype(s: str) -> torch.dtype:
    """
    Converty a class::`torch.dtype` to class::`str`.
    """
    if s in _STRING_TO_DTYPE:
        return _STRING_TO_DTYPE[s]
    else:
        raise ValueError(
            f"Unsupported dtype {s}. " f"(Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


class Serializer(Enum):
    TORCH_SAVE = "torch_save"
    BUFFER_PROTOCOL = "buffer_protocol"


BUFFER_PROTOCOL_SUPPORTED_DTYPES: List[torch.dtype] = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


def tensor_as_memoryview(tensor: torch.Tensor) -> memoryview:
    """
    Obtain the class::`memoryview` of a class::`torch.Tensor`.

    It is the caller's responsibility to ensure that the input is a CPU tensor
    and its dtype is defined in `BUFFER_PROTOCOL_SUPPORTED_DTYPES`. The
    function will raise an exception if the requirements are not met.

    Args:
        tensor: The tensor from which to obtain memoryview.

    Returns:
        The class::`memoryview` of the input tensor.
    """
    if tensor.dtype not in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
        raise ValueError(
            f"tensor_as_memoryview() doesn't support the dtype {tensor.dtype}."
        )
    if tensor.device != torch.device("cpu"):
        raise ValueError("tensor_as_memoryview() only accepts CPU tensors.")
    if not tensor.is_contiguous():
        # This is only needed if the caller didn't need to copy the tensor from
        # device to CPU. This is still more efficient than torch.save().
        tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        return _bfloat16_tensor_to_memoryview(tensor)
    return memoryview(tensor.numpy())


def _bfloat16_tensor_to_memoryview(tensor: torch.Tensor) -> memoryview:
    """
    A specialization of func::`tensor_as_memoryview` for `bfloa16`.

    Currently the memoryview of a tensor can only be obtained via its numpy
    array representation. However, numpy doesn't support `bfloat16`, so
    `bfloat16` tensor's can't be converted to a numpy array.

    Since we only need the memoryview to avoid data copies, we can workaround
    this limitation by reinterpret casting the `bfloat16` tensor to a `float16`
    tensor.

    Args:
        tensor: The `bfloat16` tensor from which to obtain memoryview.

    Returns:
        The class::`memoryview` of the input tensor.
    """
    if tensor.dtype != torch.bfloat16:
        raise ValueError(
            "The input tensor must have be of type torch.bfloat16 "
            f"(got {tensor.dtype})."
        )
    untyped_storage = tensor.storage()._untyped()
    tensor = torch.empty(tensor.size(), dtype=torch.float16)
    tensor.set_(untyped_storage)
    return memoryview(tensor.numpy())


def tensor_from_memoryview(
    mv: memoryview, dtype: torch.dtype, shape: List[int]
) -> torch.Tensor:
    return torch.reshape(torch.frombuffer(mv, dtype=dtype), shape)


def torch_save_as_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def torch_load_from_bytes(buf: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(buf))
