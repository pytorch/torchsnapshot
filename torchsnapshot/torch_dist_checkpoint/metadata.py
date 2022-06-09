# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]

# Open issues/questions that will need to be addressed
#
# 0. Please note that this is not at all the final version,
# everything is fluid and can be changed
#
# 1. We will need a metadata class for regular tensor, no point forcing it
# into a ShardedTensorMetadata.
#
# 2. This code is using pickle for serialization, we need to replace the
# serialization with something more reasonable e.g. flatbuffer before any
# serious use case.
#
# 3. To get thing moving faster, I am using tensor as the "unit" for read/write
# request. The ideal solution we have is to use a buffer instead. To make a
# decision, we will need to fully understand the performance
# implication for model store


@dataclass
class StorageMetadata:

    shard_metadata: Optional[ShardMetadata]
    # Unique identifier for this particular entity (Tensor or Shard of ShardedTensor)
    storage_key: str
    length: int
    offset: int


# Metadata for each param.
@dataclass
class ExtendedTensorMetadata:
    # Details of Tensor/ShardedTensor (dtype, shape, sharding config etc.)
    # TODO: It might not make sense to force Tensor's metadata into
    # ShardedTensorMetadata, let's create a metadata class for regular tensor
    tensor_metadata: ShardedTensorMetadata
    storage_metadata: List[StorageMetadata]


@dataclass
class Metadata:
    # Metadata for the state dict.
    # TODO, use pickle for this quick hack, must replace it with something else e.g. flatbuffer
    # before serious use case,
    state_dict_metadata: Dict[str, ExtendedTensorMetadata]

    def __getstate__(self) -> bytes:
        serialized = pickle.dumps(self.state_dict_metadata)
        return serialized

    def __setstate__(self, state: bytes) -> None:
        self.state_dict_metadata = pickle.loads(state)


@dataclass
class BytesWriteRequest:
    bytes: io.BytesIO
    storage_key: str


@dataclass
class BytesReadRequest:
    bytes: io.BytesIO
    storage_key: str


@dataclass
class TensorWriteRequest:
    tensor: torch.Tensor
    storage_key: str


@dataclass
class TensorReadRequest:
    tensor: torch.Tensor
    storage_key: str
    # offset and length to read/write w.r.t. to the storage identified by ``storage_key``
    offsets: Tuple[int, ...]
    lengths: Tuple[int, ...]
