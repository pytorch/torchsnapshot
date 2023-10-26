#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torchsnapshot.manifest import (
    ChunkedTensorEntry,
    DictEntry,
    DTensorEntry,
    Entry,
    ObjectEntry,
    Shard,
    ShardedTensorEntry,
    TensorEntry,
)
from torchsnapshot.manifest_utils import is_fully_replicated_entry

_WORLD_SIZE = 4

_MANIFEST_0: Dict[str, Entry] = {
    "0/foo": DictEntry(
        # who comes up with these names?
        keys=[
            "bar",
            "baz",
            "qux",
            "quux",
            "qux_chunked",
            "quux_chunked",
            "corge",
            "grault",
        ]
    ),
    "0/foo/bar": ObjectEntry(
        location="0/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "0/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "0/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[0, 0],
                sizes=[2, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 4],
                    replicated=False,
                ),
            )
        ]
    ),
    "0/foo/quux": TensorEntry(
        location="0/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "0/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "0/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="0/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="0/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
    # DTensor with sharding only - global shape (7,8)
    "0/foo/corge": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 5],
                tensor=TensorEntry(
                    location="sharded/foo/corge.0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 5],
                    replicated=False,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 0 sharded across mesh dim 0, tensor dim 1 across mesh dim 1
        dim_map=[[0], [1]],
    ),
    # DTensor with sharding + replication - global shape (7,8)
    "0/foo/grault": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 0],
                sizes=[7, 5],
                tensor=TensorEntry(
                    location="replicated_sharded/foo/grault.0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[7, 5],
                    replicated=True,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 1 sharded across mesh dim 0, tensor dim 0 replicated (inferred to be mesh dim 1)
        dim_map=[[-1], [0]],
    ),
    "1/foo": DictEntry(
        keys=[
            "bar",
            "baz",
            "qux",
            "quux",
            "qux_chunked",
            "quux_chunked",
            "corge",
            "grault",
        ]
    ),
    "1/foo/bar": ObjectEntry(
        location="1/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "1/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "1/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[0, 4],
                sizes=[2, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.1",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 4],
                    replicated=False,
                ),
            )
        ]
    ),
    "1/foo/quux": TensorEntry(
        location="1/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "1/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "1/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="1/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="1/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
    # DTensor with sharding only - global shape (7,8)
    "1/foo/corge": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 5],
                sizes=[5, 3],
                tensor=TensorEntry(
                    location="sharded/foo/corge.1",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 3],
                    replicated=False,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 0 sharded across mesh dim 0, tensor dim 1 across mesh dim 1
        dim_map=[[0], [1]],
    ),
    # DTensor with sharding + replication - global shape (7,8)
    "1/foo/grault": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 0],
                sizes=[7, 5],
                tensor=TensorEntry(
                    location="replicated_sharded/foo/grault.1",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[7, 5],
                    replicated=True,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 1 sharded across mesh dim 0, tensor dim 0 replicated (inferred to be mesh dim 1)
        dim_map=[[-1], [0]],
    ),
    "2/foo": DictEntry(
        keys=[
            "bar",
            "baz",
            "qux",
            "quux",
            "qux_chunked",
            "quux_chunked",
            "corge",
            "grault",
        ]
    ),
    "2/foo/bar": ObjectEntry(
        location="2/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "2/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "2/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[2, 0],
                sizes=[2, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.2",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 4],
                    replicated=False,
                ),
            )
        ]
    ),
    "2/foo/quux": TensorEntry(
        location="2/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "2/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "2/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="2/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="2/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
    # DTensor with sharding only - global shape (7,8)
    "2/foo/corge": DTensorEntry(
        shards=[
            Shard(
                offsets=[5, 0],
                sizes=[2, 5],
                tensor=TensorEntry(
                    location="sharded/foo/corge.2",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 5],
                    replicated=False,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 0 sharded across mesh dim 0, tensor dim 1 across mesh dim 1
        dim_map=[[0], [1]],
    ),
    # DTensor with sharding + replication - global shape (7,8)
    "2/foo/grault": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 5],
                sizes=[7, 3],
                tensor=TensorEntry(
                    location="replicated_sharded/foo/grault.2",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[7, 3],
                    replicated=True,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 1 sharded across mesh dim 0, tensor dim 0 replicated (inferred to be mesh dim 1)
        dim_map=[[-1], [0]],
    ),
    "3/foo": DictEntry(
        keys=[
            "bar",
            "baz",
            "qux",
            "quux",
            "qux_chunked",
            "quux_chunked",
            "corge",
            "grault",
        ]
    ),
    "3/foo/bar": ObjectEntry(
        location="3/foo/bar", serializer="torch_save", obj_type="Bar", replicated=False
    ),
    "3/foo/baz": ObjectEntry(
        location="replicated/foo/baz",
        serializer="torch_save",
        obj_type="Baz",
        replicated=True,
    ),
    "3/foo/qux": ShardedTensorEntry(
        shards=[
            Shard(
                offsets=[2, 4],
                sizes=[2, 4],
                tensor=TensorEntry(
                    location="sharded/foo/qux.3",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 4],
                    replicated=False,
                ),
            )
        ]
    ),
    "3/foo/quux": TensorEntry(
        location="3/foo/quux",
        serializer="torch_save",
        dtype="float32",
        shape=[128, 128],
        replicated=False,
    ),
    "3/foo/qux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[7, 10],
        chunks=[
            Shard(
                offsets=[0, 0],
                sizes=[5, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_0_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[5, 10],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[5, 0],
                sizes=[2, 10],
                tensor=TensorEntry(
                    location="replicated/foo/qux_chunked_5_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 10],
                    replicated=False,
                ),
            ),
        ],
        replicated=True,
    ),
    "3/foo/quux_chunked": ChunkedTensorEntry(
        dtype="float32",
        shape=[100],
        chunks=[
            Shard(
                offsets=[0],
                sizes=[50],
                tensor=TensorEntry(
                    location="3/foo/qux_chunked_0",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
            Shard(
                offsets=[50],
                sizes=[50],
                tensor=TensorEntry(
                    location="3/foo/qux_chunked_50",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[50],
                    replicated=False,
                ),
            ),
        ],
        replicated=False,
    ),
    # DTensor with sharding only - global shape (7,8)
    "3/foo/corge": DTensorEntry(
        shards=[
            Shard(
                offsets=[5, 5],
                sizes=[2, 3],
                tensor=TensorEntry(
                    location="sharded/foo/corge.3",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[2, 3],
                    replicated=False,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 0 sharded across mesh dim 0, tensor dim 1 across mesh dim 1
        dim_map=[[0], [1]],
    ),
    # DTensor with sharding + replication - global shape (7,8)
    "3/foo/grault": DTensorEntry(
        shards=[
            Shard(
                offsets=[0, 5],
                sizes=[7, 3],
                tensor=TensorEntry(
                    location="replicated_sharded/foo/grault.3",
                    serializer="torch_save",
                    dtype="float32",
                    shape=[7, 3],
                    replicated=True,
                ),
            ),
        ],
        mesh=[[0, 1], [2, 3]],
        # Tensor dim 1 sharded across mesh dim 0, tensor dim 0 replicated (inferred to be mesh dim 1)
        dim_map=[[-1], [0]],
    ),
}

# Same as _MANIFEST_0 expect that replicated entries only exist on rank 0
_MANIFEST_1: Dict[str, Entry] = {
    k: v
    for k, v in _MANIFEST_0.items()
    if not (k.startswith("1/") and is_fully_replicated_entry(v))
}

_MANIFEST_2: Dict[str, Entry] = {
    k: v
    for k, v in _MANIFEST_0.items()
    if not (k.startswith("2/") and is_fully_replicated_entry(v))
}

_MANIFEST_3: Dict[str, Entry] = {
    k: v
    for k, v in _MANIFEST_0.items()
    if not (k.startswith("3/") and is_fully_replicated_entry(v))
}
