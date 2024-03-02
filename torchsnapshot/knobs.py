#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations
import os
from contextlib import contextmanager
from typing import Any, Generator

# This file contains various non-user facing constants used throughout the
# project, and utilities for overriding the constants for testing and debugging
# purposes. Environment variable is chosen as the overriding mechanism since it
# works well for unit tests, e2e tests, and real world use cases. Sometimes it
# makes sense for the the function consuming one of these constants to also
# allow overriding it via function argument for the ease of unit testing. In
# such cases, the convention is to let the function argument take precedence.

_MAX_CHUNK_SIZE_ENV_VAR = "TORCHSNAPSHOT_MAX_CHUNK_SIZE_BYTES_OVERRIDE"
_MAX_SHARD_SIZE_ENV_VAR = "TORCHSNAPSHOT_MAX_SHARD_SIZE_BYTES_OVERRIDE"
_SLAB_SIZE_THRESHOLD_ENV_VAR = "TORCHSNAPSHOT_SLAB_SIZE_THRESHOLD_BYTES_OVERRIDE"
_MAX_PER_RANK_IO_CONCURRENCY_ENV_VAR = (
    "TORCHSNAPSHOT_MAX_PER_RANK_IO_CONCURRENCY_OVERRIDE"
)

_DEFAULT_MAX_CHUNK_SIZE_BYTES: int = 512 * 1024 * 1024
_DEFAULT_MAX_SHARD_SIZE_BYTES: int = 512 * 1024 * 1024
_DEFAULT_SLAB_SIZE_THRESHOLD_BYTES: int = 128 * 1024 * 1024
_DISABLE_BATCHING_ENV_VAR: str = "TORCHSNAPSHOT_DISABLE_BATCHING"
_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ENV_VAR: str = (
    "TORCHSNAPSHOT_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ONLY"
)

_DEFAULT_MAX_PER_RANK_IO_CONCURRENCY: int = 16


def get_max_chunk_size_bytes() -> int:
    override = os.environ.get(_MAX_CHUNK_SIZE_ENV_VAR)
    if override is not None:
        return int(override)
    return _DEFAULT_MAX_CHUNK_SIZE_BYTES


def get_max_shard_size_bytes() -> int:
    override = os.environ.get(_MAX_SHARD_SIZE_ENV_VAR)
    if override is not None:
        return int(override)
    return _DEFAULT_MAX_SHARD_SIZE_BYTES


def get_slab_size_threshold_bytes() -> int:
    override = os.environ.get(_SLAB_SIZE_THRESHOLD_ENV_VAR)
    if override is not None:
        return int(override)
    return _DEFAULT_SLAB_SIZE_THRESHOLD_BYTES


def get_max_per_rank_io_concurrency() -> int:
    override = os.environ.get(_MAX_PER_RANK_IO_CONCURRENCY_ENV_VAR)
    if override is not None:
        return int(override)
    return _DEFAULT_MAX_PER_RANK_IO_CONCURRENCY


def is_batching_disabled() -> bool:
    if os.getenv(_DISABLE_BATCHING_ENV_VAR, "False").lower() in ("true", "1"):
        return True
    return False


def is_sharded_tensor_elasticity_enabled_at_root_only() -> bool:
    if os.getenv(_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ENV_VAR, "False").lower() in (
        "true",
        "1",
    ):
        return True
    return False


@contextmanager
def _override_env_var(env_var: str, value: Any) -> Generator[None, None, None]:
    prev = os.environ.get(env_var)
    os.environ[env_var] = str(value)
    yield
    if prev is None:
        del os.environ[env_var]
    else:
        os.environ[env_var] = prev


@contextmanager
def override_max_chunk_size_bytes(
    max_chunk_size_bytes: int,
) -> Generator[None, None, None]:
    with _override_env_var(_MAX_CHUNK_SIZE_ENV_VAR, max_chunk_size_bytes):
        yield


@contextmanager
def override_max_shard_size_bytes(
    max_shard_size_bytes: int,
) -> Generator[None, None, None]:
    with _override_env_var(_MAX_SHARD_SIZE_ENV_VAR, max_shard_size_bytes):
        yield


@contextmanager
def override_is_batching_disabled(disabled: bool) -> Generator[None, None, None]:
    with _override_env_var(_DISABLE_BATCHING_ENV_VAR, disabled):
        yield


@contextmanager
def override_slab_size_threshold_bytes(
    max_shard_size_bytes: int,
) -> Generator[None, None, None]:
    with _override_env_var(_MAX_SHARD_SIZE_ENV_VAR, max_shard_size_bytes):
        yield


@contextmanager
def override_max_per_rank_io_concurrency(
    max_per_rank_io_concurrency: int,
) -> Generator[None, None, None]:
    with _override_env_var(
        _MAX_PER_RANK_IO_CONCURRENCY_ENV_VAR, max_per_rank_io_concurrency
    ):
        yield
