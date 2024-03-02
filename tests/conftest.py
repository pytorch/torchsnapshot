#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Generator

import pytest
from _pytest.fixtures import SubRequest  # @manual
from torchsnapshot.knobs import override_is_batching_disabled


@pytest.fixture(params=["batching_on", "batching_off"])
def toggle_batching(request: SubRequest) -> Generator[None, None, None]:
    with override_is_batching_disabled(request.param == "batching_off"):
        yield
