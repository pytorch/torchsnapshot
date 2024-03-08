#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, TypeVar

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


T = TypeVar("T", bound=Stateful)
AppState = Dict[str, T]
