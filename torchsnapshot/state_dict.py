#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import UserDict
from typing import Any, Dict


# pyre-fixme[24]: Python <3.9 doesn't support typing on UserDict
class StateDict(UserDict):
    """
    A dictionary that exposes ``.state_dict()`` and ``.load_state_dict()``
    methods.

    It can be used to capture objects that do not expose ``.state_dict()`` and
    ``.load_state_dict()`` methods (e.g. Tensors, Python primitive types) as
    part of the application state.
    """

    def state_dict(self) -> Dict[str, Any]:
        return self.data

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.data.update(state_dict)
