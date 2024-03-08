#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from typing import Any, Dict

import torch
import torchsnapshot


class StatefulWithRNGSideEffect:
    def state_dict(self) -> Dict[str, Any]:
        torch.rand([2])
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        torch.rand([3])


class RNGStateTest(unittest.TestCase):
    def test_rng_state(self) -> None:
        app_state = {
            "rng_state": torchsnapshot.RNGState(),
            "effectful": StatefulWithRNGSideEffect(),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot = torchsnapshot.Snapshot.take(path=tmp_dir, app_state=app_state)
            after_take = torch.rand(1)
            snapshot.restore(app_state)
            after_restore = torch.rand(1)
            torch.testing.assert_close(after_take, after_restore)
