#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from typing import cast, Dict

import torch
import torchsnapshot
from torchsnapshot import Stateful


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foo = torch.nn.Parameter(torch.randn(20, 20))


class MyStateful(Stateful):
    def __init__(self) -> None:
        self.foo = 1
        self.bar = "bar"

    def state_dict(self) -> Dict[str, object]:
        return {"foo": self.foo, "bar": self.bar}

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.foo = cast(int, state_dict["foo"])
        self.bar = cast(str, state_dict["bar"])


class StateDictTest(unittest.TestCase):
    def test_get_state_dict(self) -> None:
        my_module = MyModule()
        with tempfile.TemporaryDirectory() as path:
            torchsnapshot.Snapshot.take(
                path=path,
                app_state={"my_module": my_module},
            )
            snapshot = torchsnapshot.Snapshot(path)
            state_dict = snapshot.get_state_dict_for_key("my_module")
            self.assertTrue(torch.allclose(state_dict["foo"], my_module.foo))

    def test_get_state_dict_with_invalid_key(self) -> None:
        my_module = MyModule()
        with tempfile.TemporaryDirectory() as path:
            torchsnapshot.Snapshot.take(
                path=path,
                app_state={"my_module": my_module},
            )
            snapshot = torchsnapshot.Snapshot(path)
            with self.assertRaisesRegex(
                AssertionError, "is absent in both manifest and flattened"
            ):
                snapshot.get_state_dict_for_key("invalid_key")

    def test_generic_stateful(self) -> None:
        my_stateful = MyStateful()
        my_stateful.foo = 2
        my_stateful.bar = "baz"
        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot(path)
            snapshot.take(path, app_state={"my_stateful": my_stateful})
            state_dict = snapshot.get_state_dict_for_key("my_stateful")
            self.assertDictEqual(state_dict, my_stateful.state_dict())
