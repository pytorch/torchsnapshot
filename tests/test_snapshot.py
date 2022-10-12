#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
import torchsnapshot
from torchsnapshot import Snapshot
from torchsnapshot.manifest import PrimitiveEntry
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_state_dict(self) -> None:
        foo = torchsnapshot.StateDict(
            {
                "a": torch.rand(40, 40),
                "b": torch.rand(40, 40),
                "c": 42,
                "d/e": 43,
                "[@x]->&y^%": {"(z)": 44},
            },
        )
        bar = torchsnapshot.StateDict(
            {
                "a": torch.rand(40, 40),
                "b": torch.rand(40, 40),
                "c": 42,
                "d/e": 43,
                "[@x]->&y^%": {"(z)": 44},
            },
        )
        self.assertFalse(check_state_dict_eq(foo.state_dict(), bar.state_dict()))
        self.assertTrue(type(foo.state_dict()) == dict)

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(path, {"foo": foo})
            snapshot.restore({"foo": bar})
            assert_state_dict_eq(self, foo.state_dict(), bar.state_dict())

    def test_nn_linear(self) -> None:
        foo = torch.nn.Linear(128, 64)
        bar = torch.nn.Linear(128, 64)
        self.assertFalse(check_state_dict_eq(foo.state_dict(), bar.state_dict()))

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(path, {"foo": foo})
            snapshot.restore({"foo": bar})
            assert_state_dict_eq(self, foo.state_dict(), bar.state_dict())

    def test_nn_sequential(self) -> None:
        foo = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        bar = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        self.assertFalse(check_state_dict_eq(foo.state_dict(), bar.state_dict()))

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(path, {"foo": foo})
            snapshot.restore({"foo": bar})
            assert_state_dict_eq(self, foo.state_dict(), bar.state_dict())

    def test_adagrad(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        optim = torch.optim.Adagrad(model.parameters(), lr=0.01)

        expected = copy.deepcopy(optim.state_dict())

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(path, {"optim": optim})
            snapshot.restore({"optim": optim})

        assert_state_dict_eq(self, optim.state_dict(), expected)

    def test_invalid_app_state(self) -> None:
        not_stateful = 1
        app_state = {"optim": not_stateful}

        with tempfile.TemporaryDirectory() as path:
            self.assertRaises(TypeError, torchsnapshot.Snapshot.take, path, app_state)

            snapshot = Snapshot(path)
            self.assertRaises(TypeError, snapshot.restore, app_state)

    def test_app_state_with_primitive_types(self) -> None:
        state = torchsnapshot.StateDict(
            int_key=100,
            float_key=3.14,
            str_key="some_string",
            bool_key=True,
            bytes_key=b"\x00\x10",
        )
        restored_state = torchsnapshot.StateDict(
            int_key=None,
            float_key=None,
            str_key=None,
            bool_key=None,
            bytes_key=None,
        )

        def _assert_primitive_entry_with_type(
            location_key: str, expected_type_name: str
        ) -> None:
            assert (
                isinstance(snapshot.metadata.manifest[location_key], PrimitiveEntry)
                and snapshot.metadata.manifest[location_key].type == expected_type_name
            )

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(path, {"key": state})

            _assert_primitive_entry_with_type("0/key/int_key", "int")
            _assert_primitive_entry_with_type("0/key/str_key", "str")
            _assert_primitive_entry_with_type("0/key/bool_key", "bool")
            _assert_primitive_entry_with_type("0/key/bytes_key", "bytes")
            _assert_primitive_entry_with_type("0/key/float_key", "float")

            assert snapshot.metadata.manifest["0/key/float_key"].readable == str(
                state["float_key"]
            )

            snapshot.restore({"key": restored_state})

            assert state == restored_state
