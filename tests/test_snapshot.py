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
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_state_dict(self) -> None:
        foo = torchsnapshot.StateDict(a=torch.rand(40, 40), b=torch.rand(40, 40), c=42)
        bar = torchsnapshot.StateDict(a=torch.rand(40, 40), b=torch.rand(40, 40), c=43)
        self.assertFalse(check_state_dict_eq(foo.state_dict(), bar.state_dict()))

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
