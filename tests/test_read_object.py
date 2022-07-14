#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torchsnapshot


class ReadObjectTest(unittest.TestCase):
    def test_read_object(self) -> None:
        state = torchsnapshot.StateDict(
            foo=42,
            bar=torch.randn(20, 20),
        )

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(
                path=path, app_state={"state": state}
            )

            self.assertEqual(snapshot.read_object("0/state/foo"), 42)
            self.assertEqual(snapshot.read_object("0/state/foo", 777), 42)

            baz = torch.randn(20, 20)
            self.assertFalse(torch.allclose(baz, state["bar"]))

            loaded_bar = snapshot.read_object("0/state/bar", baz)
            self.assertEqual(id(loaded_bar), id(baz))
            self.assertNotEqual(id(loaded_bar), id(state["bar"]))
            self.assertTrue(torch.allclose(baz, state["bar"]))
