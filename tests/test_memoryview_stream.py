#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest

import torch
from torchsnapshot.memoryview_stream import MemoryviewStream


class MemoryviewStreamTest(unittest.TestCase):
    def test_memoryview_stream(self) -> None:
        tensor = torch.rand(1000)
        mv = memoryview(tensor.numpy()).cast("b")
        self.assertEqual(len(mv), 4000)

        mvs = MemoryviewStream(mv=mv)
        bio = io.BytesIO(mv.tobytes())

        self.assertTrue(mvs.readable())
        self.assertTrue(bio.readable())

        self.assertTrue(mvs.seekable())
        self.assertTrue(bio.seekable())

        buf = bytes(mvs.read(20))
        self.assertEqual(len(buf), 20)
        self.assertEqual(buf, bio.read(20))

        pos = mvs.tell()
        self.assertEqual(pos, 20)
        self.assertEqual(pos, bio.tell())

        pos = mvs.seek(500)
        self.assertEqual(pos, 500)
        self.assertEqual(pos, bio.seek(500))

        buf = bytes(mvs.read(20))
        self.assertEqual(len(buf), 20)
        self.assertEqual(buf, bio.read(20))

        pos = mvs.tell()
        self.assertEqual(pos, 520)
        self.assertEqual(pos, bio.tell())

        buf = bytes(mvs.read(4000))
        self.assertEqual(len(buf), 3480)
        self.assertEqual(buf, bio.read(4000))

        pos = mvs.tell()
        self.assertEqual(pos, 4000)
        self.assertEqual(pos, bio.tell())

        pos = mvs.seek(0)
        self.assertEqual(pos, 0)
        self.assertEqual(pos, bio.seek(0))

        buf = bytes(mvs.read(4500))
        self.assertEqual(len(buf), 4000)
        self.assertEqual(buf, bio.read(4500))
