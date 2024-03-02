#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
import unittest

import torch
from torchsnapshot.rss_profiler import measure_rss_deltas


class RSSProfilerTest(unittest.TestCase):
    def test_rss_profiler(self) -> None:
        rss_deltas = []
        with measure_rss_deltas(rss_deltas=rss_deltas):
            torch.randn(5000, 5000)
            time.sleep(2)
