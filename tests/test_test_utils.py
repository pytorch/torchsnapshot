#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq


class TestUtilsTest(unittest.TestCase):
    """
    Watch the watchmen.
    """

    def test_assert_state_dict_eq(self) -> None:
        t0 = torch.rand(16, 16)
        t1 = torch.rand(16, 16)
        a = {"foo": t0, "bar": [t1], "baz": 42}
        b = {"foo": t0, "bar": [t1], "baz": 42}
        c = {"foo": t0, "bar": [t0], "baz": 42}
        d = {"foo": t1, "bar": [t1], "baz": 42}
        e = {"foo": t0, "bar": [t1], "baz": 43}

        assert_state_dict_eq(self, a, b)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, c)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, d)
        with self.assertRaises(AssertionError):
            assert_state_dict_eq(self, a, e)

    def test_check_state_dict_eq(self) -> None:
        t0 = torch.rand(16, 16)
        t1 = torch.rand(16, 16)
        a = {"foo": t0, "bar": [t1], "baz": 42}
        b = {"foo": t0, "bar": [t1], "baz": 42}
        c = {"foo": t0, "bar": [t0], "baz": 42}
        d = {"foo": t1, "bar": [t1], "baz": 42}
        e = {"foo": t0, "bar": [t1], "baz": 43}

        self.assertTrue(check_state_dict_eq(a, b))
        self.assertFalse(check_state_dict_eq(a, c))
        self.assertFalse(check_state_dict_eq(a, d))
        self.assertFalse(check_state_dict_eq(a, e))
