#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from collections import OrderedDict

from torchsnapshot.flatten import flatten, inflate
from torchsnapshot.manifest import DictEntry, ListEntry, OrderedDictEntry


# pyre-fixme[5]: Global expression must be annotated.
_OBJ = {
    "foo": 0,
    "bar": 1,
    "baz": [
        2,
        3,
        {"qux": 4, "quxx": [5, OrderedDict(quuz=6, corge=[7, 8, 9])]},
    ],
}
_EXPECTED_MANIFEST = {
    "": DictEntry(keys=["foo", "bar", "baz"]),
    "baz": ListEntry(),
    "baz/2": DictEntry(keys=["qux", "quxx"]),
    "baz/2/quxx": ListEntry(),
    "baz/2/quxx/1": OrderedDictEntry(keys=["quuz", "corge"]),
    "baz/2/quxx/1/corge": ListEntry(),
}
_EXPECTED_FLATTENED = {
    "foo": 0,
    "bar": 1,
    "baz/0": 2,
    "baz/1": 3,
    "baz/2/qux": 4,
    "baz/2/quxx/0": 5,
    "baz/2/quxx/1/quuz": 6,
    "baz/2/quxx/1/corge/0": 7,
    "baz/2/quxx/1/corge/1": 8,
    "baz/2/quxx/1/corge/2": 9,
}


class FlattenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_flatten(self) -> None:
        manifest, flattened = flatten(obj=_OBJ)
        self.assertDictEqual(manifest, _EXPECTED_MANIFEST)
        self.assertDictEqual(flattened, _EXPECTED_FLATTENED)

    def test_inflate(self) -> None:
        manifest, flattened = flatten(obj=_OBJ)
        inflated = inflate(manifest, flattened)
        self.assertDictEqual(inflated, _OBJ)

    def test_flatten_with_prefix(self) -> None:
        manifest, flattened = flatten(obj=_OBJ, prefix="my/prefix")
        expected_manifest = {
            os.path.join("my/prefix", path) if path else "my/prefix": entry
            for path, entry in _EXPECTED_MANIFEST.items()
        }
        expected_flattened = {
            os.path.join("my/prefix", path): entry
            for path, entry in _EXPECTED_FLATTENED.items()
        }
        self.assertDictEqual(manifest, expected_manifest)
        self.assertDictEqual(flattened, expected_flattened)

    def test_inflate_with_prefix(self) -> None:
        manifest, flattened = flatten(obj=_OBJ, prefix="my/prefix")
        inflated = inflate(manifest, flattened, prefix="my/prefix")
        self.assertDictEqual(inflated, _OBJ)

    def test_keys_with_colliding_str_repr(self) -> None:
        """
        When there's a collision among the string representations of the keys,
        the dict should not be flattened.
        """
        OBJ = {"0": {"1": "foo", 1: "bar"}, 0: "baz"}
        EXPECTED_MANIFEST = {}
        EXPECTED_FLATTENED = {"": {"0": {"1": "foo", 1: "bar"}, 0: "baz"}}

        manifest, flattened = flatten(obj=OBJ)
        self.assertDictEqual(manifest, EXPECTED_MANIFEST)
        self.assertDictEqual(flattened, EXPECTED_FLATTENED)

        inflated = inflate(manifest, flattened)
        self.assertDictEqual(inflated, OBJ)

    def test_keys_of_mixed_types(self) -> None:
        """
        A dict with keys of mixed types can be flattened as long as there's no
        collision among the string representation of the keys.
        """
        OBJ = {0: {"0": "foo", 1: "bar"}, "1": "baz"}
        EXPECTED_MANIFEST = {
            # pyre-fixme[6]: For 1st param expected `List[str]` but got
            #  `List[Union[int, str]]`.
            "": DictEntry(keys=[0, "1"]),
            # pyre-fixme[6]: For 1st param expected `List[str]` but got
            #  `List[Union[int, str]]`.
            "0": DictEntry(keys=["0", 1]),
        }
        EXPECTED_FLATTENED = {"0/0": "foo", "0/1": "bar", "1": "baz"}

        manifest, flattened = flatten(obj=OBJ)
        self.assertDictEqual(manifest, EXPECTED_MANIFEST)
        self.assertDictEqual(flattened, EXPECTED_FLATTENED)

        inflated = inflate(manifest, flattened)
        self.assertDictEqual(inflated, OBJ)
