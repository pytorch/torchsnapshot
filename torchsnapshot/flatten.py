#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from collections import OrderedDict
from typing import Any, Dict, Tuple

from .manifest import DictEntry, ListEntry, Manifest, OrderedDictEntry


# pyre-ignore[2]: obj can have arbitrary type
def flatten(obj: Any, prefix: str = "") -> Tuple[Manifest, Dict[str, Any]]:
    """
    Recursively flatten a container in a reversible manner.

    Args:
        obj: The collection to flatten.
        prefix: The prefix to prepend to the keys of returned dictionaries.

    Returns:
        The flattened container and the manifest needed for inflating the
        container.

    ::
        >>> collection = {'foo': [1, 2, OrderedDict(bar=3, baz=4)]}]}
        >>> manifest, flattened = flatten(collection, prefix='my/prefix')
        >>> manifest
        {
            "my/prefix": {"type": "dict"},
            "my/prefix/foo": {"type": "list"},
            "my/prefix/foo/2": {"type": "OrderedDict", "keys": ["bar", "baz"]},
        }
        >>> flattened
        {
            "my/prefix/foo/0": 1,
            "my/prefix/foo/1": 2,
            "my/prefix/foo/2/bar": 3,
            "my/prefix/foo/2/baz": 4,
        }
    """
    manifest = {}
    flattened = {}
    if type(obj) == list:
        manifest[prefix] = ListEntry()
        for idx, elem in enumerate(obj):
            path = os.path.join(prefix, str(idx))
            m, f = flatten(elem, path)
            manifest.update(m)
            flattened.update(f)
    elif type(obj) in (dict, OrderedDict) and _should_flatten_dict(obj):
        if type(obj) == dict:
            manifest[prefix] = DictEntry(keys=list(obj.keys()))
        else:
            manifest[prefix] = OrderedDictEntry(keys=list(obj.keys()))
        for key, elem in obj.items():
            path = os.path.join(prefix, str(key))
            m, f = flatten(elem, path)
            manifest.update(m)
            flattened.update(f)
    else:
        flattened[prefix] = obj
    return manifest, flattened


# pyre-ignore[3]: Return annotation cannot be `Any`
def inflate(manifest: Manifest, flattened: Dict[str, Any], prefix: str = "") -> Any:
    """
    The reverse operation of func::`flatten`.

    Args:
        manifest: The container manifest returned by func::`flatten`.
        flattened: The flattened container.
        prefix: The path to the outermost container.

    Returns:
        The inflated container.
    """
    for path in itertools.chain(manifest.keys(), flattened.keys()):
        if not path.startswith(prefix):
            raise RuntimeError(f"{path} does not start with {prefix}")

    combined = {}
    for path, entry in manifest.items():
        if isinstance(entry, ListEntry):
            container = []
        elif isinstance(entry, DictEntry):
            container = dict.fromkeys(entry.keys)
        elif isinstance(entry, OrderedDictEntry):
            container = OrderedDict.fromkeys(entry.keys)
        else:
            raise RuntimeError(
                f"Unrecognized container entry type: {type(entry)} ({entry.type})."
            )
        trimmed_path = "/" + path[len(prefix) :]
        combined[trimmed_path] = container

    for path, obj in flattened.items():
        trimmed_path = "/" + path[len(prefix) :]
        combined[trimmed_path] = obj

    combined = OrderedDict(sorted(combined.items()))
    for path, val in combined.items():
        if path == "/":
            continue
        tokens = path.split("/")
        dir_path = "/".join(tokens[:-1]) or "/"
        if dir_path not in combined:
            raise RuntimeError(f'Container entry is absent for "{dir_path}"')
        container = combined[dir_path]
        if type(container) == list:
            container.append(val)
        elif type(container) in (dict, OrderedDict):
            key = tokens[-1]
            if key in container:
                container[key] = val
            elif _check_int(key):
                container[int(key)] = val
            else:
                raise AssertionError(f"Item {path} is not listed in the manifest.")

    return combined["/"]


# pyre-ignore [2]: Parameter `d` must have a type that does not contain `Any`.
def _should_flatten_dict(d: Dict[Any, Any]) -> bool:
    """
    Determine if a dict should be flattened.

    A dict shouldn't be flattened if:
    - There is a collision among the string representations of the keys.
    - The keys contain objects other than string or int.
    """
    if not all(isinstance(k, (str, int)) for k in d.keys()):
        return False
    if len({str(k) for k in d.keys()}) < len(d):
        return False
    return True


def _check_int(s: str) -> bool:
    if s.isdigit():
        return True
    elif len(s) > 1 and s[0] in ("-", "+"):
        return s[1:].isdigit()
    else:
        return False
