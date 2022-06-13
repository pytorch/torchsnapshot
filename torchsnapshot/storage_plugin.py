#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib_metadata import entry_points

from .io_types import StoragePlugin
from .storage_plugins.fs import FSStoragePlugin
from .storage_plugins.gcs import GCSStoragePlugin
from .storage_plugins.s3 import S3StoragePlugin


def url_to_storage_plugin(url_path: str) -> StoragePlugin:
    """
    Initialize storage plugin from url path.

    Args:
        url_path: The url path following the pattern [protocol]://[path].
            The protocol defaults to `fs` if unspecified.

    Returns:
        The initialized storage plugin.
    """
    if "://" in url_path:
        protocol, path = url_path.split("://", 1)
        if len(protocol) == 0:
            protocol = "fs"
    else:
        protocol, path = "fs", url_path

    # Built-in storage plugins
    if protocol == "fs":
        return FSStoragePlugin(root=path)
    elif protocol == "s3":
        return S3StoragePlugin(root=path)
    elif protocol == "gs":
        return GCSStoragePlugin(root=path)

    # Registered storage plugins
    registered_plugins = {
        ep.name: ep for ep in entry_points().get("storage_plugins", [])
    }
    if protocol in registered_plugins:
        entry = registered_plugins[protocol]
        factory = entry.load()
        plugin = factory(path)
        if not isinstance(plugin, StoragePlugin):
            raise RuntimeError(
                f"The factory function for {protocol} ({entry.value}) "
                "did not return a StorgePlugin object."
            )
        return plugin
    else:
        raise RuntimeError(f"Unsupported protocol: {protocol}.")
