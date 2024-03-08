#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
from typing import Any, Dict, Optional

from importlib_metadata import entry_points

from .io_types import StoragePlugin
from .storage_plugins.fs import FSStoragePlugin
from .storage_plugins.s3 import S3StoragePlugin


def url_to_storage_plugin(
    url_path: str, storage_options: Optional[Dict[str, Any]] = None
) -> StoragePlugin:
    """
    Initialize storage plugin from url path.

    Args:
        url_path: The url path following the pattern [protocol]://[path].
            The protocol defaults to `fs` if unspecified.
        storage_options: Additional keyword options for the StoragePlugin to use.
            See each StoragePlugin's documentation for customizations.

    Returns:
        The initialized storage plugin.
    """
    if "://" in url_path:
        protocol, path = url_path.split("://", 1)
        if len(protocol) == 0:
            protocol = "fs"
    else:
        protocol, path = "fs", url_path

    if storage_options is None:
        storage_options = {}

    # Built-in storage plugins
    if protocol == "fs":
        return FSStoragePlugin(root=path, storage_options=storage_options)
    elif protocol == "s3":
        return S3StoragePlugin(root=path, storage_options=storage_options)
    elif protocol == "gs":
        from torchsnapshot.storage_plugins.gcs import GCSStoragePlugin

        return GCSStoragePlugin(root=path, storage_options=storage_options)

    # Registered storage plugins
    eps = entry_points(group="storage_plugins")
    registered_plugins = {ep.name: ep for ep in eps}
    if protocol in registered_plugins:
        entry = registered_plugins[protocol]
        factory = entry.load()
        plugin = factory(path, storage_options)
        if not isinstance(plugin, StoragePlugin):
            raise RuntimeError(
                f"The factory function for {protocol} ({entry.value}) "
                "did not return a StorgePlugin object."
            )
        return plugin
    else:
        raise RuntimeError(f"Unsupported protocol: {protocol}.")


def url_to_storage_plugin_in_event_loop(
    url_path: str,
    event_loop: asyncio.AbstractEventLoop,
    storage_options: Optional[Dict[str, Any]] = None,
) -> StoragePlugin:
    async def _url_to_storage_plugin() -> StoragePlugin:
        return url_to_storage_plugin(url_path=url_path, storage_options=storage_options)

    return event_loop.run_until_complete(_url_to_storage_plugin())
