#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"A lightweight library for adding fault tolerance to large-scale PyTorch distributed training workloads"

from .rng_state import RNGState
from .snapshot import Snapshot
from .state_dict import StateDict
from .stateful import Stateful
from .version import __version__


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        return (
            get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # Jupyter
            or get_ipython().__class__.__module__ == "google.colab._shell"  # Colab
        )
    except ImportError:
        return False


# https://github.com/jupyter/notebook/issues/3397
if _is_notebook():
    # @manual=fbsource//third-party/pypi/nest-asyncio:nest-asyncio
    import nest_asyncio  # lint-fixme: DisallowNestAsyncio

    nest_asyncio.apply()

__all__ = [
    "__version__",
    "Snapshot",
    "Stateful",
    "StateDict",
    "RNGState",
]
