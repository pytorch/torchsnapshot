#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict

import torch

DDP_STATE_DICT_PREFIX: str = "module."


class DistributedDataParallelAdapter:
    """
    A convenience class to load a module's state dict saved from a DistributedDataParallel-wrapped module into a module that is not wrapped with DDP.

    Example::

        >>> module = torch.nn.Linear(2, 2)
        >>> ddp_module = DistributedDataParallel(module)
        >>> Snapshot.take(
        >>>     path="foo/bar",
        >>>     app_state={"module": ddp_module},
        >>> )

        >>> # Restore the state
        >>> snapshot = Snapshot(path="foo/bar")
        >>> adapter = DistributedDataParallelAdapter(module)
        >>> snapshot.restore({"module": adapter})
        >>> module = adapter.module
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, DDP_STATE_DICT_PREFIX
        )
        self.module.load_state_dict(state_dict)
