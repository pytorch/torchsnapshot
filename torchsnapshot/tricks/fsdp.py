#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class FSDPOptimizerAdapter:
    """
    Wrapper for FSDP optimizer to call specific FSDP optimizer state checkpointing APIs.

    Example::

        >>> module = torch.nn.Linear(2, 2)
        >>> fsdp_module = FullyShardedDataParallel(module)
        >>> optimizer = torch.optim.SGD(fsdp_module.parameters(), lr=0.1)
        >>> Snapshot.take(
        >>>     path="foo/bar",
        >>>     app_state={"module": fsdp_module, "optim": FSDPOptimizerAdapter(fsdp_module, optimizer)},
        >>> )

        >>> # Restore the state
        >>> snapshot = Snapshot(path="foo/bar")
        >>> module = torch.nn.Linear(2, 2)
        >>> fsdp_module = FullyShardedDataParallel(module)
        >>> optimizer = torch.optim.SGD(fsdp_module.parameters(), lr=0.1)
        >>> adapter = FSDPOptimizerAdapter(module)
        >>> snapshot.restore({"module": module, "optim": FSDPOptimizerAdapter(fsdp_module, optimizer)})
    """

    def __init__(self, module: FSDP, optimizer: torch.optim.Optimizer) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        optim_state_dict = FSDP.optim_state_dict(self.module, self.optimizer)
        return optim_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optim_state_dict = FSDP.optim_state_dict_to_load(
            self.module, self.optimizer, state_dict
        )
        self.optimizer.load_state_dict(optim_state_dict)
