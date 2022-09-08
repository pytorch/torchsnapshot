#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from types import MethodType
from typing import Any, Dict

from deepspeed import DeepSpeedEngine, version
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from torchsnapshot import Snapshot, StateDict

logger = logging.getLogger(__name__)


def _save_zero_checkpoint(self, save_path: str, tag: str) -> None:
    app_state = {
        "optimizer": self.optimizer,
        "objects": StateDict(ds_config=self.config, ds_version=version),
    }
    Snapshot.async_take(path=save_path, app_state=app_state)
    # TODO: demonstrate how torchsnapshot can help with zero_to_fp32.py
    if self.global_rank == 0:
        self._copy_recovery_script(save_path)


class Zero3StateAdapter:
    """
    Adapts DeepSpeedZeroOptimizer_Stage3 to expose conventional .state_dict()
    and .load_state_dict().

    Usage:

    >>> app_state = {
    >>>     "optimizer": Zero3StateAdapter(zero3_optimizer),
    >>> }
    >>> Snapshot.take(path=path, app_state=app_state)
    """

    def __init__(
        self,
        optimizer: DeepSpeedZeroOptimizer_Stage3,
        load_optimizer_states: bool = True,
        load_from_fp32_weights: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.load_optimizer_state = load_optimizer_states
        self.load_from_fp32_weights = load_from_fp32_weights

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer._rigid_load_state_dict(
            state_dict=state_dict, load_optimizer_states=self.load_optimizer_state
        )
        if len(self.optimizer.persistent_parameters) > 0:
            self.optimizer.persistent_parameters[0].partition(
                self.optimizer.persistent_parameters
            )
            self.optimizer.persistent_parameters[0].all_gather(
                self.optimizer.persistent_parameters
            )


def _load_zero_checkpoint(
    self,
    load_dir: str,
    tag: str,
    load_optimizer_states: bool = True,
) -> bool:
    snapshot = Snapshot(path=load_dir)
    app_state = {
        "optimizer": Zero3StateAdapter(
            optimizer=self.optimizer,
            load_optimizer_states=load_optimizer_states,
            load_from_fp32_weights=self.zero_load_from_fp32_weights(),
        )
    }
    snapshot.restore(app_state=app_state)
    return True


def patch_engine_to_use_torchsnapshot(engine: DeepSpeedEngine) -> None:
    """
    Patch a DeepSpeedEngine to use torchsnapshot to save its optimizer states.

    Args:
        engine: The DeepSpeedEngine to patch.

    WARNING: This function is not a proper integration with deepspeed. Its
    purpose is to demonstrate/benchmark a potential integration. Only use it at
    your own risk.
    """
    if not isinstance(engine.optimizer, DeepSpeedZeroOptimizer_Stage3):
        raise RuntimeError(
            "patch_engine_to_use_torchsnapshot only supports DeepSpeedZeroOptimizer_Stage3."
        )
    engine._save_zero_checkpoint = MethodType(_save_zero_checkpoint, engine)
    engine._load_zero_checkpoint = MethodType(_load_zero_checkpoint, engine)
