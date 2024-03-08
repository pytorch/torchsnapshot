#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict

import torch


class RNGState:
    """
    A special stateful object for saving and restoring global RNG state.

    When captured in the application state, it is guaranteed that the global
    RNG state is set to the same values after restoring from the snapshot as it
    was after taking the snapshot.

    Example:

    ::

        >>> Snapshot.take(
        >>>     path="foo/bar",
        >>>     app_state={"rng_state": RNGState()},
        >>> )
        >>> after_take = torch.rand(1)

        >>> # In the same process or in another process
        >>> snapshot = Snapshot(path="foo/bar")
        >>> snapshot.restore(app_state)
        >>> after_restore = torch.rand(1)

        >>> torch.testing.assert_close(after_take, after_restore)
    """

    # TODO: augment this to capture rng states other than torch.get_rng_state()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"rng_state": torch.get_rng_state()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        torch.set_rng_state(state_dict["rng_state"])
