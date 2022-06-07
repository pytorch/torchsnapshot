#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
class StateDict(dict):
    """
    A dict that implements the Stateful protocol. It is handy for capturing
    stateful objects that do not already implement the Stateful protocol or
    can't implement the protocol (i.e. primitive types).

    ::

        model = Model()
        progress = StateDict(current_epoch=0)
        app_state = {"model": model, "progress": progress}

        # Load from the last snapshot if available
        ...

        while progress["current_epoch"] < NUM_EPOCHS:
            # Train for an epoch
            ...
            progress["current_epoch"] += 1

            # progress is captured by the snapshot
            Snapshot.take("foo/bar", app_state, backend=...)
    """

    def state_dict(self) -> Dict[str, Any]:
        return self

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.update(state_dict)
