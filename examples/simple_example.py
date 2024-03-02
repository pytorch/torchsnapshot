#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import uuid
from typing import Optional

import torch
import torchsnapshot
from torchsnapshot.snapshot import Snapshot
from torchsnapshot.stateful import AppState

NUM_EPOCHS = 4
EPOCH_SIZE = 16
BATCH_SIZE = 8


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--restore-path", default=None)
    args: argparse.Namespace = parser.parse_args()

    torch.random.manual_seed(42)

    model = Model()
    optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    progress = torchsnapshot.StateDict(current_epoch=0)

    # torchsnapshot: define app state
    app_state: AppState = {
        "rng_state": torchsnapshot.RNGState(),
        "model": model,
        "optim": optim,
        "progress": progress,
    }
    snapshot: Optional[Snapshot] = None

    # torchsnapshot: restore app state
    if args.restore_path:
        snapshot = Snapshot(args.restore_path)
        print(f"Restoring snapshot from path: {snapshot.path}")
        snapshot.restore(app_state)

    while progress["current_epoch"] < NUM_EPOCHS:
        for _ in range(EPOCH_SIZE):
            X = torch.rand((BATCH_SIZE, 128))
            label = torch.rand((BATCH_SIZE, 1))
            pred = model(X)
            loss = loss_fn(pred, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        progress["current_epoch"] += 1

        # torchsnapshot: take snapshot
        snapshot = torchsnapshot.Snapshot.take(
            f"{args.work_dir}/{uuid.uuid4()}", app_state
        )
        print(f"Snapshot path: {snapshot.path}")


if __name__ == "__main__":
    main()  # pragma: no cover
