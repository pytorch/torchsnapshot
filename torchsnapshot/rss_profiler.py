#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import time
from contextlib import contextmanager
from datetime import timedelta
from threading import Event, Thread
from typing import Generator, List

import psutil

_DEFAULT_MEASURE_INTERVAL = timedelta(milliseconds=100)


def _measure(
    rss_deltas: List[int],
    interval: timedelta,
    baseline_rss_bytes: int,
    stop_event: Event,
) -> None:
    p = psutil.Process()
    while not stop_event.is_set():
        rss_deltas.append(p.memory_info().rss - baseline_rss_bytes)
        time.sleep(interval.total_seconds())


@contextmanager
def measure_rss_deltas(
    rss_deltas: List[int], interval: timedelta = _DEFAULT_MEASURE_INTERVAL
) -> Generator[None, None, None]:
    """
    A context manager that periodically measures RSS (resident set size) delta.

    The baseline RSS is measured when the context manager is initialized.

    Args:
        rss_deltas: The list to which the measured RSS deltas (measured in
            bytes) are appended.
        interval: The interval at which RSS deltas are measured.
    """
    baseline_rss_bytes = psutil.Process().memory_info().rss
    stop_event = Event()
    thread = Thread(
        target=_measure, args=(rss_deltas, interval, baseline_rss_bytes, stop_event)
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
