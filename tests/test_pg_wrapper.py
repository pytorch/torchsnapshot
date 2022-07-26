#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch.distributed as dist
import torch.distributed.launcher as pet
from torchsnapshot.pg_wrapper import PGWrapper
from torchsnapshot.test_utils import get_pet_launch_config


class TestPgWrapper(unittest.TestCase):
    @staticmethod
    def _worker() -> None:
        tc = unittest.TestCase()
        dist.init_process_group(backend="gloo")
        pg_wrapper = PGWrapper(pg=None)
        output_list = [None]
        input_list = [["foo"], ["bar"], ["quaz"]]
        pg_wrapper.scatter_object_list(output_list=output_list, input_list=input_list)
        rank = dist.get_rank()
        tc.assertEqual(output_list, [input_list[rank]])

    def test_scatter_obj_list_dist_initialized(self) -> None:
        lc = get_pet_launch_config(nproc=3)
        pet.elastic_launch(lc, entrypoint=self._worker)()

    def test_scatter_obj_list_dist_uninitialized(self) -> None:
        pg_wrapper = PGWrapper(pg=None)
        output_list = [None]
        src_rank = 1
        input_list = [["foo"], ["bar"], ["quaz"]]
        pg_wrapper.scatter_object_list(
            output_list=output_list, input_list=input_list, src=src_rank
        )
        self.assertEqual(output_list, [input_list[src_rank]])
