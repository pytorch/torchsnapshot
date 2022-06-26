#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import fnmatch
import functools
import io
import itertools
import logging
import os
from collections import defaultdict
from typing import Any, cast, Dict, List, Optional, Tuple, TypeVar

import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .flatten import flatten, inflate
from .io_preparer import ObjectBufferConsumer, prepare_read, prepare_write
from .io_types import IOReq, ReadReq, StoragePlugin, WriteReq
from .manifest import (
    Entry,
    get_available_entries,
    is_replicated,
    Manifest,
    SnapshotMetadata,
)
from .pg_wrapper import PGWrapper
from .rng_state import RNGState
from .scheduler import (
    get_process_memory_budget_bytes,
    sync_execute_read_reqs,
    sync_execute_write_reqs,
)
from .stateful import AppState, Stateful
from .storage_plugin import url_to_storage_plugin_in_event_loop
from .version import __version__ as torchsnapshot_version

logger: logging.Logger = logging.getLogger(__name__)

SNAPSHOT_METADATA_FNAME = ".snapshot_metadata"

T = TypeVar("T")


class Snapshot:
    """
    Snapshot represents the persisted program state at one point in time.

    Basic usage:
    ::
        # Define the program state
        app_state = {"model": model, "optimizer": optimizer"}

        # At an appropriate time, persist the program state as a snapshot
        snapshot = Snapshot.take(path=path, app_state=app_state)

        # On resuming, restore the program state from a snapshot
        snapshot.restore(app_state)

    Overview:

        At high level, torchsnapshot saves each value in state dicts as a
        file/object in the corresponding storage system. It also saves a manifest
        describing the persisted values and the structure of the original state
        dict.

        Comparing with func:`torch.save` and func:`torch.load`, torchsnapshot:

        - Enables efficient random access of persisted model weights.
        - Accelerates persistence by parallelizing writes.
            - For replicated values, persistence is parallelized across ranks.
        - Enables flexible yet robust elasticity (changing world size on
          restore).

    Elasticity:

        Elasticity is implemented via correctly making persisted values
        available to a newly joined rank, and having it correctly restores the
        corresponding runtime objects from the persisted values.

        For the purpose of elasticity, all persisted values fall into one of
        the categories in [per-rank, replicated, sharded].

        per-rank:
            By default, all non-sharded values are treated as per-rank.
            On save, the value is only saved by the owning rank.
            On load, the value is only made available to the same rank.

        replicated:
            A user can suggest any non-sharded value as replicated via glob
                patterns.
            On save, the value is only saved once (can be by any rank).
            On load, the value is made available to all ranks, including newly
                joined ranks.

        sharded:
            Specific types are always treated as sharded (e.g. ShardedTensor).
            On save, all shard-owning ranks save their shards.
            On load, all shards are made available to all ranks, including
                newly joined rank. All ranks can read from all shards for
                restoring the runtime object from persisted values.
                (ShardedTensor resharding is powered by torch.dist.checkpoint).

        If all values within a snapshot are either replicated or sharded, the
        snapshot is automatically reshard-able.

        If a snapshot contains per-rank values, it cannot be resharded unless
        the per-rank values are explicitly coerced to replicated on load.
    """

    def __init__(
        self,
        path: str,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Initializes the reference to an existing snapshot.

        Args:
            path: The location of the snapshot.
            pg: The process group for the processes restoring from the snapshot.
                When unspecified:
                    - If distributed is initialized, the global process group will be used.
                    - If distributed is not initialized, single process is assumed.
        """
        self.path: str = path
        self.pg: Optional[dist.ProcessGroup] = pg

    @classmethod
    def take(
        cls,
        path: str,
        app_state: AppState,
        pg: Optional[dist.ProcessGroup] = None,
        replicated: Optional[List[str]] = None,
    ) -> "Snapshot":
        """
        Take a snapshot from the program state.

        Args:
            app_state: The program state to take the snapshot from.
            path: The location to save the snapshot.
            pg: The process group for the processes taking the snapshot.
                When unspecified:
                    - If distributed is initialized, the global process group will be used.
                    - If distributed is not initialized, single process is assumed.
            replicated: A list of glob patterns for hinting the matching paths
                as replicated. Note that patterns not specified by all ranks
                are ignored.

        Returns:
            The newly taken snapshot.
        """
        event_loop = asyncio.new_event_loop()
        pg_wrapper = PGWrapper(pg)
        path = cls._collate_path(path, pg)
        storage = url_to_storage_plugin_in_event_loop(
            url_path=path, event_loop=event_loop
        )
        replicated = replicated or []
        # TODO: verify replicated across ranks
        # TODO: infer replicated pattern for known stateful types (e.g.
        # DistributedDataParallel)

        app_state = app_state.copy()
        rng_state_item = cls._pop_rng_state(app_state=app_state)
        rng_state_dict = None

        rank = pg_wrapper.get_rank()
        manifest: Manifest = {}

        # Invariant: for the same snapshot, the RNG state is the same after
        # .take() and .restore().
        # This can be achieved by ensuring .take() has no side effect on the
        # RNG state. Since we can't guarantee that the .state_dict() method on
        # stateful objects has no side effect on the RNG state, we retrieve the
        # RNG state before saving other stateful objects, and restore the RNG
        # state after saving other stateful objects.
        if rng_state_item is not None:
            key, stateful = rng_state_item
            rng_state_dict = stateful.state_dict()
            mnfst = cls._save_stateful(
                stateful_key=key,
                stateful=stateful,
                storage=storage,
                replicated=replicated,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
            manifest.update(mnfst)

        # Different ranks can register different sets of stateful objects,
        # whose .state_dict() methods may invoke collectives. To avoid
        # potential interleaving of different collectives, we first gather the
        # global key list, then invoke .state_dict() on stateful objects in
        # order with synchronization.
        global_keys = cls._gather_keys(keys=list(app_state.keys()), pg=pg)

        for key in global_keys:
            mnfst = cls._save_stateful(
                stateful_key=key,
                stateful=app_state.get(key),
                storage=storage,
                replicated=replicated,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
            manifest.update(mnfst)
            pg_wrapper.barrier()

        manifest = cls._gather_manifest(manifest=manifest, pg=pg_wrapper)
        if rank == 0:
            cls._write_snapshot_metadata(
                world_size=pg_wrapper.get_world_size(),
                manifest=manifest,
                storage=storage,
                event_loop=event_loop,
            )
        pg_wrapper.barrier()

        # Undo any potential side effects to the RNG state.
        if rng_state_item is not None:
            _, stateful = rng_state_item
            rng_state_dict = stateful.load_state_dict(
                cast(Dict[str, Any], rng_state_dict)  # pyre-ignore[33]
            )

        storage.sync_close(event_loop=event_loop)
        return cls(path=path, pg=pg)

    def restore(self, app_state: AppState) -> None:
        """
        Restores the program state from the snapshot.

        Args:
            app_state: The program state to restore from the snapshot.
        """
        event_loop = asyncio.new_event_loop()
        pg_wrapper = PGWrapper(self.pg)
        rank = pg_wrapper.get_rank()
        storage = url_to_storage_plugin_in_event_loop(
            url_path=self.path, event_loop=event_loop
        )

        app_state = app_state.copy()
        rng_state_item = self._pop_rng_state(app_state=app_state)

        # TODO: cache this for newly created snapshot
        snapshot_metadata = self._read_snapshot_metadata(
            storage=storage, event_loop=event_loop
        )
        manifest = snapshot_metadata.manifest
        available_entries = get_available_entries(manifest, rank)

        global_keys = self._gather_keys(keys=list(app_state.keys()), pg=self.pg)

        for key in global_keys:
            self._load_stateful(
                rank=rank,
                stateful_key=key,
                stateful=app_state.get(key),
                available_entries=available_entries,
                storage=storage,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
            pg_wrapper.barrier()

        # Restore the RNG state last to avoid potential side effects.
        if rng_state_item is not None:
            key, stateful = rng_state_item
            self._load_stateful(
                rank=rank,
                stateful_key=key,
                stateful=stateful,
                available_entries=available_entries,
                storage=storage,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
        storage.sync_close(event_loop=event_loop)

    @classmethod
    def _save_stateful(
        cls,
        stateful_key: str,
        stateful: Optional[Stateful],
        storage: StoragePlugin,
        replicated: List[str],
        pg: PGWrapper,
        event_loop: asyncio.AbstractEventLoop,
    ) -> Manifest:
        if stateful is not None:
            state_dict = stateful.state_dict()
            manifest, flattened = flatten(state_dict, prefix=stateful_key)
        else:
            manifest, flattened = {}, {}

        # This is a collective call
        replicated_paths = cls._scatter_replicated_entries(flattened, replicated, pg)

        entries: List[Entry] = []
        write_reqs: List[WriteReq] = []
        for logical_path, obj in flattened.items():
            entry, item_write_reqs = prepare_write(
                obj=obj,
                logical_path=logical_path,
                rank=pg.get_rank(),
                replicated=logical_path in replicated_paths,
            )
            entries.append(entry)
            write_reqs += item_write_reqs

        memory_budget_bytes = get_process_memory_budget_bytes(pg=pg)
        sync_execute_write_reqs(
            write_reqs=write_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=pg.get_rank(),
            event_loop=event_loop,
        )
        manifest.update(dict(zip(flattened.keys(), entries)))
        return manifest

    @staticmethod
    def _scatter_replicated_entries(
        flattened: Dict[str, Any],
        replicated: List[str],
        pg: PGWrapper,
    ) -> List[str]:
        rank = pg.get_rank()
        world_size = pg.get_world_size()
        replicated_paths = []
        for path, val in flattened.items():
            if any(fnmatch.fnmatch(path, p) for p in replicated) and not isinstance(
                val, ShardedTensor
            ):
                replicated_paths.append(path)

        # pyre-ignore
        obj_list: List[List[str]] = [None] * world_size
        pg.all_gather_object(obj_list, replicated_paths)
        if rank == 0:
            # A path is only treated as replicated if:
            # (1) The path matches one of the patterns specified in `replicated`
            # (2) The path exists on all ranks
            # (3) The value is not sharded
            path_count = defaultdict(int)
            for paths in obj_list:
                for path in paths:
                    path_count[path] += 1
            replicated_paths = list(
                filter(lambda p: path_count[p] == world_size, replicated_paths)
            )

            # Split the work among ranks
            obj_list = [replicated_paths]
            pg.broadcast_object_list(obj_list, src=0)
        else:
            # pyre-ignore
            obj_list = [None]
            pg.broadcast_object_list(obj_list, src=0)
            replicated_paths = obj_list[0]

        # A naive way of spliting write load across ranks
        # TODO: balance write load across ranks
        for idx, path in enumerate(replicated_paths):
            if idx % world_size != rank:
                del flattened[path]

        return replicated_paths

    @classmethod
    def _load_stateful(
        cls,
        rank: int,
        stateful_key: str,
        stateful: Optional[Stateful],
        available_entries: Manifest,
        storage: StoragePlugin,
        pg: PGWrapper,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        if stateful is None:
            return

        # There are two ways to restore a stateful:
        # 1. Reconstruct the state dict from storage and use it to call .load_state_dict()
        # 2. Obtain the state dict via .state_dict(), restore its values from storage,
        # then use it to call .load_state_dict()
        #
        # When .state_dict() returns references to the original tensors, #2 is
        # more memory-efficient, because a tensor loaded from storage can be
        # freed as soon as its value is copied to the original tensor.
        state_dict = stateful.state_dict()
        mnfst, flattened = flatten(state_dict, prefix=stateful_key)
        del state_dict

        read_reqs: List[ReadReq] = []
        for logical_path, obj in flattened.items():
            if logical_path not in available_entries:
                raise RuntimeError(
                    f"""
When restoring from the snapshot, stateful object "{stateful_key}" requested
path "{logical_path}" which was not available to rank {rank}.

- If the entry does not exist in the snapshot, it means that the state dict
  entry was introduced after the snapshot was taken. To partially restore from
  the snapshot, please explicitly ignore the state dict entries missing from
  the snapshot.

- If the entry exists in the snapshot, it could mean that the world size has
  changed and the entry was not marked as replicated when the snapshot was
  taken. To resolve the issue, try any of:
    - Re-taking the snapshot with the new world size
    - Re-taking the snapshot with the original world size, ensuring all
          non-sharded values are marked as replicated
    - Coerce the missing entry into replicated on restore"""
                )

            rrs = prepare_read(
                entry=available_entries[logical_path],
                obj_out=obj,
            )
            for rr in rrs:
                buffer_consumer = rr.buffer_consumer
                if isinstance(buffer_consumer, ObjectBufferConsumer):
                    # ObjectBufferConsumer deals with objects that can not be
                    # in-place restored. We need to replace the original object
                    # in the flattened dictionary with the object materialized
                    # by the buffer consumer.
                    buffer_consumer.set_consume_callback(
                        functools.partial(dict.__setitem__, flattened, logical_path)
                    )
            read_reqs += rrs

        memory_budget_bytes = get_process_memory_budget_bytes(pg=pg)
        sync_execute_read_reqs(
            read_reqs=read_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=pg.get_rank(),
            event_loop=event_loop,
        )

        state_dict = inflate(mnfst, flattened, prefix=stateful_key)
        stateful.load_state_dict(state_dict)

    @staticmethod
    def _write_snapshot_metadata(
        world_size: int,
        manifest: Manifest,
        storage: StoragePlugin,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        # TODO: use semantic versioning for backward compatibility
        snapshot_metadata = SnapshotMetadata(
            version=torchsnapshot_version, world_size=world_size, manifest=manifest
        )
        io_req = IOReq(
            path=SNAPSHOT_METADATA_FNAME,
            buf=io.BytesIO(snapshot_metadata.to_yaml().encode("utf-8")),
        )
        storage.sync_write(io_req=io_req, event_loop=event_loop)

    @staticmethod
    def _read_snapshot_metadata(
        storage: StoragePlugin, event_loop: asyncio.AbstractEventLoop
    ) -> SnapshotMetadata:
        io_req = IOReq(path=SNAPSHOT_METADATA_FNAME)
        storage.sync_read(io_req=io_req, event_loop=event_loop)
        yaml_str = io_req.buf.getvalue().decode("utf-8")
        return SnapshotMetadata.from_yaml(yaml_str)

    @staticmethod
    def _collate_path(path: str, pg: Optional[dist.ProcessGroup]) -> str:
        obj_list = [path]
        PGWrapper(pg).broadcast_object_list(obj_list, src=0)
        if obj_list[0] != path:
            logger.warning(
                f"Rank {PGWrapper(pg).get_rank()} specified a path ({path}) "
                f"different from rank 0 ({obj_list[0]}). Using path specified by rank 0."
            )
        return obj_list[0]

    @staticmethod
    def _gather_keys(keys: List[str], pg: Optional[dist.ProcessGroup]) -> List[str]:
        gathered_keys = [None] * PGWrapper(pg).get_world_size()
        # pyre-ignore
        gathered_keys[PGWrapper(pg).get_rank()] = keys
        PGWrapper(pg).all_gather_object(gathered_keys, keys)
        # pyre-ignore
        return sorted(set(itertools.chain.from_iterable(gathered_keys)))

    @staticmethod
    def _pop_rng_state(
        app_state: AppState,
    ) -> Optional[Tuple[str, RNGState]]:
        rng_state_items = {
            key: stateful
            for key, stateful in app_state.items()
            if isinstance(stateful, RNGState)
        }
        if len(rng_state_items) > 1:
            raise RuntimeError(
                "Multiple RNGState objects in app state: "
                f"{list(rng_state_items.keys())}"
            )
        elif len(rng_state_items) == 1:
            key, stateful = list(rng_state_items.items())[0]
            del app_state[key]
            return key, stateful
        else:
            return None

    @staticmethod
    def _gather_manifest(manifest: Dict[str, Any], pg: PGWrapper) -> Dict[str, Any]:
        manifests = [None] * pg.get_world_size()
        pg.all_gather_object(manifests, manifest)
        manifests = cast(List[Manifest], manifests)

        global_manifest = {}
        replicated_entries = {}
        for manifest in manifests:
            for path, entry in manifest.items():
                if is_replicated(entry):
                    replicated_entries[path] = entry

        for rank, manifest in enumerate(manifests):
            for path, entry in replicated_entries.items():
                if path not in manifest:
                    manifest[path] = entry
            for logical_path, entry in manifest.items():
                global_manifest[os.path.join(str(rank), logical_path)] = entry

        return global_manifest
