Getting Started
***************


:class:`~torchsnapshot.Snapshot` is the core API of TorchSnapshot. The class represents application state persisted in storage. A user can take a snapshot of an application via :meth:`Snapshot.take() <torchsnapshot.Snapshot.take>` (a.k.a saving a checkpoint), and restore the state of an application from a snapshot via :meth:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` (a.k.a loading a checkpoint).


Installation
------------

Please refer to `README.md <https://github.com/pytorch/torchsnapshot#install>`_ for installation instructions.


.. _app-state:


Describing the Application State
--------------------------------

Before using :class:`~torchsnapshot.Snapshot` to save or restore application state, the user needs to **describe the application state**. This is done by creating a dictionary that contains all **stateful objects** that the user wishes to capture as application state. Any object that exposes ``.state_dict()`` and ``.load_state_dict()`` are considered stateful objects. Common PyTorch objects such as :class:`~torch.nn.Module`, :class:`~torch.optim.Optimizer`, and LR schedulers all qualify as stateful objects and can be captured directly. Objects that don't meet this requirement can be captured via :class:`~torchsnapshot.StateDict`.

.. code-block:: Python

    from torchsnapshot import StateDict

    app_state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": StateDict(iterations=0)
    }


Taking a Snapshot
-----------------

Once the :ref:`application state <app-state>` is described, users can take a snapshot of the application via :func:`Snapshot.take() <torchsnapshot.Snapshot.take>`, which persists the application state at the user specified path and returns a reference to the snapshot.

TorchSnapshot provides performant and reliable integration with commonly used cloud object storages out of the box. Users can select a different storage backend by prepending a URI prefix to the path (e.g. ``s3://`` for `S3 <https://aws.amazon.com/s3/>`_, ``gs://`` for `Google Cloud Storage <https://cloud.google.com/storage>`_). By default, the prefix is ``fs://`` which suggests that the path is a file system location.

.. code-block:: Python

    from torchsnapshot import Snapshot

    # Persist the application state to local FS or network FS
    snapshot = Snapshot.take(path="/path/to/my/snapshot", app_state=app_state)

    # Alternatively, persist the application state to S3
    snapshot = Snapshot.take(
        path="s3://bucket/path/to/my/snapshot",
        app_state=app_state
    )


.. note::
   Do not move GPU tensors to CPU before saving them with TorchSnapshot. TorchSnapshot implements various optimizations for increasing the throughput and decreasing the host memory usage of GPU-to-storage transfers. Moving GPU tensors to CPU manually will lower the throughput and increase the chance of "out of memory" issues.


Restoring From a Snapshot
-------------------------

To restore from a snapshot, the user first needs to obtain a reference to the snapshot. As seen previously, in the process where the snapshot is taken, a reference to the snapshot is returned by :func:`Snapshot.take() <torchsnapshot.Snapshot.take>`. In another process, which is more common for resumption, a reference can be obtained by creating a :class:`~torchsnapshot.Snapshot` object with the snapshot path.

To restore the :ref:`application state <app-state>` from the snapshot, invoke :func:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` with the application state:

.. code-block:: Python

    from torchsnapshot import Snapshot

    snapshot = Snapshot(path="/path/to/my/snapshot")
    snapshot.restore(app_state=app_state)

.. note::

    :func:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` restores stateful objects in-place whenever possible to avoid creating unneccessary intermediate copies of the state.


Distributed Snapshot
--------------------

TorchSnapshot supports distributed applications as first class citizens. To take a snapshot of a distributed application, simply invoke :func:`Snapshot.take() <torchsnapshot.Snapshot.take>` on all ranks simultaneously (similar to calling a torch.distributed API). The persisted application state will be organized as a single snapshot.

TorchSnapshot drastically improves the checkpointing performance of distributed data parallel applications by distributing the write workload evenly across all ranks (`benchmarks <https://github.com/pytorch/torchsnapshot/tree/main/benchmarks/ddp>`_). The speedup is a result of better GPU copy unit utilization and storage I/O parallelization.


.. code-block:: Python

    ddp_model = DistributedDataParallel(model)
    app_state = {"model": ddp_model}
    snapshot = Snapshot.take(path="/path/to/my/snapshot", app_state=app_state)


Snapshot Content Access
-----------------------

Objects within a snapshot can be efficiently accessed without fetching the entire snapshot, even when the snapshot is stored in cloud object storage. This is useful for transfer learning and post-processing models that are too large to fit in a single host/device.

.. code-block:: Python

    snapshot = Snapshot(path="/path/to/my/snapshot")

    # Available object paths can be queried with snapshot.get_manifest()
    layer_0_weight = snapshot.read_object(path="0/model/layer_0.weight")


Taking a Snapshot Asynchronously
--------------------------------

When host memory is abundant, users can leverage it with :func:`Snapshot.async_take() <torchsnapshot.Snapshot.async_take>` to allow training to resume before all storage I/O completes. :func:`Snapshot.async_take() <torchsnapshot.Snapshot.async_take>` returns as soon as it stages the snapshot content in host RAM and schedules storage I/O in background. This can drastically reduce the time blocked for checkpointing especially when the underly storage is slow.


.. code-block:: Python

    pending_snapshot = Snapshot.async_take(
        path="/path/to/my/snapshot",
        app_state=app_state,
    )

    # Users can query the pending snapshot's status
    if pending_snapshot.done():
        ...

    # ... or wait for the pending snapshot to complete
    snapshot = pending_snapshot.wait()

.. note::

    Despite having "async" in the API name, the snapshot created via :func:`Snapshot.async_take() <torchsnapshot.Snapshot.async_take>` is consistent and deterministic.


Reproducibility
---------------

TorchSnapshot provides a utility called :class:`RNGState <torchsnapshot.rng_state.RNGState>` to help users manage reproducibility. If an :class:`RNGState <torchsnapshot.rng_state.RNGState>` object is captured in the application state, TorchSnapshot ensures that the global RNG state is set to the same values after restoring from the snapshot as it was after taking the snapshot.

.. code-block:: Python

    from torchsnapshot import Snapshot, RNGState

    app_state = {"model": model, "optimizer": optimizer, "rng_state": RNGState()}
    snapshot = Snapshot.take(path="/path/to/my/snapshot", app_state=app_state)
    # global RNG state => {x}

    # In the same process or in another process
    snapshot.restore(app_state=app_state)
    # global RNG state => {x}


Elasticity (Experimental)
-------------------------

Distributed applications can restore from a snapshot taken with a different world size as long as the snapshot only contains **replicated** objects or **sharded** objects:

- A replicated object is an object that (1) exists on all ranks under the same state dict key and (2) holds the same value on all ranks during :func:`Snapshot.take() <torchsnapshot.Snapshot.take>`. An example of a replicated object is a tensor in :class:`DistributedDataParallel`'s state dict. When a replicated object is restored, it is made available to all newly joined ranks.
- A sharded object is an object whose state is sharded across multiple ranks. Currently the only supported sharded object is :class:`ShardedTensor`. :class:`ShardedTensor`\s under the same state dict keys on different ranks are treated as part of the same global tensor. When the sharding of a global tensor changed due to world size change on restore, the global tensors will be automatically resharded correctly.

.. note::

    If an object is neither replicated or sharded, it can only be loaded by the saving rank on restore. This prevents accidentally treating non-elastic models as an elastic one.
