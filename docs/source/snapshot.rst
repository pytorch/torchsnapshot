Snapshot
========

:class:`~torchsnapshot.Snapshot` is the core API of TorchSnapshot. The class represents application state persisted in storage. A user can take a snapshot of an application via :meth:`Snapshot.take() <torchsnapshot.Snapshot.take>` (a.k.a saving a checkpoint), or restore the state of an application from a snapshot via :meth:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` (a.k.a loading a checkpoint).


.. _app-state:

Describing the application state
--------------------------------

Before using :class:`~torchsnapshot.Snapshot` to save or restore application state, the user needs to **describe the application state**. This is done by creating a dictionary that contains all **stateful objects** that the user wishes to capture as application state:

.. code-block:: Python

    app_state = {"model": model, "optimizer": optimizer}

Any object that exposes ``.state_dict()`` and ``.load_state_dict()`` are considered **stateful objects**. Common PyTorch objects such as :class:`~torch.nn.Module`, :class:`~torch.optim.Optimizer`, and LR Schedulers all qualify as **stateful objects** and can be captured directly. Objects that don't meet this requirement can be captured via :class:`~torchsnapshot.StateDict`:

.. code-block:: Python

    from torchsnapshot import StateDict

    extra_state = StateDict(iterations=0)
    app_state = {"model": model, "optimizer": optimizer, "extra_state": extra_state}


Taking a snapshot
-----------------

Once the :ref:`application state <app-state>` is described, the user can take a snapshot of the application via :func:`Snapshot.take() <torchsnapshot.Snapshot.take>`. :func:`Snapshot.take() <torchsnapshot.Snapshot.take>` persists the application state to the user specified path and returns a :class:`~torchsnapshot.Snapshot` object, which is a reference to the snapshot.

.. code-block:: Python

    from torchsnapshot import Snapshot

    snapshot = Snapshot.take(path="/path/to/my/snapshot", app_state=app_state)

The user specified path can optionally be prepended with a URI prefix. By default, the prefix is ``fs://``, which suggests that the path is a file system location. TorchSnapshot also provides performant and reliable integration with commonly used cloud object storages. A storage backend can be selected by prepending the corresponding URI prefix (e.g. ``s3://`` for S3, ``gs://`` for Google Cloud Storage).

.. code-block:: Python

    snapshot = Snapshot.take(
        path="s3://bucket/path/to/my/snapshot",
        app_state=app_state
    )


.. note::
   Do not move GPU tensors to CPU before saving them with TorchSnapshot. TorchSnapshot implements various optimizations for increasing the throughput and decreasing the host memory usage of GPU-to-storage transfers. Moving GPU tensors to CPU manually will lower the throughput and increase the chance of "out of memory" issues.


Restoring from a snapshot
-------------------------

To restore from a snapshot, the user first need to obtain a reference to the snapshot. As seen previously, in the process where the snapshot is taken, a reference to the snapshot is returned by :func:`Snapshot.take() <torchsnapshot.Snapshot.take>`. In another process (which is more common for resumption), a reference can be obtained by creating a :class:`~torchsnapshot.Snapshot` object with the snapshot path:

.. code-block:: Python

    from torchsnapshot import Snapshot

    snapshot = Snapshot(path="/path/to/my/snapshot")

To restore the :ref:`application state <app-state>` from the snapshot, invoke :func:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` with the application state:

.. code-block:: Python

    snapshot.restore(app_state=app_state)

.. note::

    :func:`Snapshot.restore() <torchsnapshot.Snapshot.restore>` restores stateful objects in-place to avoid creating unneccessary intermediate copies of the state.


Distributed snapshot
--------------------

TODO


Elasticity
----------

TODO


Reproducibility
---------------

TODO


Taking a snapshot asynchronously
--------------------------------

TODO


API Reference
-------------


.. autoclass:: torchsnapshot.Snapshot
   :members:
   :undoc-members:
