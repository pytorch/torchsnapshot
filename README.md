# torchsnapshot

A light-weight library for adding fault tolerance to large-scale PyTorch distributed training workloads.

## Install

Requires Python >= 3.7.

From pip:

```bash
pip install torchsnapshot
```

## Concepts
- **Stateful object** - an object that whose state can be obtained via `.state_dict()` and restored via `.load_state_dict()`. Most PyTorch components (e.g. `Module`, `Optimizer`, `LRScheduler`) already implement this [protocol](https://github.com/facebookresearch/torchsnapshot/blob/main/torchsnapshot/stateful.py).
- **App state** - the application state described using multiple stateful objects.
- **Snapshot** - the persisted app state.


## Basic Usage

Describing the application state with multiple stateful objects:
```python
app_state = {"model": model, "optimizer": optimizer}
```


Taking a snapshot of the application state:
```python
from torchsnapshot import Snapshot

# File System
snapshot = Snapshot.take(path="/foo/bar/baz", app_state=app_state)

# S3
snapshot = Snapshot.take(path="s3://foo/bar", app_state=app_state)

# Google Cloud Storage
snapshot = Snapshot.take(path="gcs://foo/bar", app_state=app_state)
```

Referencing an existing snapshot:
```python
snapshot = Snapshot(path="foo/bar/baz")
```


Restoring the application state from a snapshot:
```python
snapshot.restore(app_state=app_state)
```

See the [example directory](https://github.com/facebookresearch/torchsnapshot/tree/main/examples) for more examples.


## License

torchsnapshot is BSD licensed, as found in the [LICENSE](LICENSE) file.
