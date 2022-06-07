# torchsnapshot

a lightweight library for fault tolerance to PyTorch programs through program state persistence

## Install

Requires Python >= 3.7.

From pip:

```bash
pip install torchsnapshot
```


## Usage

```python
import torchsnapshot

# Define the program state
app_state = {"model": model, "optimizer": optimizer"}

# At an appropriate time, persist the program state as a snapshot
snapshot = Snapshot.take(path=path, app_state=app_state)

# On resuming, restore the program state from a snapshot
snapshot.restore(app_state)
```


## License

torchsnapshot is BSD licensed, as found in the [LICENSE](LICENSE) file.
