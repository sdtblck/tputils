## Utilities for TPUs.

- **TPUMaker** - a convenience class for creating TPU pods.

**Example Usage:**
```python
from tputils import TPUMaker

t = TPUMaker(project="youdreamof-1543654322305", zone="europe-west4-a")
t.make_tpu(size=32, name="test")
```

- **TPUKeepAlive** - runs a python script or function that runs on TPUs, creating a tpu before running calling the function, and remaking and rerunning
the function if the tpu gets preempted.

**Example Usage:**
```python
from tputils import TPUKeepAlive

# create a runner that will make a tpu v3-32 named test and restart it every 12 hours, or when preempted.
t = TPUKeepAlive(size=32, name="test", restart_after=43200, project="youdreamof-1543654322305",
                 zone="europe-west4-a", tf_version="2.4.0")
t.run_script("python main.py --tpu test --model gpt3_medium")
```
