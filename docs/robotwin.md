# RoboTwin

## Setup

Use [RoboTwin 2.0 (`stable_2.0`)](http://github.com/RoboTwin-Platform/RoboTwin/tree/13c3c47ff4312dd62484bcd51be034af55c062d1).
Follow its [installation guide](http://robotwin-platform.github.io/doc/usage/robotwin-install.html)
to set up a separate environment and download the assets. Then install the UNITAS client there:

```bash
cd /path/to/UNITAS
python -m pip install -r evaluation/requirements.txt
python -m pip install --no-deps -e .
```

## Evaluation

Start [server.py](../src/unitas/inference/server.py) in the
[UNITAS environment](../README.md#installation), from the repository root:

```bash
conda activate unitas
python src/unitas/inference/server.py --domain robotwin \
  --checkpoint checkpoints/unitas_robotwin_finetuned.pt \
  --florence-dir checkpoints/Florence-2-large --port 8001
```

In the RoboTwin environment, run [client.py](../evaluation/robotwin/client.py)
from the UNITAS root:

```bash
python evaluation/robotwin/client.py --robotwin-root /path/to/RoboTwin \
  --host 127.0.0.1 --port 8001 --task-name all \
  --task-config demo_clean --num-episodes 50 --output-dir outputs/robotwin_clean

python evaluation/robotwin/client.py --robotwin-root /path/to/RoboTwin \
  --host 127.0.0.1 --port 8001 --task-name all \
  --task-config demo_randomized --num-episodes 50 --output-dir outputs/robotwin_randomized
```

For a quick check, use `--task-name lift_pot --num-episodes 1` with a separate output
directory. Add `--no-save-videos` to disable recording.

For the scratch model, use `checkpoints/unitas_robotwin_scratch.pt` and a new
output directory.
