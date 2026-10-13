# VLABench

## Setup

Use [VLABench `cf588fe`](http://github.com/OpenMOSS/VLABench/tree/cf588fe60c0c7282174fe979f5913170cfe69017)
and follow its [installation instructions](http://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/README.md#installation)
in a separate environment, including the simulation assets. Then install the UNITAS client
and apply its SDK compatibility patch:

```bash
cd /path/to/UNITAS
python -m pip install -r evaluation/requirements.txt
python -m pip install --no-deps -e .
python -m evaluation.vlabench.sdk --repo /path/to/VLABench
```

Evaluation uses our five [track configurations](../evaluation/vlabench/configs/tracks),
with 2,460 episodes.

## Evaluation

Start [server.py](../src/unitas/inference/server.py) in the
[UNITAS environment](../README.md#installation), from the repository root:

```bash
conda activate unitas
python src/unitas/inference/server.py --domain vlabench \
  --checkpoint checkpoints/unitas_vlabench_finetuned.pt \
  --florence-dir checkpoints/Florence-2-large --port 8000
```

In the VLABench environment, run [client.py](../evaluation/vlabench/client.py)
and [aggregate.py](../evaluation/vlabench/aggregate.py) from the UNITAS root:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python evaluation/vlabench/client.py \
  --vlabench-root /path/to/VLABench/VLABench \
  --host 127.0.0.1 --port 8000 --output-dir outputs/vlabench

python evaluation/vlabench/aggregate.py --output-dir outputs/vlabench
```

`--vlabench-root` points to the inner directory containing `assets/` and `configs/`.
For a quick check, add `--track track_1_in_distribution_new`, `--tasks select_book`
and `--episodes-per-task 1`. Completed episodes are skipped on restart.

For the scratch model, use `checkpoints/unitas_vlabench_scratch.pt` and a new output directory.
