# LIBERO

## Setup

Install [LIBERO](http://github.com/Lifelong-Robot-Learning/LIBERO#readme) and its
assets following the official instructions in a separate Python 3.10 environment.
Then install the UNITAS client with **MuJoCo 3.2.3**:

```bash
cd /path/to/UNITAS
python -m pip install -r evaluation/requirements.txt mujoco==3.2.3
python -m pip install --no-deps -e .
```

Configure the SDK paths with `python -c "import libero.libero"`.
`LIBERO_CONFIG_PATH`, when set, points to the directory containing `config.yaml`.

## Evaluation

Start [server.py](../src/unitas/inference/server.py)
in the [UNITAS environment](../README.md#installation):

```bash
conda activate unitas
cd /path/to/UNITAS
python src/unitas/inference/server.py --domain libero \
  --checkpoint checkpoints/unitas_libero_finetuned.pt \
  --florence-dir checkpoints/Florence-2-large --port 8000
```

In the LIBERO environment, run [client.py](../evaluation/libero/client.py)
from the UNITAS root:

```bash
cd /path/to/UNITAS
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for suite in libero_spatial libero_object libero_goal libero_10; do
  max_steps=800
  if [ "$suite" = libero_10 ]; then max_steps=900; fi
  python evaluation/libero/client.py --host 127.0.0.1 --port 8000 \
    --suite "$suite" --tasks all --episodes-per-task 50 \
    --seed 42 --settle-steps 10 --max-steps "$max_steps" \
    --no-save-videos --output-dir outputs/libero
done
```

For a quick check, run one suite with `--tasks 0 --episodes-per-task 1`.
Add `--resume` to continue a run with the same checkpoint and evaluation settings.

For the scratch model, use `checkpoints/unitas_libero_scratch.pt` and a new output directory.

## LIBERO-Plus

Install [LIBERO-Plus `4976dc3`](http://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182)
and its assets following the official setup in a separate environment.
Then run the [client installation commands above](#setup), including **MuJoCo 3.2.3**.
Set `LIBERO_CONFIG_PATH` to the Plus configuration directory.

Keep the fine-tuned LIBERO server running. In the Plus environment, run
[client.py](../evaluation/libero_plus/client.py)
and [result aggregator](../evaluation/libero_plus/aggregate.py):

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python evaluation/libero_plus/client.py \
  --libero-plus-root /path/to/LIBERO-plus \
  --host 127.0.0.1 --port 8000 --seed 7 --settle-steps 10 \
  --output-dir outputs/libero_plus

python evaluation/libero_plus/aggregate.py \
  --libero-plus-root /path/to/LIBERO-plus --output-dir outputs/libero_plus
```

Add `--suites libero_spatial --task-ids 0` for a quick check.
