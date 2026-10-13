# Scene flow

## Bundled samples

Six RGB-D samples are included in `assets/scene_flow/observations/`; no simulator is needed.

| Sample | Input filename |
|---|---|
| RoboTwin · lift pot | `robotwin_7c59d8eb33_0088.npz` |
| RoboTwin · open drawer | `robotwin_clean_7779f30bed_0150.npz` |
| LIBERO · place mug | `libero_3d040e000e_0077.npz` |
| LIBERO · place bowl | `libero_592e39dde0_0087.npz` |
| VLABench · insert flower | `vlabench_1ffbd3923b_0129.npz` |
| VLABench · select book | `vlabench_6247189d0f_0064.npz` |

After [installation](../README.md#installation), run from the repository root.
[inference.py](../src/unitas/scene_flow/inference.py) predicts gripper motion, then
dense scene flow conditioned on it. [visualize.py](../src/unitas/scene_flow/visualize.py)
displays the saved result.

```bash
sample=robotwin_7c59d8eb33_0088
python src/unitas/scene_flow/inference.py \
  --checkpoint checkpoints/unitas_scene_flow.pt \
  --florence-dir checkpoints/Florence-2-large \
  --input "assets/scene_flow/observations/${sample}.npz" \
  --output "outputs/${sample}.npz"

python src/unitas/scene_flow/visualize.py --input "outputs/${sample}.npz" --port 8080
```

Open `http://127.0.0.1:8080`. Change `sample` to any filename stem above.
Drag the timeline, adjust playback and point size, or show motion trails.

## Your own observations

Add `--save-observation observations/input.json` to a benchmark client command.
Convert the capture with [prepare.py](../src/unitas/scene_flow/prepare.py) in the UNITAS environment:

```bash
python src/unitas/scene_flow/prepare.py \
  --checkpoint checkpoints/unitas_scene_flow.pt \
  --observation observations/input.json --output observations/input.npz
```

Use the resulting NPZ as `--input` for inference.
For a custom query cloud, add `--queries queries.npz` to preparation; this file
contains `xyz0: [N,3]` in world coordinates (metres) and `colors: [N,3]` as uint8 RGB.
Use `--query-frame model` if points are already in the model frame.

The output NPZ contains scene and gripper trajectories, colors, EEF actions and frame rates.

For a remote viewer, add `--host 0.0.0.0` and open `http://<server-ip>:8080`, or
forward the viewer port over SSH.
