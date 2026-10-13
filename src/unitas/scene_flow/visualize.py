from __future__ import annotations

import argparse
import time
from pathlib import Path
from threading import RLock

import numpy as np


def load_prediction(path: Path):
    with np.load(path, allow_pickle=False) as data:
        scene = data["scene"].copy()
        gripper = data["gripper"].copy()
        colors = data["colors"].copy()
        fps = float(data["fps"])
        gripper_fps = float(data["gripper_fps"])
        instruction = str(data["instruction"].item())
    for name, value in [("scene", scene), ("gripper", gripper)]:
        if (
            value.ndim != 3
            or value.shape[-1] != 3
            or min(value.shape[:2]) == 0
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be nonempty, finite [T,N,3] trajectories")
    if colors.shape != scene.shape[1:] or colors.dtype != np.uint8:
        raise ValueError("colors must be uint8 RGB with shape [N,3]")
    if not np.isfinite([fps, gripper_fps]).all() or min(fps, gripper_fps) <= 0:
        raise ValueError("Trajectory rates must be finite and positive")
    return (scene, gripper, colors, fps, gripper_fps, instruction)


def trajectory_segments(trajectory):
    return np.stack([trajectory[:-1], trajectory[1:]], axis=2).reshape(-1, 2, 3)


class TrajectoryViewer:
    def __init__(self, server, prediction, *, trails=100):
        self.server = server
        self.scene, self.gripper, colors, self.fps, self.gripper_fps, instruction = prediction
        self._lock = RLock()
        self._last_tick = time.monotonic()
        self.last_frame = len(self.scene) - 1
        server.scene.set_up_direction("+z")
        server.gui.add_markdown(instruction)
        with server.gui.add_folder("Playback"):
            self.frame = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, self.last_frame),
                step=1,
                initial_value=0,
                disabled=self.last_frame == 0,
            )
            self.timestamp = server.gui.add_markdown("")
            self.play = server.gui.add_checkbox("Play", initial_value=False)
            self.loop = server.gui.add_checkbox("Loop", initial_value=True)
            buttons = server.gui.add_button_group("Step", options=("Previous", "Next", "Restart"))
            self.speed = server.gui.add_slider("Speed", min=0.25, max=2, step=0.25, initial_value=1)
        with server.gui.add_folder("Display"):
            point_size = server.gui.add_slider(
                "Scene point size", min=0.001, max=0.02, step=0.001, initial_value=0.004
            )
            gripper_size = server.gui.add_slider(
                "Gripper point size", min=0.001, max=0.03, step=0.001, initial_value=0.007
            )
            gripper_visible = server.gui.add_checkbox("Gripper", initial_value=True)
            trail_visible = server.gui.add_checkbox("Scene trails", initial_value=False)
            reset_camera = server.gui.add_button("Reset view")

        self.cloud = server.scene.add_point_cloud(
            "/scene",
            points=self.scene[0],
            colors=colors,
            point_size=point_size.value,
            point_shape="rounded",
            precision="float32",
        )
        self.robot = server.scene.add_point_cloud(
            "/gripper",
            points=self.gripper[0],
            colors=np.array([230, 50, 170], np.uint8),
            point_size=gripper_size.value,
            point_shape="rounded",
            precision="float32",
        )
        count = min(max(trails, 0), self.scene.shape[1])
        lines = None
        if count and self.last_frame:
            ids = np.linspace(0, self.scene.shape[1] - 1, count, dtype=int)
            gradient = np.linspace([255, 140, 40], [140, 70, 220], self.last_frame).astype(np.uint8)
            line_colors = np.repeat(gradient, count, axis=0)[:, None, :].repeat(2, axis=1)
            lines = server.scene.add_line_segments(
                "/scene-trails",
                points=trajectory_segments(self.scene[:, ids]),
                colors=line_colors,
                line_width=1,
                visible=False,
            )
        else:
            trail_visible.disabled = True

        @self.frame.on_update
        def update_frame(event):
            with self._lock:
                if event.client is not None:
                    self.play.value = False
                self._draw_frame()

        @self.play.on_update
        def update_play(_):
            with self._lock:
                self._last_tick = time.monotonic()
                if self.play.value and self.frame.value == self.last_frame:
                    self.frame.value = 0

        @buttons.on_click
        def step(event):
            with self._lock:
                self.play.value = False
                offset = -1 if event.target.value == "Previous" else 1
                self.frame.value = (
                    0
                    if event.target.value == "Restart"
                    else min(max(self.frame.value + offset, 0), self.last_frame)
                )

        @point_size.on_update
        def resize_scene(_):
            self.cloud.point_size = point_size.value

        @gripper_size.on_update
        def resize_gripper(_):
            self.robot.point_size = gripper_size.value

        @gripper_visible.on_update
        def show_gripper(_):
            self.robot.visible = gripper_visible.value

        @trail_visible.on_update
        def show_trails(_):
            if lines is not None:
                lines.visible = trail_visible.value

        @reset_camera.on_click
        def reset_view(event):
            if event.client is not None:
                self._set_camera(event.client)

        server.on_client_connect(self._set_camera)
        self._draw_frame()

    def _draw_frame(self):
        frame = min(self.frame.value, self.last_frame)
        gripper_frame = min(round(frame / self.fps * self.gripper_fps), len(self.gripper) - 1)
        with self.server.atomic():
            self.cloud.points = self.scene[frame]
            self.robot.points = self.gripper[gripper_frame]
            self.timestamp.content = (
                f"{frame / self.fps:.2f} / {self.last_frame / self.fps:.2f} s"
                f" · frame {frame} / {self.last_frame}"
            )

    def _set_camera(self, client):
        center = self.scene[0].mean(0)
        extent = max(float(np.ptp(self.scene[0], axis=0).max()), 0.3)
        with client.atomic():
            # Setting position also translates look_at, so set the target last.
            client.camera.position = center + np.array([extent, -extent, extent * 0.7])
            client.camera.up_direction = (0, 0, 1)
            client.camera.look_at = center

    def tick(self):
        with self._lock:
            now = time.monotonic()
            if not self.play.value or now - self._last_tick < 1 / (self.fps * self.speed.value):
                return
            self._last_tick = now
            if self.frame.value < self.last_frame:
                self.frame.value += 1
            elif self.loop.value:
                self.frame.value = 0
            else:
                self.play.value = False


def main():
    parser = argparse.ArgumentParser(
        description="View predicted scene and gripper trajectories in Viser."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--trails", type=int, default=100)
    args = parser.parse_args()
    import viser

    prediction = load_prediction(args.input)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        viewer = TrajectoryViewer(server, prediction, trails=args.trails)
        print(f"Viser: http://{args.host}:{args.port}")
        while True:
            viewer.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
