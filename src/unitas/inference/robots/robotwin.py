# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin-specific simulation geometry helpers."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh

from unitas.models.config import normalize_robotwin_robot_type
from unitas.inference.robots.assets import asset_root
from unitas.inference.robots.sampler import deterministic_sample_surface


def _stable_int_hash(*parts: object) -> int:
    payload = "\x00".join((str(part) for part in parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def _translation_matrix(
    x: float, y: float, z: float
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, float(x)),
        (0.0, 1.0, 0.0, float(y)),
        (0.0, 0.0, 1.0, float(z)),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class RobotWinEmbodimentSpec:
    robot_type: str
    urdf_path: str
    left_gripper_links: tuple[str, ...]
    right_gripper_links: tuple[str, ...]
    gripper_surface_points_path: str | None = None
    left_root_transform: tuple[tuple[float, float, float, float], ...] = _translation_matrix(
        0.0, 0.0, 0.0
    )
    right_root_transform: tuple[tuple[float, float, float, float], ...] = _translation_matrix(
        0.0, 0.0, 0.0
    )


_ASSET_ROOT = asset_root()
_ROBOTWIN_EMBODIMENT_ROOT = _ASSET_ROOT / "robotwin_embodiments"
ROBOTWIN_EMBODIMENT_SPECS: dict[str, RobotWinEmbodimentSpec] = {
    "aloha": RobotWinEmbodimentSpec(
        robot_type="aloha",
        urdf_path=str(_ROBOTWIN_EMBODIMENT_ROOT / "aloha-agilex/urdf/arx5_description_isaac.urdf"),
        left_gripper_links=("fl_link6", "fl_link7", "fl_link8"),
        right_gripper_links=("fr_link6", "fr_link7", "fr_link8"),
        gripper_surface_points_path=str(
            _ROBOTWIN_EMBODIMENT_ROOT / "aloha-agilex/gripper_surface_points_20_per_arm.json"
        ),
    ),
    "arx_x5": RobotWinEmbodimentSpec(
        robot_type="arx_x5",
        urdf_path=str(_ROBOTWIN_EMBODIMENT_ROOT / "ARX-X5/X5A.urdf"),
        left_gripper_links=("link6", "link7", "link8"),
        right_gripper_links=("link6", "link7", "link8"),
        right_root_transform=_translation_matrix(0.0, -0.8, 0.0),
    ),
    "piper": RobotWinEmbodimentSpec(
        robot_type="piper",
        urdf_path=str(_ROBOTWIN_EMBODIMENT_ROOT / "piper/piper.urdf"),
        left_gripper_links=("link6", "link7", "link8"),
        right_gripper_links=("link6", "link7", "link8"),
        right_root_transform=_translation_matrix(0.0, -0.8, 0.0),
    ),
    "franka_panda": RobotWinEmbodimentSpec(
        robot_type="franka_panda",
        urdf_path=str(_ROBOTWIN_EMBODIMENT_ROOT / "franka-panda/panda.urdf"),
        left_gripper_links=("panda_hand", "panda_leftfinger", "panda_rightfinger"),
        right_gripper_links=("panda_hand", "panda_leftfinger", "panda_rightfinger"),
        right_root_transform=_translation_matrix(0.0, -0.8, 0.0),
    ),
    "ur5_wsg": RobotWinEmbodimentSpec(
        robot_type="ur5_wsg",
        urdf_path=str(_ROBOTWIN_EMBODIMENT_ROOT / "ur5-wsg/ur5_wsg_gripper.urdf"),
        left_gripper_links=(
            "wsg_50_base_link",
            "gripper_left",
            "finger_left",
            "gripper_right",
            "finger_right",
        ),
        right_gripper_links=(
            "wsg_50_base_link",
            "gripper_left",
            "finger_left",
            "gripper_right",
            "finger_right",
        ),
        right_root_transform=_translation_matrix(0.8, 0.0, 0.1),
    ),
}


def _robotwin_base_positions(robot_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right robot-base origins in the stored source frame."""
    spec = ROBOTWIN_EMBODIMENT_SPECS[normalize_robotwin_robot_type(robot_type)]
    left_root = np.asarray(spec.left_root_transform, dtype=np.float32)
    right_root = np.asarray(spec.right_root_transform, dtype=np.float32)
    if spec.robot_type == "aloha":
        sampler = RobotWinLiteSampler(
            urdf_path=spec.urdf_path, device="cpu", link_whitelist=[], robot_type=spec.robot_type
        )
        left_base = (left_root @ sampler._link_matrix("fl_base_link", {}))[:3, 3]
        right_base = (right_root @ sampler._link_matrix("fr_base_link", {}))[:3, 3]
        return (left_base.astype(np.float32), right_base.astype(np.float32))
    return (left_root[:3, 3].copy(), right_root[:3, 3].copy())


def bimanual_midpoint_transform_from_bases(
    left_base: np.ndarray, right_base: np.ndarray
) -> np.ndarray:
    """Build model-from-source using the midpoint and horizontal base axis."""
    left_base = np.asarray(left_base, dtype=np.float64).reshape(3)
    right_base = np.asarray(right_base, dtype=np.float64).reshape(3)
    origin_source = 0.5 * (left_base + right_base)
    z_model_in_source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_model_in_source = left_base - right_base
    y_model_in_source[2] = 0.0
    norm = float(np.linalg.norm(y_model_in_source))
    if norm <= 1e-09:
        raise ValueError(
            f"Cannot construct bimanual_midpoint: left/right robot bases have the same horizontal position ({left_base}, {right_base})"
        )
    y_model_in_source /= norm
    x_model_in_source = np.cross(y_model_in_source, z_model_in_source)
    x_model_in_source /= np.linalg.norm(x_model_in_source)
    z_model_in_source = np.cross(x_model_in_source, y_model_in_source)
    source_from_model_rotation = np.stack(
        (x_model_in_source, y_model_in_source, z_model_in_source), axis=1
    )
    model_from_source_rotation = source_from_model_rotation.T
    model_from_source = np.eye(4, dtype=np.float32)
    model_from_source[:3, :3] = model_from_source_rotation.astype(np.float32)
    model_from_source[:3, 3] = (-model_from_source_rotation @ origin_source).astype(np.float32)
    return model_from_source


@lru_cache(maxsize=None)
def _cached_bimanual_midpoint_transform(robot_type: str) -> tuple[tuple[float, ...], ...]:
    left_base, right_base = _robotwin_base_positions(robot_type)
    transform = bimanual_midpoint_transform_from_bases(left_base, right_base)
    return tuple((tuple((float(value) for value in row)) for row in transform))


def robotwin_model_from_source(robot_type: str, coordinate_frame: str) -> np.ndarray:
    """Return the selected RoboTwin model-from-source rigid transform."""
    coordinate_frame = str(coordinate_frame).strip().lower()
    if coordinate_frame == "source":
        return np.eye(4, dtype=np.float32)
    if coordinate_frame != "bimanual_midpoint":
        raise ValueError(
            f"robotwin_coordinate_frame must be 'source' or 'bimanual_midpoint', got {coordinate_frame!r}"
        )
    robot_type = normalize_robotwin_robot_type(robot_type)
    return np.asarray(_cached_bimanual_midpoint_transform(robot_type), dtype=np.float32).copy()


def _robotwin_rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = (np.cos(roll), np.sin(roll))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cy, sy = (np.cos(yaw), np.sin(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _robotwin_origin_to_matrix(origin: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if origin is None:
        return matrix
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float32)
    if xyz.size == 3:
        matrix[:3, 3] = xyz
    if rpy.size == 3:
        matrix[:3, :3] = _robotwin_rpy_to_matrix(rpy)
    return matrix


def _robotwin_axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-09:
        return np.eye(4, dtype=np.float32)
    axis = axis / norm
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    return matrix


def _robotwin_identity_batch(batch_size: int) -> np.ndarray:
    return np.broadcast_to(np.eye(4, dtype=np.float32), (int(batch_size), 4, 4)).copy()


def _robotwin_axis_angle_to_matrix_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    matrices = _robotwin_identity_batch(angles.shape[0])
    if norm <= 1e-09:
        return matrices
    axis = axis / norm
    x, y, z = [float(v) for v in axis]
    c = np.cos(angles)
    s = np.sin(angles)
    one_c = 1.0 - c
    rotation = matrices[:, :3, :3]
    rotation[:, 0, 0] = c + x * x * one_c
    rotation[:, 0, 1] = x * y * one_c - z * s
    rotation[:, 0, 2] = x * z * one_c + y * s
    rotation[:, 1, 0] = y * x * one_c + z * s
    rotation[:, 1, 1] = c + y * y * one_c
    rotation[:, 1, 2] = y * z * one_c - x * s
    rotation[:, 2, 0] = z * x * one_c - y * s
    rotation[:, 2, 1] = z * y * one_c + x * s
    rotation[:, 2, 2] = c + z * z * one_c
    return matrices


def _robotwin_translation_to_matrix(axis: np.ndarray, distance: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(axis, dtype=np.float32) * float(distance)
    return matrix


def _robotwin_translation_to_matrix_batch(axis: np.ndarray, distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    matrices = _robotwin_identity_batch(distances.shape[0])
    matrices[:, :3, 3] = np.asarray(axis, dtype=np.float32)[None] * distances[:, None]
    return matrices


class RobotWinLiteSampler:
    """Small URDF FK surface sampler for RoboTwin ALOHA assets."""

    def __init__(
        self,
        urdf_path: str,
        device: str = "cpu",
        link_whitelist: list[str] | None = None,
        root_transform: np.ndarray | tuple[tuple[float, ...], ...] | None = None,
        robot_type: str | None = None,
        side: str | None = None,
    ):
        self.urdf_path = str(urdf_path)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._link_whitelist = set(link_whitelist) if link_whitelist is not None else None
        self.root_transform = np.asarray(
            root_transform if root_transform is not None else np.eye(4, dtype=np.float32),
            dtype=np.float32,
        )
        if self.root_transform.shape != (4, 4):
            raise ValueError(
                f"root_transform must have shape (4, 4), got {self.root_transform.shape}"
            )
        self.robot_type = robot_type
        self.side = side
        self._child_to_joint: dict[str, dict] = {}
        self._link_mesh_specs: dict[str, dict] = {}
        self._presampled_points: dict[str, np.ndarray] | None = None
        self._presampled_normals: dict[str, np.ndarray] | None = None
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        self._presample_cache: dict[
            tuple[int, int], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = {}
        self._parse_urdf()

    def _parse_urdf(self) -> None:
        urdf_path = Path(self.urdf_path)
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            spec = self._find_link_mesh_spec(urdf_path.parent, link)
            if spec is not None:
                self._link_mesh_specs[link_name] = spec
        self.joint_names = []
        self.joint_defaults = {}
        self.joint_limits = {}
        for joint in root.findall("joint"):
            joint_type = joint.attrib.get("type", "fixed")
            child = joint.find("child").attrib["link"]
            parent = joint.find("parent").attrib["link"]
            axis_el = joint.find("axis")
            axis = (
                np.fromstring(axis_el.attrib.get("xyz", "1 0 0"), sep=" ", dtype=np.float32)
                if axis_el is not None
                else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            name = joint.attrib["name"]
            self._child_to_joint[child] = {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "origin": _robotwin_origin_to_matrix(joint.find("origin")),
                "axis": axis,
            }
            if joint_type != "fixed":
                self.joint_names.append(name)
                self.joint_defaults[name] = 0.0
                limit = joint.find("limit")
                lower = float(limit.attrib.get("lower", "-inf")) if limit is not None else -np.inf
                upper = float(limit.attrib.get("upper", "inf")) if limit is not None else np.inf
                self.joint_limits[name] = (lower, upper)

    def _find_link_mesh_spec(self, urdf_dir: Path, link: ET.Element) -> dict | None:
        for group_name in ("collision", "visual"):
            group = link.find(group_name)
            if group is None:
                continue
            mesh_el = group.find("geometry/mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename")
            if not filename:
                continue
            scale = np.fromstring(mesh_el.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float32)
            if scale.size != 3:
                scale = np.ones((3,), dtype=np.float32)
            return {
                "path": (urdf_dir / filename).resolve(),
                "origin": _robotwin_origin_to_matrix(group.find("origin")),
                "scale": scale.astype(np.float32),
            }
        return None

    def _load_mesh(self, link_name: str) -> trimesh.Trimesh:
        if link_name in self._mesh_cache:
            return self._mesh_cache[link_name]
        spec = self._link_mesh_specs[link_name]
        mesh = trimesh.load(spec["path"], force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not np.allclose(spec["scale"], 1.0):
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices * spec["scale"][None]
        self._mesh_cache[link_name] = mesh
        return mesh

    def presample(
        self, num_points: int, gripper_filter: str = "both", seed: int | None = None
    ) -> None:
        del gripper_filter
        if num_points <= 0:
            raise ValueError("num_points must be greater than 0")
        cache_key = None
        if seed is not None:
            cache_key = (int(num_points), int(seed) % (2**32 - 1))
            cached = self._presample_cache.get(cache_key)
            if cached is not None:
                self._presampled_points, self._presampled_normals = cached
                return
        link_names = [
            link_name
            for link_name in self._link_mesh_specs.keys()
            if self._link_whitelist is None or link_name in self._link_whitelist
        ]
        if not link_names:
            raise ValueError("No RobotWin links available for sampling")
        meshes = []
        for link_name in link_names:
            try:
                meshes.append((link_name, self._load_mesh(link_name)))
            except Exception:
                continue
        if not meshes:
            raise ValueError("No loadable RobotWin meshes found for sampling")
        areas = np.asarray([max(float(mesh.area), 1e-12) for _, mesh in meshes], dtype=np.float64)
        probs = areas / areas.sum()
        if int(num_points) >= len(meshes):
            counts = np.ones((len(meshes),), dtype=np.int64)
            remaining = int(num_points) - len(meshes)
            counts += np.floor(probs * remaining).astype(np.int64)
        else:
            counts = np.floor(probs * int(num_points)).astype(np.int64)
        while int(counts.sum()) < int(num_points):
            counts[int(np.argmax(probs - counts / max(int(num_points), 1)))] += 1
        while int(counts.sum()) > int(num_points):
            idx = int(np.argmax(counts))
            counts[idx] -= 1
        rng = np.random.RandomState(int(seed) % (2**32 - 1) if seed is not None else None)
        self._presampled_points = {}
        self._presampled_normals = {}
        for (link_name, mesh), count in zip(meshes, counts):
            count = int(count)
            if count <= 0:
                continue
            points, normals = deterministic_sample_surface(mesh, count, rng)
            self._presampled_points[link_name] = points.astype(np.float32)
            self._presampled_normals[link_name] = normals.astype(np.float32)
        if cache_key is not None:
            self._presample_cache[cache_key] = (self._presampled_points, self._presampled_normals)

    def _link_matrix(self, link_name: str, joint_values: dict[str, float]) -> np.ndarray:
        if link_name not in self._child_to_joint:
            return np.eye(4, dtype=np.float32)
        joint = self._child_to_joint[link_name]
        matrix = self._link_matrix(joint["parent"], joint_values) @ joint["origin"]
        value = float(joint_values.get(joint["name"], 0.0))
        if joint["type"] in {"revolute", "continuous"}:
            matrix = matrix @ _robotwin_axis_angle_to_matrix(joint["axis"], value)
        elif joint["type"] == "prismatic":
            matrix = matrix @ _robotwin_translation_to_matrix(joint["axis"], value)
        return matrix.astype(np.float32)

    @staticmethod
    def _joint_value_batch(
        joint_values: dict[str, np.ndarray], joint_name: str, batch_size: int
    ) -> np.ndarray:
        value = joint_values.get(joint_name)
        if value is None:
            return np.zeros((batch_size,), dtype=np.float32)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.full((batch_size,), float(arr[0]), dtype=np.float32)
        raise ValueError(f"joint {joint_name!r} has {arr.shape[0]} values, expected {batch_size}")

    @staticmethod
    def _joint_batch_size(joint_values: dict[str, np.ndarray]) -> int:
        for value in joint_values.values():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return int(np.asarray(value).reshape(-1).shape[0])
        return 1

    def _link_matrices_batch(
        self, link_names: tuple[str, ...] | list[str], joint_values: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        batch_size = self._joint_batch_size(joint_values)
        memo: dict[str, np.ndarray] = {}

        def compute(link_name: str) -> np.ndarray:
            if link_name in memo:
                return memo[link_name]
            if link_name not in self._child_to_joint:
                matrix = _robotwin_identity_batch(batch_size)
                memo[link_name] = matrix
                return matrix
            joint = self._child_to_joint[link_name]
            matrix = np.matmul(compute(joint["parent"]), joint["origin"][None])
            value = self._joint_value_batch(joint_values, joint["name"], batch_size)
            if joint["type"] in {"revolute", "continuous"}:
                matrix = np.matmul(
                    matrix, _robotwin_axis_angle_to_matrix_batch(joint["axis"], value)
                )
            elif joint["type"] == "prismatic":
                matrix = np.matmul(
                    matrix, _robotwin_translation_to_matrix_batch(joint["axis"], value)
                )
            matrix = matrix.astype(np.float32, copy=False)
            memo[link_name] = matrix
            return matrix

        return {link_name: compute(link_name) for link_name in link_names}

    def compute_points(
        self, joint_values: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points, colors, normals, _ = self.compute_points_with_link_transforms(
            joint_values, extra_link_names=[]
        )
        return (points, colors, normals)

    def compute_points_with_link_transforms(
        self, joint_values: dict[str, torch.Tensor], *, extra_link_names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self._presampled_points is None or self._presampled_normals is None:
            raise RuntimeError("presample(...) must be called before compute_points(...)")
        first = next(iter(joint_values.values()))
        batch_size = int(first.reshape(-1).shape[0])
        joint_np = {
            name: value.detach().cpu().numpy().reshape(-1) for name, value in joint_values.items()
        }
        total_points = sum((points.shape[0] for points in self._presampled_points.values()))
        all_points = np.zeros((batch_size, total_points, 3), dtype=np.float32)
        all_normals = np.zeros((batch_size, total_points, 3), dtype=np.float32)
        all_colors = np.zeros((batch_size, total_points, 3), dtype=np.uint8)
        all_colors[:, :, 0] = 255
        all_colors[:, :, 2] = 255
        mesh_link_names = list(self._presampled_points.keys())
        requested_links = list(dict.fromkeys(extra_link_names))
        link_matrices = self._link_matrices_batch(
            list(dict.fromkeys([*mesh_link_names, *requested_links])), joint_np
        )
        world_link_matrices = {
            name: np.matmul(self.root_transform[None], matrix)
            for name, matrix in link_matrices.items()
        }
        offset = 0
        for link_name, points in self._presampled_points.items():
            normals = self._presampled_normals[link_name]
            spec = self._link_mesh_specs[link_name]
            points_h = np.concatenate(
                [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
            )
            n = points.shape[0]
            matrix = np.matmul(world_link_matrices[link_name], spec["origin"][None])
            all_points[:, offset : offset + n] = np.einsum(
                "bij,nj->bni", matrix, points_h, optimize=True
            )[..., :3]
            all_normals[:, offset : offset + n] = np.einsum(
                "bij,nj->bni", matrix[:, :3, :3], normals, optimize=True
            )
            offset += n
        return (
            torch.as_tensor(all_points, device=self.device, dtype=self.dtype),
            torch.as_tensor(all_colors, device=self.device),
            torch.as_tensor(all_normals, device=self.device, dtype=self.dtype),
            {
                name: torch.as_tensor(
                    world_link_matrices[name], device=self.device, dtype=self.dtype
                )
                for name in requested_links
            },
        )


def _robotwin_side_seed(seed: int | None, side: str) -> int | None:
    if seed is None:
        return None
    return _stable_int_hash("robotwin", int(seed), side) % (2**32 - 1)


@lru_cache(maxsize=None)
def _load_robotwin_gripper_surface_points(
    asset_path: str, robot_type: str, side: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    path = Path(asset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed RobotWin gripper-point asset is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "robotwin_gripper_surface_points_v1":
        raise ValueError(f"Unsupported gripper-point asset schema in {path}")
    if normalize_robotwin_robot_type(payload.get("robot_type")) != robot_type:
        raise ValueError(f"Robot type mismatch in gripper-point asset {path}")
    if payload.get("coordinate_frame") != "urdf_mesh":
        raise ValueError(f"Gripper points in {path} must use the urdf_mesh coordinate frame")
    points_per_arm = int(payload.get("points_per_arm", -1))
    arm = payload.get("arms", {}).get(side)
    if not isinstance(arm, dict) or int(arm.get("count", -1)) != points_per_arm:
        raise ValueError(f"Invalid {side} arm metadata in {path}")
    spec = ROBOTWIN_EMBODIMENT_SPECS[robot_type]
    allowed_links = set(getattr(spec, f"{side}_gripper_links"))
    points_by_link: dict[str, np.ndarray] = {}
    normals_by_link: dict[str, np.ndarray] = {}
    for link in arm.get("links", []):
        link_name = str(link.get("link_name", ""))
        if link_name not in allowed_links or link_name in points_by_link:
            raise ValueError(f"Invalid or duplicate link {link_name!r} in {path}")
        points = np.asarray(link.get("points"), dtype=np.float32)
        normals = np.asarray(link.get("normals"), dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or normals.shape != points.shape
            or (not np.isfinite(points).all())
            or (not np.isfinite(normals).all())
        ):
            raise ValueError(f"Invalid points/normals for link {link_name!r} in {path}")
        points_by_link[link_name] = points
        normals_by_link[link_name] = normals
    if sum((len(points) for points in points_by_link.values())) != points_per_arm:
        raise ValueError(f"Point count does not match metadata in {path}")
    return (points_by_link, normals_by_link, points_per_arm)


def configure_robotwin_gripper_surface_points(
    sampler: RobotWinLiteSampler, *, robot_type: str, side: str, num_points: int, seed: int | None
) -> None:
    """Load the shared embodiment point set, with seeded sampling as fallback."""
    robot_type = normalize_robotwin_robot_type(robot_type)
    side = str(side).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    spec = ROBOTWIN_EMBODIMENT_SPECS[robot_type]
    if spec.gripper_surface_points_path is not None and int(num_points) == 20:
        points, normals, asset_count = _load_robotwin_gripper_surface_points(
            spec.gripper_surface_points_path, robot_type, side
        )
        if asset_count != int(num_points):
            raise ValueError(
                f"Fixed asset has {asset_count} points per arm, requested {num_points}"
            )
        sampler._presampled_points = {
            link_name: values.copy() for link_name, values in points.items()
        }
        sampler._presampled_normals = {
            link_name: values.copy() for link_name, values in normals.items()
        }
        return
    sampler.presample(num_points, seed=_robotwin_side_seed(seed, side))
