# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_kinematics as pk
import torch
import urchin as urdfpy

GRIPPER_KEYWORDS = ["finger", "knuckle", "robotiq", "gripper"]
GRIPPER_KEYWORDS += [
    "Part__Feature002_011",
    "Part__Feature005_005",
    "Part__Feature_011",
    "Part__Feature001_011",
    "Part__Feature005_000",
]
GRIPPER_KEYWORDS += ["Part__Feature_007", "Part__Feature001_006"]
R1PRO_INERTIAL_FRAME_OFFSET = torch.tensor([0.049262, 8.8054e-05, -0.18255], dtype=torch.float32)


def deterministic_sample_surface(mesh, count: int, rng: np.random.RandomState):
    """Deterministically sample points and normals on a mesh surface."""
    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.area_faces.astype(np.float64)
    p = areas / areas.sum()
    face_idx = rng.choice(len(faces), size=count, p=p)
    tri = verts[faces[face_idx]]
    r = rng.random((count, 2)).astype(np.float64)
    over = r.sum(axis=1) > 1.0
    r[over] = 1.0 - r[over]
    u = r[:, 0:1]
    v = r[:, 1:2]
    w = 1.0 - u - v
    points = (w * tri[:, 0] + u * tri[:, 1] + v * tri[:, 2]).astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-09)
    return (points, normals)


def is_gripper(mesh_name: str) -> bool:
    """Check if a mesh is a gripper."""
    return any((keyword.lower() in mesh_name.lower() for keyword in GRIPPER_KEYWORDS))


def is_gripper_filtered(mesh_name: str, gripper_filter: str) -> bool:
    """Check if a mesh should be included based on gripper filter."""
    if not is_gripper(mesh_name):
        return True
    mesh_name_lower = mesh_name.lower()
    if gripper_filter == "both":
        return True
    elif gripper_filter == "left":
        return "left" in mesh_name_lower and "right" not in mesh_name_lower
    elif gripper_filter == "right":
        return "right" in mesh_name_lower and "left" not in mesh_name_lower
    else:
        raise ValueError(
            f"Invalid gripper_filter: {gripper_filter}. Must be 'both', 'left', or 'right'"
        )


def get_mesh_name(mesh, idx):
    """Get mesh name from trimesh object."""
    try:
        return f"{mesh.source.file_name.lower()}_{idx}"
    except AttributeError:
        return (
            f"{mesh.metadata.get('name', mesh.metadata.get('file_name', 'unknown')).lower()}_{idx}"
        )


class RobotSampler:
    """GPU-accelerated robot point sampler using pytorch_kinematics for batched forward kinematics."""

    def __init__(
        self,
        urdf_path: str,
        gripper_only: bool = True,
        device: str = "cuda",
        apply_r1pro_inertial_frame_offset: bool = False,
        link_whitelist: Optional[List[str]] = None,
    ):
        """Initialize the GPU-accelerated RobotSampler."""
        self.urdf_path = urdf_path
        self.gripper_only = gripper_only
        self.apply_r1pro_inertial_frame_offset = apply_r1pro_inertial_frame_offset
        self._link_whitelist = set(link_whitelist) if link_whitelist is not None else None
        self.device = torch.device(device)
        self.dtype = torch.float32
        assert os.path.exists(urdf_path), f"URDF file not found: {urdf_path}"
        self._init_kinematics()
        self._presampled_points: Optional[Dict[str, torch.Tensor]] = None
        self._presampled_normals: Optional[Dict[str, torch.Tensor]] = None
        self._presample_key = None
        self._mesh_names: Optional[List[str]] = None
        self._mesh_offsets: Optional[Dict[str, torch.Tensor]] = None
        self._mesh_to_link: Optional[Dict[str, str]] = None
        self._initialize_mesh_mappings()

    def _init_kinematics(self):
        """Initialize pytorch_kinematics chain from URDF."""
        self.robot_urdf = urdfpy.URDF.load(self.urdf_path)
        with open(self.urdf_path, "rb") as f:
            urdf_string = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_string)
        self.chain = self.chain.to(device=self.device, dtype=self.dtype)
        self.joint_names = []
        self.joint_defaults = {}
        self.joint_limits = {}
        joints_list = self.chain.get_joints()
        raw_limits = self.chain.get_joint_limits()
        bulk_limits = list(zip(*raw_limits))
        for idx, joint in enumerate(joints_list):
            if getattr(joint, "joint_type", "fixed") == "fixed":
                continue
            joint_name = joint.name
            self.joint_names.append(joint_name)
            self.joint_defaults[joint_name] = 0.0
            lower, upper = bulk_limits[idx]
            assert lower is not None and upper is not None, f"Joint {joint_name} has no limits"
            self.joint_limits[joint_name] = (float(lower), float(upper))

    def _initialize_mesh_mappings(self) -> None:
        """Initialize mesh-to-link mappings and visual offsets without point sampling."""
        reference_cfg = self.joint_defaults.copy()
        fk_ref = self.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
        self._mesh_offsets = {}
        self._mesh_to_link = {}
        self._mesh_names = []
        mesh_ref_transforms = {}
        zero_cfg = {
            jn: torch.zeros(1, device=self.device, dtype=self.dtype) for jn in self.joint_names
        }
        link_tf_ref = self.chain.forward_kinematics(zero_cfg)
        for i, mesh in enumerate(fk_ref):
            mesh_name = get_mesh_name(mesh, i)
            self._mesh_names.append(mesh_name)
            mesh_T_np = fk_ref[mesh]
            mesh_T = torch.from_numpy(mesh_T_np.astype(np.float32)).to(
                self.device, dtype=self.dtype
            )
            mesh_ref_transforms[mesh_name] = mesh_T
            matched_link_name = None
            min_err = float("inf")
            for link_name, link_tf in link_tf_ref.items():
                link_mat = link_tf.get_matrix()[0].to(self.device, dtype=self.dtype)
                err = torch.norm(link_mat - mesh_T, p="fro").item()
                if err < min_err:
                    min_err = err
                    matched_link_name = link_name
            assert matched_link_name is not None, f"No link found for mesh '{mesh_name}'"
            link_T = (
                link_tf_ref[matched_link_name].get_matrix()[0].to(self.device, dtype=self.dtype)
            )
            offset_T = torch.matmul(torch.inverse(link_T), mesh_T)
            self._mesh_offsets[mesh_name] = offset_T.detach()
            self._mesh_to_link[mesh_name] = matched_link_name

        def rebind_mesh_to_link(mesh_name: str, link_name: str) -> None:
            link_T = link_tf_ref[link_name].get_matrix()[0].to(self.device, dtype=self.dtype)
            mesh_T = mesh_ref_transforms[mesh_name].to(self.device, dtype=self.dtype)
            self._mesh_offsets[mesh_name] = torch.matmul(torch.inverse(link_T), mesh_T).detach()
            self._mesh_to_link[mesh_name] = link_name

        if "panda_finger_joint1" in self.joint_names and "panda_finger_joint2" in self.joint_names:
            tri_meshes = sorted([mn for mn in self._mesh_names if "tri_finger.obj" in mn])
            if (
                len(tri_meshes) >= 2
                and "panda_rightfinger" in link_tf_ref
                and ("panda_leftfinger" in link_tf_ref)
            ):
                rebind_mesh_to_link(tri_meshes[0], "panda_rightfinger")
                rebind_mesh_to_link(tri_meshes[1], "panda_leftfinger")
            libero_finger_meshes = [
                mn
                for mn in self._mesh_names
                if "part__feature_007" in mn or "part__feature001_006" in mn
            ]
            if (
                len(libero_finger_meshes) >= 4
                and "panda_rightfinger" in link_tf_ref
                and ("panda_leftfinger" in link_tf_ref)
            ):
                for mesh_name in libero_finger_meshes[:2]:
                    rebind_mesh_to_link(mesh_name, "panda_rightfinger")
                for mesh_name in libero_finger_meshes[2:4]:
                    rebind_mesh_to_link(mesh_name, "panda_leftfinger")

    def presample(
        self, num_points: int, gripper_filter: str = "both", seed: int | None = None
    ) -> None:
        """Pre-sample points and normals from robot visual meshes."""
        assert num_points > 0, "num_points must be greater than 0"
        key = (int(num_points), gripper_filter, seed, tuple(sorted(self.joint_defaults.items())))
        if seed is not None and key == self._presample_key:
            return
        reference_cfg = self.joint_defaults.copy()
        fk_ref = self.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
        mesh_data = []
        for i, mesh in enumerate(fk_ref):
            mesh_name = get_mesh_name(mesh, i)
            if self.gripper_only and (not is_gripper(mesh_name)):
                continue
            if self.gripper_only:
                if not is_gripper_filtered(mesh_name, gripper_filter):
                    continue
            if mesh_name not in self._mesh_names:
                continue
            if self._link_whitelist is not None:
                link_name = self._mesh_to_link.get(mesh_name, None)
                if link_name not in self._link_whitelist:
                    continue
            effective_area = mesh.area
            if "hand_camera_part" in mesh_name.lower():
                effective_area *= 1e-06
            mesh_data.append({"name": mesh_name, "mesh": mesh, "area": effective_area})
        if not mesh_data:
            raise ValueError("No valid meshes found for sampling")
        total_area = sum((data["area"] for data in mesh_data))
        sampled_points = {}
        sampled_normals = {}
        rng = np.random.RandomState(int(seed) % (2**32 - 1) if seed is not None else None)
        allocated = 0
        for i, data in enumerate(mesh_data):
            mesh_name = data["name"]
            mesh = data["mesh"]
            if i == len(mesh_data) - 1:
                count = num_points - allocated
            else:
                ratio = data["area"] / total_area
                count = int(num_points * ratio)
            count = max(0, count)
            allocated += count
            if count > 0:
                points, normals = deterministic_sample_surface(mesh, count, rng)
                sampled_points[mesh_name] = torch.from_numpy(points).to(self.device)
                sampled_normals[mesh_name] = torch.from_numpy(normals).to(self.device)
            else:
                sampled_points[mesh_name] = torch.zeros(
                    (0, 3), dtype=self.dtype, device=self.device
                )
                sampled_normals[mesh_name] = torch.zeros(
                    (0, 3), dtype=self.dtype, device=self.device
                )
        self._presampled_points = sampled_points
        self._presampled_normals = sampled_normals
        self._presample_key = key if seed is not None else None

    def fk(
        self, joint_values: Dict[str, torch.Tensor], link_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform batched forward kinematics and return transforms for the *specified* links only."""
        assert self._mesh_to_link is not None, "Mesh mappings not initialized"
        complete_joint_values = {}
        batch_size = None
        for joint_name, values in joint_values.items():
            if joint_name in self.joint_names:
                if isinstance(values, (list, np.ndarray)):
                    values = torch.tensor(values, dtype=self.dtype, device=self.device)
                elif not isinstance(values, torch.Tensor):
                    values = torch.tensor([values], dtype=self.dtype, device=self.device)
                values = values.to(self.device, dtype=self.dtype)
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if batch_size is None:
                    batch_size = values.shape[0]
                elif values.shape[0] != batch_size:
                    raise ValueError("Inconsistent batch sizes in joint_values")
                complete_joint_values[joint_name] = values
        assert batch_size is not None, "No valid joint values provided"
        for joint_name in self.joint_names:
            if joint_name not in complete_joint_values:
                default_value = torch.full(
                    (batch_size,),
                    self.joint_defaults[joint_name],
                    dtype=self.dtype,
                    device=self.device,
                )
                complete_joint_values[joint_name] = default_value
        if link_names is None or len(link_names) == 0:
            link_names = list(dict.fromkeys(self._mesh_to_link.values()))
        frame_indices = self.chain.get_frame_indices(*link_names)
        transform_dict = self.chain.forward_kinematics(
            complete_joint_values, frame_indices=frame_indices
        )
        link_transforms = {ln: transform_dict[ln].get_matrix() for ln in link_names}
        if self.apply_r1pro_inertial_frame_offset:
            T_offset = torch.eye(4, dtype=self.dtype, device=self.device)
            T_offset[:3, 3] = R1PRO_INERTIAL_FRAME_OFFSET
            for ln in link_transforms.keys():
                T_link = link_transforms[ln]
                link_transforms[ln] = torch.matmul(T_offset, T_link)
        return link_transforms

    def compute_points(
        self, joint_values: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute robot points, normals, and colors for all timesteps using batched operations."""
        points, colors, normals, _ = self.compute_points_with_link_transforms(joint_values)
        return (points, colors, normals)

    def compute_points_with_link_transforms(
        self, joint_values: Dict[str, torch.Tensor], *, extra_link_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sampled points and requested link transforms with one FK call."""
        mesh_link_names = (
            list(dict.fromkeys(self._mesh_to_link.values())) if self._mesh_to_link else []
        )
        requested_link_names = list(dict.fromkeys(extra_link_names or []))
        fk_link_names = list(dict.fromkeys([*mesh_link_names, *requested_link_names]))
        link_transforms = self.fk(joint_values, link_names=fk_link_names)
        batch_size = next(iter(link_transforms.values())).shape[0] if link_transforms else 1
        total_points = sum((points.shape[0] for points in self._presampled_points.values()))
        all_points = torch.zeros(
            (batch_size, total_points, 3), dtype=self.dtype, device=self.device
        )
        all_normals = torch.zeros(
            (batch_size, total_points, 3), dtype=self.dtype, device=self.device
        )
        all_colors = torch.zeros(
            (batch_size, total_points, 3), dtype=torch.uint8, device=self.device
        )
        all_colors[:, :, 0] = 255
        all_colors[:, :, 2] = 255
        point_idx = 0
        for mesh_name in self._presampled_points.keys():
            local_points = self._presampled_points[mesh_name]
            local_normals = self._presampled_normals[mesh_name]
            n_points = local_points.shape[0]
            if n_points == 0:
                continue
            if self._mesh_to_link is not None:
                link_name = self._mesh_to_link.get(mesh_name, None)
                if link_name is not None and link_name in link_transforms:
                    T_link = link_transforms[link_name]
                    offset_T = self._mesh_offsets[mesh_name]
                    T = torch.matmul(T_link, offset_T)
                else:
                    continue
            else:
                continue
            local_points_homo = torch.cat(
                [local_points, torch.ones(n_points, 1, dtype=self.dtype, device=self.device)], dim=1
            )
            world_points_homo = torch.bmm(
                T, local_points_homo.T.unsqueeze(0).repeat(batch_size, 1, 1)
            )
            world_points = world_points_homo[:, :3, :].transpose(1, 2)
            R = T[:, :3, :3]
            world_normals = torch.bmm(R, local_normals.T.unsqueeze(0).repeat(batch_size, 1, 1))
            world_normals = world_normals.transpose(1, 2)
            all_points[:, point_idx : point_idx + n_points] = world_points
            all_normals[:, point_idx : point_idx + n_points] = world_normals
            point_idx += n_points
        requested_transforms = {
            name: link_transforms[name] for name in requested_link_names if name in link_transforms
        }
        return (all_points, all_colors, all_normals, requested_transforms)
