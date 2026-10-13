# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions copyright (c) 2023 Stanford Vision and Learning Group.
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Includes transformation routines from OmniGibson (MIT) and deoxys_control
# (Apache-2.0), adapted through PointWorld.
# http://github.com/StanfordVL/OmniGibson
# http://github.com/UT-Austin-RPL/deoxys_control
#
# MIT License
#
# Copyright (c) 2023 Stanford Vision and Learning Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pose conversions used by the online robot adapters."""

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from numba import njit
except ImportError:

    def njit(**kwargs):
        return lambda function: function


@njit(cache=True, fastmath=True)
def _mat2quat_single(R):
    """
    3×3 rotation matrix → quaternion (x,y,z,w)
    Uses Shoemake’s 1992 branch-on-trace method.
    """
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array((x, y, z, w), dtype=R.dtype)


@njit(cache=True, fastmath=True)
def _quat2mat_single(q):
    """
    Quaternion (x,y,z,w) → 3×3 rotation matrix
    """
    x, y, z, w = q
    xx, yy, zz = (x * x, y * y, z * z)
    xy, xz, yz = (x * y, x * z, y * z)
    wx, wy, wz = (w * x, w * y, w * z)
    M = np.empty((3, 3), dtype=q.dtype)
    M[0, 0] = 1.0 - 2.0 * (yy + zz)
    M[0, 1] = 2.0 * (xy - wz)
    M[0, 2] = 2.0 * (xz + wy)
    M[1, 0] = 2.0 * (xy + wz)
    M[1, 1] = 1.0 - 2.0 * (xx + zz)
    M[1, 2] = 2.0 * (yz - wx)
    M[2, 0] = 2.0 * (xz - wy)
    M[2, 1] = 2.0 * (yz + wx)
    M[2, 2] = 1.0 - 2.0 * (xx + yy)
    return M


@njit(cache=True, fastmath=True)
def _mat2quat_kernel(poses_mat, out):
    N = poses_mat.shape[0]
    for i in range(N):
        out[i, :3] = poses_mat[i, :3, 3]
        out[i, 3:] = _mat2quat_single(poses_mat[i, :3, :3])


@njit(cache=True, fastmath=True)
def _quat2mat_kernel(poses_quat, out):
    N = poses_quat.shape[0]
    for i in range(N):
        out[i, :3, 3] = poses_quat[i, :3]
        R = _quat2mat_single(poses_quat[i, 3:])
        out[i, 0, :3] = R[0]
        out[i, 1, :3] = R[1]
        out[i, 2, :3] = R[2]


def convert_pose_mat2quat(poses_mat: np.ndarray) -> np.ndarray:
    """
    (4,4)  or (N,4,4) → (7,) or (N,7)  [x y z qx qy qz qw]
    """
    was_single = poses_mat.ndim == 2
    if was_single:
        poses_mat = poses_mat[None]
    out = np.empty((poses_mat.shape[0], 7), dtype=poses_mat.dtype)
    _mat2quat_kernel(poses_mat, out)
    return out[0] if was_single else out


def convert_pose_quat2mat(poses_quat: np.ndarray) -> np.ndarray:
    """
    (7,) or (N,7) → (4,4) or (N,4,4)
    """
    was_single = poses_quat.ndim == 1
    if was_single:
        poses_quat = poses_quat[None]
    out = np.empty((poses_quat.shape[0], 4, 4), dtype=poses_quat.dtype)
    out[:, 3, :] = np.array((0.0, 0.0, 0.0, 1.0), dtype=poses_quat.dtype)
    _quat2mat_kernel(poses_quat, out)
    return out[0] if was_single else out


def as_homogeneous(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"homogeneous transform must have shape [4,4], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("homogeneous transform contains non-finite values")
    return value


def pose_xyzw_to_matrix(pose: object) -> np.ndarray:
    """Convert ``[x,y,z,qx,qy,qz,qw]`` to a homogeneous transform."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"pose must have shape [7], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("pose contains non-finite values")
    quaternion = value[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-08:
        raise ValueError("pose quaternion has near-zero norm")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def matrix_to_pose_xyzw(matrix: object) -> np.ndarray:
    """Convert a homogeneous transform to ``[x,y,z,qx,qy,qz,qw]``."""
    transform = as_homogeneous(matrix)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.concatenate((transform[:3, 3], quaternion)).astype(np.float32)
