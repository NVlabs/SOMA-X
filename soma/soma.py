# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix

from .correctives_model import CorrectivesMLP
from .geometry._warp_init import ensure_warp_initialized
from .geometry.batched_skinning import BatchedSkinning
from .geometry.lbs import batch_rodrigues
from .geometry.rig_utils import (
    apply_joint_orient_local,
    precompute_joint_orient,
)
from .geometry.skeleton_transfer import SkeletonTransfer
from .identity_model import create_identity_model
from .units import Unit


class SOMALayer(nn.Module):
    def __init__(
        self,
        data_root=None,
        low_lod=False,
        device="cuda",
        identity_model_type="mhr",
        mode="warp",
        output_unit=Unit.METERS,
        identity_model_kwargs=None,
    ):
        super().__init__()

        self.identity_model_kwargs = (
            identity_model_kwargs if identity_model_kwargs is not None else {}
        )

        if data_root is None or not Path(data_root).exists():
            if data_root is not None:
                print(f"data_root '{data_root}' not found, downloading assets from HuggingFace...")
            else:
                print("No data_root provided, downloading assets from HuggingFace...")
            from .assets import get_assets_dir

            data_root = get_assets_dir()

        data_root = Path(data_root)

        # Check for core asset file
        core_asset = data_root / "SOMA_neutral.npz"
        if not core_asset.exists():
            raise FileNotFoundError(
                f"Core asset 'SOMA_neutral.npz' not found in '{data_root}'.\n"
                "Please ensure the assets are correctly downloaded and extracted."
            )

        self.identity_model_type = identity_model_type
        try:
            self.rig_data = np.load(core_asset, allow_pickle=False)
        except Exception as e:
            raise RuntimeError(
                f"Error loading core asset 'SOMA_neutral.npz': {e}\n"
                "Please ensure the assets are correctly downloaded with 'git lfs pull'."
            ) from e
        self.device = device
        self.data_root = data_root
        self.low_lod = low_lod
        self.mode = mode
        self.output_unit = output_unit

        # Pre-initialize Warp in the main process so DataLoader forked workers
        # inherit _initialized=True and skip wp.init() (avoids CUDA error 3 in workers).
        ensure_warp_initialized()

        shape_mean = torch.from_numpy(self.rig_data["mean"]).to(device)
        self.register_buffer("shape_mean", shape_mean, persistent=False)
        self.register_buffer(
            "shape_pca", torch.from_numpy(self.rig_data["shapedirs"]).to(device), persistent=False
        )
        self.register_buffer(
            "shape_eigenvalues",
            torch.from_numpy(self.rig_data["eigenvalues"]).to(device),
            persistent=False,
        )
        self.num_shape_components = self.shape_pca.shape[0]
        nv_lod_mid_to_low = self.rig_data["lod_mid_to_low"]
        self.parents = [i - 1 for i in self.rig_data["joint_parent_ids"]][1:]

        bind_shape = torch.from_numpy(self.rig_data["bind_shape"]).to(device)
        skinning_weights_np = csc_matrix(
            (
                self.rig_data["skinning_weights_data"],
                self.rig_data["skinning_weights_indices"],
                self.rig_data["skinning_weights_indptr"],
            ),
            shape=self.rig_data["skinning_weights_shape"],
        ).todense()
        skinning_weights = skinning_weights_np
        if low_lod:
            self.register_buffer(
                "nv_lod_mid_to_low",
                torch.from_numpy(nv_lod_mid_to_low).long().to(device),
                persistent=False,
            )
            self.register_buffer(
                "faces",
                torch.from_numpy(self.rig_data["triangles_low"]).to(device),
                persistent=False,
            )
            self.register_buffer("bind_shape", bind_shape[nv_lod_mid_to_low], persistent=False)
            skinning_weights = skinning_weights[nv_lod_mid_to_low]
        else:
            self.register_buffer(
                "faces", torch.from_numpy(self.rig_data["triangles"]).to(device), persistent=False
            )
            self.register_buffer("bind_shape", bind_shape.to(device), persistent=False)
            self.nv_lod_mid_to_low = None

        facial_inner_geometry = np.concatenate(
            [
                self.rig_data["segment_eye_bags"],
                self.rig_data["segment_mouth_bag"],
            ]
        )
        facial_inner_geometry = torch.from_numpy(facial_inner_geometry).to(device)

        if low_lod:
            num_high_verts = bind_shape.shape[0]
            inverse_lod_map = torch.full((num_high_verts,), -1, dtype=torch.long, device=device)
            inverse_lod_map[self.nv_lod_mid_to_low] = torch.arange(
                self.nv_lod_mid_to_low.shape[0], device=device
            )
            facial_low = inverse_lod_map[facial_inner_geometry.long()]
            facial_inner_geometry = facial_low[facial_low >= 0]

        self.register_buffer(
            "skinning_weights", torch.from_numpy(skinning_weights).to(device), persistent=False
        )
        self.register_buffer(
            "joint_parent_ids",
            torch.from_numpy(self.rig_data["joint_parent_ids"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "bind_pose_world",
            torch.from_numpy(self.rig_data["bind_pose_world"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "bind_pose_local",
            torch.from_numpy(self.rig_data["bind_pose_local"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "t_pose_world",
            torch.from_numpy(self.rig_data["t_pose_world"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "t_pose_local",
            torch.from_numpy(self.rig_data["t_pose_local"]).to(device),
            persistent=False,
        )
        self.register_buffer("facial_inner_geometry", facial_inner_geometry, persistent=False)

        self.skeleton_transfer = SkeletonTransfer(
            self.joint_parent_ids,
            self.bind_pose_world,
            self.bind_shape,
            self.skinning_weights,
            rotation_method="kabsch",
            vertex_ids_to_exclude=self.facial_inner_geometry,
        )

        self.batched_skinning = BatchedSkinning(
            self.joint_parent_ids,
            self.skinning_weights,
            self.bind_pose_world,
            self.bind_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
        )

        self.identity_model = create_identity_model(
            identity_model_type,
            data_root,
            low_lod,
            device,
            output_unit=output_unit,
            nv_lod_mid_to_low=self.nv_lod_mid_to_low,
            soma_low_lod_faces=self.faces if low_lod else None,
            vertex_ids_to_exclude=self.facial_inner_geometry,
            **self.identity_model_kwargs,
        )

        self.correctives_model = CorrectivesMLP.load_checkpoint(
            self.data_root / "correctives_model.pt",
            map_location=device,
            v_index_map=self.nv_lod_mid_to_low,
            output_unit=output_unit,
        )
        self._corrective_config = {"first_joint_index": 0, "input_type": "tfm"}

        if self.t_pose_world is not None:
            self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
                self.t_pose_world, self.joint_parent_ids
            )
        else:
            self._t_pose_orient = None
            self._t_pose_orient_parent_T = None

        self._cached_rest_shape = None
        self._cached_bind_transforms_world = None
        self._cached_global_scale = 1.0

    def _apply(self, fn):
        super()._apply(fn)
        self.device = self.bind_pose_world.device
        self.dtype = self.bind_pose_world.dtype
        # BatchedSkinning is not an nn.Module, so its internal tensors (bone weights,
        # joint orient, skeleton levels, etc.) are not moved by the default _apply.
        # Reinitialize it from the registered buffers, which are now on the new device.
        self.batched_skinning = BatchedSkinning(
            self.joint_parent_ids,
            self.skinning_weights,
            self.bind_pose_world,
            self.bind_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
        )
        # _t_pose_orient / _t_pose_orient_parent_T are plain attributes (not buffers).
        # Recompute them on the new device.
        if self.t_pose_world is not None:
            self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
                self.t_pose_world, self.joint_parent_ids
            )
        return self

    def _pad_poses(self, poses_rot):
        ident = (
            torch.eye(3, device=poses_rot.device).unsqueeze(0).repeat(poses_rot.shape[0], 1, 1, 1)
        )
        poses_rot = torch.cat([ident, poses_rot], dim=1)
        return poses_rot

    def _apply_joint_orient(self, poses_rot_relative):
        """Convert relative-to-T-pose rotations to absolute skinning space.
        Matches BatchedSkinning.pose() when joint_orient is set."""
        if self._t_pose_orient is None:
            return poses_rot_relative
        return apply_joint_orient_local(
            poses_rot_relative, self._t_pose_orient, self._t_pose_orient_parent_T
        )

    # ------------------------------------------------------------------
    # Public accessors for skeleton topology and T-pose data
    # ------------------------------------------------------------------

    @property
    def joint_names(self):
        """List of joint name strings (length J), indexed to match rotation arrays."""
        return list(self.rig_data["joint_names"])

    def get_tpose_rotations(self):
        """Return T-pose rotation matrices for all joints.

        Returns:
            (J, 3, 3) tensor — the upper-left 3×3 rotation block of
            each joint's T-pose world transform.
        """
        return self.t_pose_world[:, :3, :3]

    def prepare_identity(
        self,
        identity_coeffs,
        scale_params=None,
        repose_to_bind_pose=True,
        global_scale=1.0,
    ):
        """Compute and cache the rest shape and skeleton for a given identity.

        Call this once when the identity changes, then call :meth:`pose` for
        each new pose.  This avoids recomputing the identity model and skeleton
        transfer on every frame.

        Args:
            identity_coeffs: (batch_size, C) identity coefficients.
            scale_params: (batch_size, S) optional scale parameters (required for MHR).
            global_scale: Uniform scale factor applied to the rest shape (default 1.0).
                Can be a scalar or a (batch_size,) tensor.
        """
        self._cached_rest_shape = self.identity_model(
            identity_coeffs, scale_params, global_scale=global_scale
        )
        self._cached_bind_transforms_world = self.skeleton_transfer.fit(self._cached_rest_shape)
        if repose_to_bind_pose:
            self.batched_skinning.rebind(
                self._cached_bind_transforms_world, self._cached_rest_shape
            )
            self._cached_rest_shape, self._cached_bind_transforms_world = (
                self.batched_skinning.pose(
                    local_rotations=self.bind_pose_local[..., :3, :3],
                    hips_translations=self.bind_pose_local[..., 1, :3, 3],
                    align_translation=torch.tensor([0, 0, 0], device=self.device),
                    return_transforms=True,
                    absolute_pose=True,
                )
            )
        self._cached_global_scale = global_scale

    def pose(
        self,
        poses,
        transl=None,
        pose2rot=True,
        apply_correctives=True,
        absolute_pose=False,
    ):
        """Pose the cached identity.  :meth:`prepare_identity` must be called first.

        Args:
            poses: (batch_size, num_joints, 3) rotation vectors, or
                   (batch_size, num_joints, 3, 3) rotation matrices when *pose2rot=False*.
            transl: (batch_size, 3) root translation.
            pose2rot: if True, convert rotation vectors to matrices via Rodrigues.
            apply_correctives: if True, apply pose-dependent corrective offsets.
            absolute_pose: if True, interpret input poses as absolute rotations instead of relative to joint orient.
        Returns:
            dict with 'vertices' (batch_size, V, 3) and 'joints' (batch_size, J, 3).
        """
        if self._cached_rest_shape is None or self._cached_bind_transforms_world is None:
            raise RuntimeError("No cached identity. Call prepare_identity() before pose().")

        rest_shape = self._cached_rest_shape
        rest_bind_transforms_world = self._cached_bind_transforms_world

        batch_size, num_joints = poses.shape[:2]
        if transl is None:
            transl = torch.zeros(batch_size, 3, device=poses.device)
        if pose2rot:
            poses_rot = batch_rodrigues(poses.view(-1, 3)).view(batch_size, num_joints, 3, 3)
        else:
            poses_rot = poses.view(batch_size, num_joints, 3, 3)

        poses_rot = self._pad_poses(poses_rot)

        if apply_correctives and self.correctives_model is not None:
            if absolute_pose:
                correctives_input = poses_rot
            else:
                correctives_input = self._apply_joint_orient(poses_rot)
            out_correctives = self.correctives_model(correctives_input)["out"]
            gs = self._cached_global_scale
            if isinstance(gs, torch.Tensor):
                out_correctives = out_correctives * gs.reshape(-1, 1, 1)
            elif gs != 1.0:
                out_correctives = out_correctives * gs
            rest_shape = rest_shape + out_correctives

        if rest_bind_transforms_world.shape[0] == 1 and batch_size > 1:
            bind_transforms = rest_bind_transforms_world.expand(batch_size, -1, -1, -1)
            rest_shape = rest_shape.expand(batch_size, -1, -1)
        else:
            bind_transforms = rest_bind_transforms_world
        self.batched_skinning.rebind(bind_transforms, rest_shape)

        vertices, T_world = self.batched_skinning.pose(
            local_rotations=poses_rot,
            hips_translations=transl,
            return_transforms=True,
            absolute_pose=absolute_pose,
        )

        joints = T_world[..., :3, 3]
        return {
            "vertices": vertices,
            "joints": joints[:, 1:, :],
        }

    def forward(
        self,
        poses,
        identity_coeffs,
        scale_params=None,
        transl=None,
        pose2rot=True,
        apply_correctives=True,
        absolute_pose=False,
        global_scale=1.0,
    ):
        """
        Args:
            poses: (batch_size, num_joints, 3)
            identity_coeffs: (batch_size, 45)
            scale_params: (batch_size, 68) optional body-part scales (required for MHR).
            transl: (batch_size, 3)
            global_scale: Uniform scale factor (default 1.0). Scalar or (batch_size,) tensor.
        Returns:
            vertices: (batch_size, num_vertices, 3)
            joints: (batch_size, num_joints, 3)
        """
        self.prepare_identity(
            identity_coeffs,
            scale_params,
            repose_to_bind_pose=apply_correctives,
            global_scale=global_scale,
        )
        return self.pose(
            poses,
            transl=transl,
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            absolute_pose=absolute_pose,
        )
