# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private native-MHR pose inversion utilities.

This module recovers native MHR pose/model parameters from MHR-topology posed
vertices.  It intentionally stays separate from public SOMA conversion code so
MHR-specific DOF handling, co-located ankle distribution, and parameter-matrix
projection can be removed or hidden for public releases.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..geometry.lbs_warp import linear_blend_skinning
from ..geometry.rig_utils import joint_world_to_local
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..geometry.transforms import (
    SE3_from_Rt,
    SE3_inverse,
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    quaternion_xyzw_to_matrix,
)
from .pose_inversion import (
    PoseInversionResult,
    _bexpand,
    _bexpand4,
    _build_world_transforms,
    _precompute_refit_cache,
    _refit_joint,
    _run_refit_passes,
    _to_sparse_weights,
    _update_root_translation,
)

_BODY_PASS_GROUPS = [
    ["root"],
    ["c_spine0", "l_upleg", "r_upleg"],
    ["c_spine1", "l_lowleg", "r_lowleg"],
    ["c_spine2", "l_foot", "r_foot"],
    ["c_spine3", "l_talocrural", "r_talocrural"],
    ["l_clavicle", "r_clavicle", "c_neck", "l_subtalar", "r_subtalar"],
    ["l_uparm", "r_uparm", "c_head", "l_transversetarsal", "r_transversetarsal"],
    ["l_lowarm", "r_lowarm", "l_ball", "r_ball"],
    ["l_wrist_twist", "r_wrist_twist"],
    ["l_wrist", "r_wrist"],
]

_FINGER_PASS_GROUPS = [
    ["l_thumb0", "r_thumb0", "l_pinky0", "r_pinky0"],
    [
        "l_index1",
        "r_index1",
        "l_middle1",
        "r_middle1",
        "l_ring1",
        "r_ring1",
        "l_pinky1",
        "r_pinky1",
        "l_thumb1",
        "r_thumb1",
    ],
    [
        "l_index2",
        "r_index2",
        "l_middle2",
        "r_middle2",
        "l_ring2",
        "r_ring2",
        "l_pinky2",
        "r_pinky2",
        "l_thumb2",
        "r_thumb2",
    ],
    [
        "l_index3",
        "r_index3",
        "l_middle3",
        "r_middle3",
        "l_ring3",
        "r_ring3",
        "l_pinky3",
        "r_pinky3",
        "l_thumb3",
        "r_thumb3",
    ],
]

_COLOCATED_ANKLE_PAIRS = (("l_foot", "l_talocrural"), ("r_foot", "r_talocrural"))
_FLEXIBLE_SLICE = slice(130, 136)
_DISABLED_POSE_PARAM_IDS = (
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    19,
    20,
    21,
    22,
    23,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
)
_ACTIVE_SPINE_PARAM_BOUNDS = {
    7: (-0.9, 0.9),
    9: (-0.7, 0.7),
    11: (-0.5, 1.5),
    13: (-0.9, 0.9),
    15: (-0.7, 0.7),
    17: (-0.5, 1.5),
}
_REDUCED_DOF_REFIT_ITERS = 8
_REDUCED_DOF_REFIT_LR = 5e-2


class MHRPoseInversionResult(dict):
    """Dictionary result with attribute access for internal MHR inversion."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _joint_names_from_dof_names(joint_dof_names):
    n_dofs = len(joint_dof_names)
    n_joints = n_dofs // 7
    return [str(joint_dof_names[j * 7]).split(".", 1)[0] for j in range(n_joints)]


def _mhr_skeleton_state_to_transforms(skel_state):
    """Convert MHR TorchScript skeleton-state rows to world transforms."""
    R = quaternion_xyzw_to_matrix(skel_state[..., 3:7])
    R = R * skel_state[..., 7, None, None]
    return SE3_from_Rt(R, skel_state[..., :3])


def _load_parameter_transform(device, dtype, npz_path, ridge_lambda=1e-7):
    with np.load(npz_path, allow_pickle=False) as full_data:
        parameter_transform_np = full_data["parameter_transform"]
        parameter_names = [str(x) for x in full_data["parameter_names"]]
        joint_dof_names = [str(x) for x in full_data["joint_dof_names"]]
        pre_rotations = full_data["pre_rotations"]

    joint_names = _joint_names_from_dof_names(joint_dof_names)
    n_joints = len(joint_names)
    root_idx = joint_names.index("root")
    main_joint_mask = torch.tensor(
        [not name.endswith("_proc") for name in joint_names],
        device=device,
        dtype=torch.bool,
    )
    num_main = int(main_joint_mask.sum().item())
    joint_orients = torch.from_numpy(pre_rotations).to(device=device, dtype=dtype)

    pose_skip = torch.tensor(
        [name.startswith("scale_") for name in parameter_names[:204]],
        device=device,
        dtype=torch.bool,
    )
    pose_indices = torch.where(~pose_skip)[0]

    root_dofs = root_idx * 7 + torch.arange(6, device=device, dtype=torch.long)
    other_j = torch.cat(
        [
            torch.arange(root_idx, device=device, dtype=torch.long),
            torch.arange(root_idx + 1, n_joints, device=device, dtype=torch.long),
        ]
    )
    other_dofs = (
        other_j.unsqueeze(1) * 7 + torch.tensor([3, 4, 5], device=device, dtype=torch.long)
    ).reshape(-1)
    dof_indices = torch.cat([root_dofs, other_dofs])

    non_root_main = torch.cat([main_joint_mask[:root_idx], main_joint_mask[root_idx + 1 :]])
    dof_mask = torch.cat(
        [
            torch.ones(6, device=device, dtype=torch.bool),
            non_root_main.repeat_interleave(3),
        ]
    )

    P_full = torch.from_numpy(parameter_transform_np).to(device=device, dtype=torch.float64)
    P_pose = P_full[dof_indices][:, pose_indices]
    P_pose_main = P_pose[dof_mask]

    n_pose = P_pose_main.shape[1]
    eye = torch.eye(n_pose, device=device, dtype=P_pose_main.dtype)
    AtA_lam = P_pose_main.T @ P_pose_main + ridge_lambda * eye
    P_inv_pose_main = torch.linalg.solve(AtA_lam, P_pose_main.T).T.to(dtype=dtype)

    pose_names = [name for name in parameter_names[:204] if not name.startswith("scale_")]

    return {
        "joint_names": joint_names,
        "joint_orients": joint_orients,
        "main_joint_mask": main_joint_mask,
        "num_main": num_main,
        "pose_indices": pose_indices,
        "pose_names": pose_names,
        "P_pose_main": P_pose_main.to(dtype=dtype),
        "P_inv_pose_main": P_inv_pose_main,
        "dof_mask": dof_mask,
    }


def _get_dof_masks(pt_data):
    """Return local Euler-axis masks for MHR joints with fewer than 3 active DOFs."""
    joint_names = pt_data["joint_names"]
    main_joint_mask = pt_data["main_joint_mask"]
    root_idx = joint_names.index("root")
    non_root_main = [
        j for j in range(len(joint_names)) if j != root_idx and bool(main_joint_mask[j].item())
    ]

    masks = {}
    P = pt_data["P_pose_main"]
    for k, j_idx in enumerate(non_root_main):
        jname = joint_names[j_idx]
        row_start = 6 + 3 * k
        active = tuple(P[row_start + d].abs().max().item() > 1e-10 for d in range(3))
        if sum(active) < 3:
            masks[jname] = active
    return masks


def _groups_to_indices(groups, joint_names):
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    out = []
    for group in groups:
        idxs = [name_to_idx[name] for name in group if name in name_to_idx]
        if idxs:
            out.append(idxs)
    return out


def _as_batched_tensor(value, shape, batch_size, device, dtype, name):
    if value is None:
        return torch.zeros(batch_size, *shape, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape == shape:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == len(shape) + 1 and tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, *shape)
    if tensor.shape != (batch_size, *shape):
        raise ValueError(f"{name} must have shape {(batch_size, *shape)} or {shape}.")
    return tensor


def _constrain_dof_local(pose_local, joint_names, pt_data, dof_masks):
    joint_orients = pt_data["joint_orients"]
    pt_joint_names = pt_data["joint_names"]
    B = pose_local.shape[0]
    zeros = torch.zeros(B, device=pose_local.device, dtype=pose_local.dtype)

    for jname, (active_rx, active_ry, active_rz) in dof_masks.items():
        if jname not in joint_names or jname not in pt_joint_names:
            continue
        j_idx = joint_names.index(jname)
        pt_idx = pt_joint_names.index(jname)

        R_orient = joint_orients[pt_idx]
        R_local = R_orient.T.unsqueeze(0) @ pose_local[:, j_idx, :3, :3]
        euler = matrix_to_euler_xyz(R_local)
        constrained = torch.stack(
            [
                euler[:, 0] if active_rx else zeros,
                euler[:, 1] if active_ry else zeros,
                euler[:, 2] if active_rz else zeros,
            ],
            dim=-1,
        )
        pose_local[:, j_idx, :3, :3] = R_orient.unsqueeze(0) @ euler_xyz_to_matrix(constrained)


def _refit_reduced_dof_joint(
    pose_local,
    target,
    j_idx,
    W,
    D,
    cache,
    jcache,
    pt_data,
    active_axes,
    *,
    iters,
    lr,
):
    """Re-fit one MHR joint directly in its active local Euler axes."""
    joint_name = cache["joint_names"][j_idx]
    pt_joint_names = pt_data["joint_names"]
    if joint_name not in pt_joint_names:
        return

    pt_idx = pt_joint_names.index(joint_name)
    R_orient = pt_data["joint_orients"][pt_idx]
    B = pose_local.shape[0]
    active_ids = torch.tensor(
        [axis for axis, is_active in enumerate(active_axes) if is_active],
        device=pose_local.device,
        dtype=torch.long,
    )

    if len(active_ids) == 0:
        pose_local[:, j_idx, :3, :3] = R_orient.unsqueeze(0)
        return

    arm_vids = jcache["arm_vids"]
    bind_verts = jcache["bind_verts_arm"]
    sub_bw = jcache["sub_bone_weights"]
    sub_bi = jcache["sub_bone_indices"]
    non_bw = jcache["non_bone_weights"]
    non_bi = jcache["non_bone_indices"]
    sub_w_sum = jcache["sub_weight_sum"]
    bv = _bexpand(bind_verts, B)

    q_world = linear_blend_skinning(bv, sub_bw, sub_bi, D).detach()
    c_xyz = linear_blend_skinning(bv, non_bw, non_bi, D).detach()

    W_p_inv = SE3_inverse(W[:, j_idx]).detach()
    R_inv = W_p_inv[:, :3, :3]
    t_inv = W_p_inv[:, :3, 3]

    sw = sub_w_sum.view(1, -1, 1)
    src = (q_world @ R_inv.transpose(-2, -1) + t_inv.unsqueeze(1) * sw).detach()

    p_parent = W[:, j_idx, :3, 3].detach()
    tgt = (target[:, arm_vids, :] - c_xyz - p_parent.unsqueeze(1) * sw).detach()

    parent_idx = cache["parent_ids_list"][j_idx]
    if parent_idx == j_idx:
        parent_world = None
    else:
        parent_world = W[:, parent_idx, :3, :3].detach()

    R_current = R_orient.T.unsqueeze(0) @ pose_local[:, j_idx, :3, :3]
    current_euler = matrix_to_euler_xyz(R_current).detach()
    params = current_euler[:, active_ids].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=lr)
    best_loss = float("inf")
    best_params = params.detach().clone()

    with torch.enable_grad():
        for _ in range(iters):
            optimizer.zero_grad()
            euler = torch.zeros_like(current_euler)
            euler[:, active_ids] = params
            R_local = R_orient.unsqueeze(0) @ euler_xyz_to_matrix(euler)
            if parent_world is None:
                R_world = R_local
            else:
                R_world = parent_world @ R_local
            pred = src @ R_world.transpose(-2, -1)
            loss = torch.nn.functional.mse_loss(pred, tgt)
            cur_loss = float(loss.detach().cpu())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_params = params.detach().clone()
            loss.backward()
            optimizer.step()

    euler = torch.zeros_like(current_euler)
    euler[:, active_ids] = best_params
    pose_local[:, j_idx, :3, :3] = R_orient.unsqueeze(0) @ euler_xyz_to_matrix(euler)


def _redistribute_colocated_transforms(pose_local, joint_names, pt_data):
    """Split co-located MHR ankle rotations between foot and talocrural joints."""
    joint_orients = pt_data["joint_orients"]
    pt_joint_names = pt_data["joint_names"]
    B = pose_local.shape[0]
    zeros = torch.zeros(B, device=pose_local.device, dtype=pose_local.dtype)

    for foot_name, talo_name in _COLOCATED_ANKLE_PAIRS:
        if foot_name not in joint_names or talo_name not in joint_names:
            continue
        foot_idx = joint_names.index(foot_name)
        talo_idx = joint_names.index(talo_name)
        foot_pt = pt_joint_names.index(foot_name)
        talo_pt = pt_joint_names.index(talo_name)

        R_orient_foot = joint_orients[foot_pt].unsqueeze(0).expand(B, -1, -1)
        R_orient_talo = joint_orients[talo_pt].unsqueeze(0).expand(B, -1, -1)

        R_combined = pose_local[:, foot_idx, :3, :3] @ pose_local[:, talo_idx, :3, :3]
        R_c = R_orient_foot.transpose(-2, -1) @ R_combined
        rz_t = torch.atan2(R_c[:, 1, 0], R_c[:, 1, 1])
        B_mat = euler_xyz_to_matrix(torch.stack([zeros, zeros, rz_t], dim=-1))
        M = R_orient_talo @ B_mat
        A = R_c @ M.transpose(-2, -1)
        sy = (-A[:, 2, 0]).clamp(-1.0, 1.0)
        ry_f = torch.asin(sy)
        rx_f = torch.atan2(A[:, 2, 1], A[:, 2, 2])

        pose_local[:, foot_idx, :3, :3] = R_orient_foot @ euler_xyz_to_matrix(
            torch.stack([rx_f, ry_f, zeros], dim=-1)
        )
        pose_local[:, talo_idx, :3, :3] = R_orient_talo @ euler_xyz_to_matrix(
            torch.stack([zeros, zeros, rz_t], dim=-1)
        )


def _transforms_to_dofs(transforms, data_joint_names, pt_data, root_bind_translation=None):
    """Convert local MHR diagnostic transforms to the parameter-transform DOF vector."""
    pt_joint_names = pt_data["joint_names"]
    joint_orients = pt_data["joint_orients"]
    main_joint_mask = pt_data["main_joint_mask"]
    root_idx_pt = pt_joint_names.index("root")
    main_indices = [j for j, is_main in enumerate(main_joint_mask.tolist()) if is_main]
    non_root_main = [j for j in main_indices if j != root_idx_pt]

    B = transforms.shape[0]
    dofs = torch.zeros(
        B, 3 + len(main_indices) * 3, device=transforms.device, dtype=transforms.dtype
    )

    root_data_idx = data_joint_names.index("root")
    root_trans = transforms[:, root_data_idx, :3, 3].clone()
    if root_bind_translation is not None:
        root_trans = root_trans - root_bind_translation
    dofs[:, :3] = root_trans

    root_euler = matrix_to_euler_xyz(
        joint_orients[root_idx_pt].T.unsqueeze(0) @ transforms[:, root_data_idx, :3, :3]
    )
    dofs[:, 3:6] = root_euler

    for k, pt_idx in enumerate(non_root_main):
        joint_name = pt_joint_names[pt_idx]
        if joint_name not in data_joint_names:
            continue
        data_idx = data_joint_names.index(joint_name)
        euler = matrix_to_euler_xyz(
            joint_orients[pt_idx].T.unsqueeze(0) @ transforms[:, data_idx, :3, :3]
        )
        dofs[:, 6 + 3 * k : 6 + 3 * (k + 1)] = euler

    return dofs


def _disabled_pose_param_ids(device):
    return torch.tensor(_DISABLED_POSE_PARAM_IDS, device=device, dtype=torch.long)


def _apply_pose_param_constraints(
    pose_params,
    *,
    freeze_disabled_pose_params,
    bound_active_spine,
):
    if freeze_disabled_pose_params:
        pose_params[:, _disabled_pose_param_ids(pose_params.device)] = 0.0
    else:
        pose_params[:, _FLEXIBLE_SLICE] = 0.0

    if bound_active_spine:
        for param_idx, (lo, hi) in _ACTIVE_SPINE_PARAM_BOUNDS.items():
            pose_params[:, param_idx].clamp_(lo, hi)


def _pose_local_from_result(result, B, J, root_idx, device, dtype):
    pose_local = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
    pose_local[:, :, :3, :3] = result["rotations"]
    pose_local[:, root_idx, :3, 3] = result["root_translation"]
    return pose_local


def _reconstruct_from_local_pose(pose_local, cache, bind_shape, bone_weights, bone_indices):
    B = pose_local.shape[0]
    W = _build_world_transforms(pose_local, cache)
    D = W @ _bexpand4(cache["W_bind_inv"], B)
    return linear_blend_skinning(_bexpand(bind_shape, B), bone_weights, bone_indices, D)


def _result_from_local_pose(
    pose_local, target, cache, bind_shape, bone_weights, bone_indices, root_idx
):
    vertices = _reconstruct_from_local_pose(
        pose_local, cache, bind_shape, bone_weights, bone_indices
    )
    return PoseInversionResult(
        rotations=pose_local[:, :, :3, :3].clone(),
        root_translation=pose_local[:, root_idx, :3, 3].clone(),
        per_vertex_error=torch.norm(vertices - target, dim=-1),
    )


class MHRPoseInversion:
    """Internal native-MHR pose inversion helper."""

    def __init__(
        self,
        data_root: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        use_warp_for_rotations: bool = True,
        skeleton_transfer_rotation_method: str = "auto",
        refit_rotation_method: str = "auto",
        use_reduced_dof_refit: bool = False,
        reduced_dof_refit_iters: int = _REDUCED_DOF_REFIT_ITERS,
        reduced_dof_refit_lr: float = _REDUCED_DOF_REFIT_LR,
        use_identity_reference_for_skeleton: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_warp_for_rotations = use_warp_for_rotations
        self.skeleton_transfer_rotation_method = skeleton_transfer_rotation_method
        self.refit_rotation_method = refit_rotation_method
        self.use_reduced_dof_refit = use_reduced_dof_refit
        self.reduced_dof_refit_iters = int(reduced_dof_refit_iters)
        self.reduced_dof_refit_lr = float(reduced_dof_refit_lr)
        self.use_identity_reference_for_skeleton = use_identity_reference_for_skeleton

        mhr_root = self.data_root / "MHR"
        rig_path = mhr_root / "MHR_base_rig.npz"
        pt_path = mhr_root / "parameter_transform.npz"
        if not rig_path.is_file():
            raise FileNotFoundError(f"Missing MHR rig asset: {rig_path}")
        if not pt_path.is_file():
            raise FileNotFoundError(f"Missing MHR parameter transform asset: {pt_path}")

        rig_data = np.load(rig_path, allow_pickle=False)
        self.joint_names = [str(x) for x in rig_data["joint_names"]]
        self.joint_parent_ids = torch.from_numpy(rig_data["joint_parent_ids"]).to(self.device)
        self.bind_world = torch.from_numpy(rig_data["bind_pose_world"]).to(
            device=self.device, dtype=dtype
        )
        self.bind_local = torch.from_numpy(rig_data["bind_pose_local"]).to(
            device=self.device, dtype=dtype
        )
        self.bind_shape = torch.from_numpy(rig_data["bind_shape"]).to(
            device=self.device, dtype=dtype
        )
        self.skinning_weights = torch.from_numpy(rig_data["skinning_weights"]).to(
            device=self.device, dtype=dtype
        )

        self.root_joint_idx = self.joint_names.index("root")
        self.root_bind_translation = self.bind_local[self.root_joint_idx, :3, 3]
        self.pt_data = _load_parameter_transform(self.device, dtype, pt_path)
        self.dof_masks = _get_dof_masks(self.pt_data)

        self.skel_transfer = self._build_skeleton_transfer(self.bind_world, self.bind_shape)
        self.cache = self._build_refit_cache(self.bind_world, self.bind_shape)

        max_k = int((self.skinning_weights > 1e-6).sum(dim=1).max().item())
        self.bone_weights, self.bone_indices = _to_sparse_weights(self.skinning_weights, max_k)
        self.model_state_joint_indices = torch.tensor(
            [self.pt_data["joint_names"].index(name) for name in self.joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._mhr_model = None

    def _load_mhr_model(self):
        if self._mhr_model is None:
            self._mhr_model = torch.jit.load(
                self.data_root / "MHR" / "mhr_model_lod1.pt",
                map_location=self.device,
            )
        return self._mhr_model

    def _build_skeleton_transfer(self, bind_world, bind_shape):
        return SkeletonTransfer(
            self.joint_parent_ids,
            bind_world,
            bind_shape,
            self.skinning_weights,
            rotation_method=self.skeleton_transfer_rotation_method,
            root_joint_idx=self.root_joint_idx,
            use_warp_for_rotations=self.use_warp_for_rotations,
        )

    def _build_refit_cache(self, bind_world, bind_shape):
        cache = _precompute_refit_cache(
            self.joint_names,
            self.joint_parent_ids,
            bind_world,
            bind_shape,
            self.skinning_weights,
            bind_world,
            root_idx=self.root_joint_idx,
        )
        cache["root_idx"] = self.root_joint_idx
        cache["body_groups"] = _groups_to_indices(_BODY_PASS_GROUPS, self.joint_names)
        cache["finger_groups"] = _groups_to_indices(_FINGER_PASS_GROUPS, self.joint_names)
        cache["constrained_set"] = set()
        cache["constrained_data"] = None
        cache["body_level_data"] = None
        cache["finger_level_data"] = None
        cache["all_level_data"] = None
        return cache

    def _set_bind_state(self, bind_world, bind_shape):
        self.bind_world = bind_world
        self.bind_local = joint_world_to_local(bind_world, self.joint_parent_ids)
        self.bind_shape = bind_shape
        self.root_bind_translation = self.bind_local[self.root_joint_idx, :3, 3]
        self.skel_transfer = self._build_skeleton_transfer(bind_world, bind_shape)
        self.cache = self._build_refit_cache(bind_world, bind_shape)

    def _bind_state(self):
        return {
            "bind_world": self.bind_world,
            "bind_local": self.bind_local,
            "bind_shape": self.bind_shape,
            "root_bind_translation": self.root_bind_translation,
            "skel_transfer": self.skel_transfer,
            "cache": self.cache,
        }

    def _restore_bind_state(self, state):
        self.bind_world = state["bind_world"]
        self.bind_local = state["bind_local"]
        self.bind_shape = state["bind_shape"]
        self.root_bind_translation = state["root_bind_translation"]
        self.skel_transfer = state["skel_transfer"]
        self.cache = state["cache"]

    def _identity_reference_state(
        self,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        reference_pose_params,
        *,
        apply_correctives,
    ):
        model = self._load_mhr_model()
        model_params = torch.cat([reference_pose_params, scale_params], dim=1)
        with torch.no_grad():
            bind_shape, skel_state = model(
                identity_coeffs,
                model_params,
                face_expr_coeffs,
                apply_correctives,
            )
        bind_world = _mhr_skeleton_state_to_transforms(skel_state)[
            :, self.model_state_joint_indices
        ]
        return bind_world, bind_shape

    def _fit_skeletal_and_project(
        self,
        target,
        *,
        body_iters,
        finger_iters,
        full_iters,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        identity_reference_pose_params,
        apply_correctives,
    ):
        if not self.use_identity_reference_for_skeleton:
            skeletal_result = self.fit_skeletal_transforms(
                target,
                body_iters=body_iters,
                finger_iters=finger_iters,
                full_iters=full_iters,
            )
            init_pose_params, skeletal_pose_local = self._skeletal_to_pose_params(skeletal_result)
            return skeletal_result, init_pose_params, skeletal_pose_local

        if target.shape[0] != 1:
            raise ValueError("Identity-conditioned MHR skeleton fitting expects one frame.")

        bind_world, bind_shape = self._identity_reference_state(
            identity_coeffs,
            scale_params,
            face_expr_coeffs,
            identity_reference_pose_params,
            apply_correctives=apply_correctives,
        )
        saved_state = self._bind_state()
        try:
            self._set_bind_state(bind_world[0], bind_shape[0])
            skeletal_result = self.fit_skeletal_transforms(
                target,
                body_iters=body_iters,
                finger_iters=finger_iters,
                full_iters=full_iters,
            )
            init_pose_params, skeletal_pose_local = self._skeletal_to_pose_params(skeletal_result)
        finally:
            self._restore_bind_state(saved_state)
        return skeletal_result, init_pose_params, skeletal_pose_local

    def _constrain_mhr_local(self, pose_local, group=None):
        if group is None:
            _redistribute_colocated_transforms(pose_local, self.joint_names, self.pt_data)
            _constrain_dof_local(pose_local, self.joint_names, self.pt_data, self.dof_masks)
            return

        group_names = [self.joint_names[int(idx)] for idx in group]
        limited = {name: self.dof_masks[name] for name in group_names if name in self.dof_masks}
        if limited:
            _constrain_dof_local(pose_local, self.joint_names, self.pt_data, limited)

    def _finish_body_or_full_pass(self, pose_local):
        _redistribute_colocated_transforms(pose_local, self.joint_names, self.pt_data)
        _constrain_dof_local(pose_local, self.joint_names, self.pt_data, self.dof_masks)

    def _run_mhr_refit_groups(self, pose_local, target, groups):
        if self.use_reduced_dof_refit and self.reduced_dof_refit_iters > 0:
            joint_cache = self.cache["joint_cache"]
            B = pose_local.shape[0]
            for group in groups:
                W = _build_world_transforms(pose_local, self.cache)
                D = W @ _bexpand4(self.cache["W_bind_inv"], B)
                for j_idx in group:
                    jcache = joint_cache.get(j_idx)
                    if jcache is None:
                        continue
                    joint_name = self.joint_names[int(j_idx)]
                    active_axes = self.dof_masks.get(joint_name)
                    if active_axes is None:
                        _refit_joint(
                            pose_local,
                            target,
                            int(j_idx),
                            W,
                            D,
                            self.cache,
                            jcache,
                            None,
                            self.refit_rotation_method,
                        )
                    else:
                        _refit_reduced_dof_joint(
                            pose_local,
                            target,
                            int(j_idx),
                            W,
                            D,
                            self.cache,
                            jcache,
                            self.pt_data,
                            active_axes,
                            iters=self.reduced_dof_refit_iters,
                            lr=self.reduced_dof_refit_lr,
                        )
                self._constrain_mhr_local(pose_local, group)
            return

        for group in groups:
            _run_refit_passes(
                pose_local,
                target,
                self.cache,
                [group],
                False,
                None,
                self.refit_rotation_method,
            )
            self._constrain_mhr_local(pose_local, group)

    def _skeletal_to_pose_params(self, skeletal_result):
        B = skeletal_result["rotations"].shape[0]
        pose_local = _pose_local_from_result(
            skeletal_result,
            B,
            len(self.joint_names),
            self.root_joint_idx,
            skeletal_result["rotations"].device,
            skeletal_result["rotations"].dtype,
        )
        self._finish_body_or_full_pass(pose_local)
        dofs = _transforms_to_dofs(
            pose_local,
            self.joint_names,
            self.pt_data,
            root_bind_translation=self.root_bind_translation,
        )
        pose_params = dofs @ self.pt_data["P_inv_pose_main"]
        params_204 = torch.zeros(B, 204, dtype=pose_params.dtype, device=pose_params.device)
        params_204[:, self.pt_data["pose_indices"]] = pose_params
        return params_204[:, :136], pose_local

    def pose_params_to_local_transforms(self, pose_params: torch.Tensor) -> torch.Tensor:
        """Convert native MHR pose/model params to diagnostic local transforms."""
        pose_params = torch.as_tensor(pose_params, device=self.device, dtype=self.dtype)
        if pose_params.ndim == 1:
            pose_params = pose_params.unsqueeze(0)
        if pose_params.ndim != 2 or pose_params.shape[1] not in (136, 204):
            raise ValueError("pose_params must have shape (B, 136) or (B, 204).")

        B = pose_params.shape[0]
        params_204 = torch.zeros(B, 204, device=self.device, dtype=self.dtype)
        params_204[:, : pose_params.shape[1]] = pose_params
        dofs = params_204[:, self.pt_data["pose_indices"]] @ self.pt_data["P_pose_main"].T

        pt_joint_names = self.pt_data["joint_names"]
        joint_orients = self.pt_data["joint_orients"]
        main_joint_mask = self.pt_data["main_joint_mask"]
        root_idx_pt = pt_joint_names.index("root")
        main_indices = [j for j, is_main in enumerate(main_joint_mask.tolist()) if is_main]
        non_root_main = [j for j in main_indices if j != root_idx_pt]

        euler_by_pt = {}
        euler_by_pt[root_idx_pt] = dofs[:, 3:6]
        for k, pt_idx in enumerate(non_root_main):
            euler_by_pt[pt_idx] = dofs[:, 6 + 3 * k : 6 + 3 * (k + 1)]

        pose_local = self.bind_local.unsqueeze(0).expand(B, -1, -1, -1).clone()
        pose_local[:, self.root_joint_idx, :3, 3] = dofs[:, :3] + self.root_bind_translation
        for pt_idx, euler in euler_by_pt.items():
            joint_name = pt_joint_names[pt_idx]
            if joint_name not in self.joint_names:
                continue
            data_idx = self.joint_names.index(joint_name)
            pose_local[:, data_idx, :3, :3] = joint_orients[pt_idx].unsqueeze(
                0
            ) @ euler_xyz_to_matrix(euler)
        return pose_local

    def model_skeleton_state_to_local_transforms(self, skel_state: torch.Tensor) -> torch.Tensor:
        """Map MHR TorchScript skeleton state to diagnostic local transforms by name."""
        skel_state = torch.as_tensor(skel_state, device=self.device, dtype=self.dtype)
        if skel_state.ndim == 2:
            skel_state = skel_state.unsqueeze(0)
        if skel_state.ndim != 3 or skel_state.shape[-1] != 8:
            raise ValueError("skel_state must have shape (B, J, 8).")
        world = _mhr_skeleton_state_to_transforms(skel_state)[:, self.model_state_joint_indices]
        return joint_world_to_local(world, self.joint_parent_ids)

    def model_params_to_local_transforms(
        self,
        identity_coeffs: torch.Tensor,
        model_params: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None = None,
        *,
        apply_correctives: bool = False,
    ) -> torch.Tensor:
        model_params = torch.as_tensor(model_params, device=self.device, dtype=self.dtype)
        if model_params.ndim == 1:
            model_params = model_params.unsqueeze(0)
        if model_params.ndim != 2 or model_params.shape[1] != 204:
            raise ValueError("model_params must have shape (B, 204).")

        B = model_params.shape[0]
        identity_coeffs = _as_batched_tensor(
            identity_coeffs, (45,), B, self.device, self.dtype, "identity_coeffs"
        )
        face_expr_coeffs = _as_batched_tensor(
            face_expr_coeffs, (72,), B, self.device, self.dtype, "face_expr_coeffs"
        )
        model = self._load_mhr_model()
        _, skel_state = model(identity_coeffs, model_params, face_expr_coeffs, apply_correctives)
        return self.model_skeleton_state_to_local_transforms(skel_state)

    @torch.no_grad()
    def fit_skeletal_transforms(
        self,
        posed_vertices_mhr: torch.Tensor,
        *,
        body_iters: int = 10,
        finger_iters: int = 2,
        full_iters: int = 1,
    ) -> PoseInversionResult:
        target = posed_vertices_mhr.to(device=self.device, dtype=self.dtype)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.shape[-2:] != self.bind_shape.shape:
            raise ValueError(
                f"Expected MHR vertices with shape (B, {self.bind_shape.shape[0]}, 3), "
                f"got {tuple(target.shape)}."
            )

        pose_world = self.skel_transfer.fit(target)
        pose_local = joint_world_to_local(pose_world, self.joint_parent_ids)
        self._constrain_mhr_local(pose_local, None)

        for _ in range(body_iters):
            self._run_mhr_refit_groups(pose_local, target, self.cache["body_groups"])
            self._finish_body_or_full_pass(pose_local)
            _update_root_translation(pose_local, target, self.cache)

        for _ in range(finger_iters):
            self._run_mhr_refit_groups(pose_local, target, self.cache["finger_groups"])

        all_groups = self.cache["body_groups"] + self.cache["finger_groups"]
        for _ in range(full_iters):
            self._run_mhr_refit_groups(pose_local, target, all_groups)
            self._finish_body_or_full_pass(pose_local)
            _update_root_translation(pose_local, target, self.cache)

        return _result_from_local_pose(
            pose_local,
            target,
            self.cache,
            self.bind_shape,
            self.bone_weights,
            self.bone_indices,
            self.root_joint_idx,
        )

    def _refine_pose_params(
        self,
        init_pose_params,
        target,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        *,
        refine_iters,
        lr,
        optimize_flexibles,
        freeze_disabled_pose_params,
        bound_active_spine,
        apply_correctives,
    ):
        if refine_iters <= 0:
            return init_pose_params, [], torch.zeros(0, device=target.device, dtype=target.dtype)

        model = self._load_mhr_model()
        pose_opt = init_pose_params.detach().clone().requires_grad_(True)
        fixed_flex = init_pose_params[:, _FLEXIBLE_SLICE].detach().clone()
        disabled_ids = _disabled_pose_param_ids(target.device)
        fixed_disabled = init_pose_params[:, disabled_ids].detach().clone()
        optimizer = torch.optim.Adam([pose_opt], lr=lr)
        loss_history = []
        best_loss = float("inf")
        best_pose = init_pose_params.detach().clone()

        for _ in range(refine_iters):
            optimizer.zero_grad()
            pose_for_model = pose_opt
            if not optimize_flexibles:
                pose_for_model = pose_opt.clone()
                if freeze_disabled_pose_params:
                    pose_for_model[:, disabled_ids] = fixed_disabled
                else:
                    pose_for_model[:, _FLEXIBLE_SLICE] = fixed_flex
            if bound_active_spine:
                pose_for_model = pose_for_model.clone()
                for param_idx, (lo, hi) in _ACTIVE_SPINE_PARAM_BOUNDS.items():
                    pose_for_model[:, param_idx].clamp_(lo, hi)

            pred, _ = model(
                identity_coeffs,
                torch.cat([pose_for_model, scale_params], dim=1),
                face_expr_coeffs,
                apply_correctives,
            )
            loss = torch.nn.functional.mse_loss(pred, target)
            cur_loss = float(loss.detach().cpu())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_pose = pose_for_model.detach().clone()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if not optimize_flexibles:
                    if freeze_disabled_pose_params:
                        pose_opt[:, disabled_ids] = fixed_disabled
                    else:
                        pose_opt[:, _FLEXIBLE_SLICE] = fixed_flex
                if bound_active_spine:
                    for param_idx, (lo, hi) in _ACTIVE_SPINE_PARAM_BOUNDS.items():
                        pose_opt[:, param_idx].clamp_(lo, hi)
            loss_history.append(cur_loss)

        with torch.no_grad():
            pose_final = best_pose.detach().clone()
            if not optimize_flexibles:
                if freeze_disabled_pose_params:
                    pose_final[:, disabled_ids] = fixed_disabled
                else:
                    pose_final[:, _FLEXIBLE_SLICE] = fixed_flex
            if bound_active_spine:
                for param_idx, (lo, hi) in _ACTIVE_SPINE_PARAM_BOUNDS.items():
                    pose_final[:, param_idx].clamp_(lo, hi)
            pred, _ = model(
                identity_coeffs,
                torch.cat([pose_final, scale_params], dim=1),
                face_expr_coeffs,
                apply_correctives,
            )
            per_vertex_error = torch.norm(pred - target, dim=-1)

        return pose_final, loss_history, per_vertex_error

    def fit(
        self,
        posed_vertices_mhr: torch.Tensor,
        *,
        identity_coeffs: torch.Tensor | None = None,
        scale_params: torch.Tensor | None = None,
        face_expr_coeffs: torch.Tensor | None = None,
        identity_reference_pose_params: torch.Tensor | None = None,
        body_iters: int = 10,
        finger_iters: int = 2,
        full_iters: int = 1,
        refine_iters: int = 0,
        lr: float = 1e-4,
        optimize_flexibles: bool = False,
        freeze_disabled_pose_params: bool = False,
        bound_active_spine: bool = False,
        apply_correctives: bool = False,
        batch_size: int | None = None,
    ) -> MHRPoseInversionResult:
        target = posed_vertices_mhr.to(device=self.device, dtype=self.dtype)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        B = target.shape[0]

        if batch_size is not None and B > batch_size:
            chunks = []
            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                chunk_kwargs = {}
                for key, value in (
                    ("identity_coeffs", identity_coeffs),
                    ("scale_params", scale_params),
                    ("face_expr_coeffs", face_expr_coeffs),
                    ("identity_reference_pose_params", identity_reference_pose_params),
                ):
                    if value is not None:
                        value_t = torch.as_tensor(value)
                        chunk_kwargs[key] = value_t[start:end] if value_t.ndim > 1 else value
                chunks.append(
                    self.fit(
                        target[start:end],
                        body_iters=body_iters,
                        finger_iters=finger_iters,
                        full_iters=full_iters,
                        refine_iters=refine_iters,
                        lr=lr,
                        optimize_flexibles=optimize_flexibles,
                        freeze_disabled_pose_params=freeze_disabled_pose_params,
                        bound_active_spine=bound_active_spine,
                        apply_correctives=apply_correctives,
                        batch_size=None,
                        **chunk_kwargs,
                    )
                )
            return MHRPoseInversionResult(
                {
                    "pose_params": torch.cat([c["pose_params"] for c in chunks], dim=0),
                    "model_params": torch.cat([c["model_params"] for c in chunks], dim=0),
                    "init_pose_params": torch.cat([c["init_pose_params"] for c in chunks], dim=0),
                    "per_vertex_error": torch.cat([c["per_vertex_error"] for c in chunks], dim=0),
                    "pre_refine_per_vertex_error": torch.cat(
                        [c["pre_refine_per_vertex_error"] for c in chunks], dim=0
                    ),
                    "skeletal_per_vertex_error": torch.cat(
                        [c["skeletal_per_vertex_error"] for c in chunks], dim=0
                    ),
                    "pose_local": torch.cat([c["pose_local"] for c in chunks], dim=0),
                    "skeletal_pose_local": torch.cat(
                        [c["skeletal_pose_local"] for c in chunks], dim=0
                    ),
                    "loss_history": [c["loss_history"] for c in chunks],
                    "iters_run": sum(c["iters_run"] for c in chunks),
                }
            )

        identity_coeffs = _as_batched_tensor(
            identity_coeffs, (45,), B, self.device, self.dtype, "identity_coeffs"
        )
        scale_params = _as_batched_tensor(
            scale_params, (68,), B, self.device, self.dtype, "scale_params"
        )
        face_expr_coeffs = _as_batched_tensor(
            face_expr_coeffs, (72,), B, self.device, self.dtype, "face_expr_coeffs"
        )
        identity_reference_pose_params = _as_batched_tensor(
            identity_reference_pose_params,
            (136,),
            B,
            self.device,
            self.dtype,
            "identity_reference_pose_params",
        )

        if self.use_identity_reference_for_skeleton and B > 1:
            chunks = []
            for frame_idx in range(B):
                chunks.append(
                    self.fit(
                        target[frame_idx : frame_idx + 1],
                        identity_coeffs=identity_coeffs[frame_idx : frame_idx + 1],
                        scale_params=scale_params[frame_idx : frame_idx + 1],
                        face_expr_coeffs=face_expr_coeffs[frame_idx : frame_idx + 1],
                        identity_reference_pose_params=identity_reference_pose_params[
                            frame_idx : frame_idx + 1
                        ],
                        body_iters=body_iters,
                        finger_iters=finger_iters,
                        full_iters=full_iters,
                        refine_iters=refine_iters,
                        lr=lr,
                        optimize_flexibles=optimize_flexibles,
                        freeze_disabled_pose_params=freeze_disabled_pose_params,
                        bound_active_spine=bound_active_spine,
                        apply_correctives=apply_correctives,
                        batch_size=None,
                    )
                )
            return MHRPoseInversionResult(
                {
                    "pose_params": torch.cat([c["pose_params"] for c in chunks], dim=0),
                    "model_params": torch.cat([c["model_params"] for c in chunks], dim=0),
                    "init_pose_params": torch.cat([c["init_pose_params"] for c in chunks], dim=0),
                    "per_vertex_error": torch.cat([c["per_vertex_error"] for c in chunks], dim=0),
                    "pre_refine_per_vertex_error": torch.cat(
                        [c["pre_refine_per_vertex_error"] for c in chunks], dim=0
                    ),
                    "skeletal_per_vertex_error": torch.cat(
                        [c["skeletal_per_vertex_error"] for c in chunks], dim=0
                    ),
                    "pose_local": torch.cat([c["pose_local"] for c in chunks], dim=0),
                    "skeletal_pose_local": torch.cat(
                        [c["skeletal_pose_local"] for c in chunks], dim=0
                    ),
                    "loss_history": [c["loss_history"] for c in chunks],
                    "iters_run": sum(c["iters_run"] for c in chunks),
                }
            )

        skeletal_result, init_pose_params, skeletal_pose_local = self._fit_skeletal_and_project(
            target,
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            identity_coeffs=identity_coeffs,
            scale_params=scale_params,
            face_expr_coeffs=face_expr_coeffs,
            identity_reference_pose_params=identity_reference_pose_params,
            apply_correctives=apply_correctives,
        )
        init_pose_params = init_pose_params.clone()
        if not optimize_flexibles:
            _apply_pose_param_constraints(
                init_pose_params,
                freeze_disabled_pose_params=freeze_disabled_pose_params,
                bound_active_spine=bound_active_spine,
            )

        model = self._load_mhr_model()
        with torch.no_grad():
            pre_vertices, _ = model(
                identity_coeffs,
                torch.cat([init_pose_params, scale_params], dim=1),
                face_expr_coeffs,
                apply_correctives,
            )
            pre_refine_error = torch.norm(pre_vertices - target, dim=-1)

        pose_params, loss_history, refined_error = self._refine_pose_params(
            init_pose_params,
            target,
            identity_coeffs,
            scale_params,
            face_expr_coeffs,
            refine_iters=refine_iters,
            lr=lr,
            optimize_flexibles=optimize_flexibles,
            freeze_disabled_pose_params=freeze_disabled_pose_params,
            bound_active_spine=bound_active_spine,
            apply_correctives=apply_correctives,
        )
        model_params = torch.cat([pose_params, scale_params], dim=1)
        pose_local = self.model_params_to_local_transforms(
            identity_coeffs,
            model_params,
            face_expr_coeffs,
            apply_correctives=apply_correctives,
        )

        return MHRPoseInversionResult(
            pose_params=pose_params,
            model_params=model_params,
            init_pose_params=init_pose_params,
            per_vertex_error=refined_error if refine_iters > 0 else pre_refine_error,
            pre_refine_per_vertex_error=pre_refine_error,
            skeletal_per_vertex_error=skeletal_result["per_vertex_error"],
            pose_local=pose_local,
            skeletal_pose_local=skeletal_pose_local,
            skeletal_result=skeletal_result,
            loss_history=loss_history,
            iters_run=refine_iters,
        )
