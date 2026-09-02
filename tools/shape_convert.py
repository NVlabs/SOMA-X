# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MHR shape to SMPL shape converter.

This class handles the conversion of MHR shape parameters to SMPL shape parameters.
"""

import argparse
import logging
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from soma.body import SOMALayer
from soma.geometry.barycentric_interp import BarycentricInterpolator
from soma.geometry.batched_skinning import BatchedSkinning
from tools.logging_utils import add_logging_args, configure_logging
from tools.vis_pyrender import MeshRenderer, look_at, set_pyopengl_platform

logger = logging.getLogger(__name__)


def get_smooth_noise(T, dim, device, num_keyframes=None, mode="normal"):
    if num_keyframes is None:
        num_keyframes = max(3, T // 30)

    if mode == "normal":
        keyframes = torch.randn(1, dim, num_keyframes, device=device)
    elif mode == "uniform":
        keyframes = torch.rand(1, dim, num_keyframes, device=device)

    res = F.interpolate(keyframes, size=T, mode="linear", align_corners=True)[0].T
    return res


class ShapeTransfer(nn.Module):
    """
    MHR shape to SMPL shape converter.
    """

    def __init__(self, data_root, device="cuda"):
        """
        Initialize the shape transfer.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.device = device

        self.mhr_soma = SOMALayer(
            data_root=data_root, device=device, identity_model_type="mhr", mode="warp"
        )
        self.smpl_soma = SOMALayer(
            data_root=data_root, device=device, identity_model_type="smpl", mode="warp"
        )
        self.soma_to_smpl = self.get_soma_to_smpl_interpolator()

        smpl_rest_mesh = trimesh.load(
            self.data_root / "SMPL" / "base_body.obj", maintain_order=True, process=False
        )
        smpl_rest_shape = torch.from_numpy(smpl_rest_mesh.vertices).float().to(device).unsqueeze(0)
        self.smpl_rest_shape_soma = self.smpl_soma.identity_model.identity_model_to_soma(
            smpl_rest_shape
        )
        self.posed_world_smpl_tpose = self.mhr_soma.skeleton_transfer.fit(self.smpl_rest_shape_soma)

    def get_soma_to_smpl_interpolator(self):
        mesh_smpl = trimesh.load(
            self.data_root / "SMPL" / "base_body.obj", maintain_order=True, process=False
        )
        V_smpl = torch.from_numpy(mesh_smpl.vertices).float().to(self.device)
        mesh_soma = trimesh.load(
            self.data_root / "SMPL" / "SOMA_wrap.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(self.device)
        F_soma = torch.from_numpy(mesh_soma.faces).int().to(self.device)
        return BarycentricInterpolator(V_soma, F_soma, V_smpl)

    def forward(self, identity_coeffs, scale_params):
        """
        Forward pass.
        """
        batch_size = identity_coeffs.shape[0]
        device = identity_coeffs.device

        # 1. Get MHR rest shape
        mhr_rest_shape_soma = self.mhr_soma.identity_model(identity_coeffs, scale_params)

        # 2. Get MHR posed world transforms
        posed_world_mhr = self.mhr_soma.skeleton_transfer.fit(mhr_rest_shape_soma)

        # 3. Skin the MHR rest shape to get it under the SMPL rest pose
        batched_skinning = BatchedSkinning(
            self.mhr_soma.joint_parent_ids,
            self.mhr_soma.skinning_weights,
            posed_world_mhr,
            mhr_rest_shape_soma,
            joint_orient=self.posed_world_smpl_tpose[0],
            mode=self.mhr_soma.mode,
        )

        pose_rotations = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 78, 3, 3)
        pose_translations = self.posed_world_smpl_tpose[:, 1, :3, 3].expand(batch_size, 3)

        vertices, T_world = batched_skinning.pose(
            local_rotations=pose_rotations,
            global_translation=pose_translations,
            return_transforms=True,
        )

        # 4. Get the SMPL topology vertices
        mhr_vertices_smpl = self.soma_to_smpl(vertices)

        # 5. Solve the betas
        B = (
            mhr_vertices_smpl - self.smpl_soma.identity_model.identity_model.v_template[None]
        ).reshape(batch_size, -1)
        A = self.smpl_soma.identity_model.identity_model.shape_dirs.reshape(-1, 10)

        betas = torch.linalg.lstsq(A, B.T).solution.T

        return betas


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape transfer.")
    parser.add_argument("--data_root", type=str, default="./assets", help="Path to the data root.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="out/")
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--pyopengl-platform", default="osmesa")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    set_pyopengl_platform(args.pyopengl_platform)

    device = "cuda"
    shape_transfer = ShapeTransfer(args.data_root, device)
    T = args.sequence_length

    mhr_im = shape_transfer.mhr_soma.identity_model
    identity_coeffs = get_smooth_noise(T, mhr_im.num_identity_coeffs, device)
    scale_params = get_smooth_noise(T, mhr_im.num_scale_params, device, mode="normal") * 0.2
    zero_pose = torch.zeros(1, 77, 3, device=device)
    zero_transl = torch.zeros(1, 3, device=device)

    betas = shape_transfer(identity_coeffs, scale_params)

    smpl_vertices = shape_transfer.smpl_soma(zero_pose, betas, None, zero_transl)["vertices"]
    mhr_vertices = shape_transfer.mhr_soma(zero_pose, identity_coeffs, scale_params, zero_transl)[
        "vertices"
    ]

    smpl_vertices = smpl_vertices.detach().cpu().numpy()
    mhr_vertices = mhr_vertices.detach().cpu().numpy()
    faces = shape_transfer.mhr_soma.faces.cpu().numpy()

    logger.info("Rendering videos...")
    colors = {
        "mhr": (0.98, 0.65, 0.15, 1.0),
        "anny": (0.25, 0.75, 1.0, 1.0),
        "smpl": (0.55, 0.15, 0.85, 1.0),
    }

    def save_video(frames, path, fps=30):
        imageio.mimsave(path, frames, fps=fps)
        logger.info(f"Saved {path}")

    renderer = MeshRenderer(image_size=args.image_size, light_intensity=5)

    cam_pose = look_at(
        eye=np.array([0.0, 0.0, 6.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light_dir = np.array([0.0, -0.5, -1.0])

    faces = shape_transfer.mhr_soma.faces.cpu().numpy()
    frames = []

    for t in tqdm(range(T)):
        mhr_img = renderer.render(
            mhr_vertices[t],
            faces,
            mesh_color=colors["mhr"],
            cam_pose=cam_pose,
            light_dir=light_dir,
            metallic=0.0,
            roughness=0.5,
            base_color_factor=[0.9, 0.9, 0.9, 1.0],
        )
        smpl_img = renderer.render(
            smpl_vertices[t],
            faces,
            mesh_color=colors["smpl"],
            cam_pose=cam_pose,
            light_dir=light_dir,
            metallic=0.0,
            roughness=0.5,
            base_color_factor=[0.9, 0.9, 0.9, 1.0],
        )
        merged_img = (0.5 * mhr_img + 0.5 * smpl_img).astype(np.uint8)
        img = np.concatenate([mhr_img, merged_img, smpl_img], axis=1)
        frames.append(img[..., ::-1])

    renderer.delete()
    save_video(frames, Path(args.output_dir) / "shape_transfer.mp4")
