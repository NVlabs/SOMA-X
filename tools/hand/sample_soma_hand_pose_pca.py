#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample a SOMAHand pose PCA prior and render the posed skinned mesh.

Example::

    .venv/Scripts/python.exe tools/hand/sample_soma_hand_pose_pca.py \
        --hand-asset assets/SOMAHand.npz \
        --output-prefix out/somahand_pose_pca_samples \
        --hand-type left --num-poses 60 --fps 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma import SOMAHandLayer  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402
from tools.vis_pyrender import (  # noqa: E402
    MeshRenderer,
    default_pyopengl_platform,
    look_at,
    set_pyopengl_platform,
)

logger = logging.getLogger(__name__)


def load_pose_prior(path: Path, hand_type: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate one side of a bind-relative SOMAHand pose prior."""
    prefix = f"{hand_type}_pose_"
    keys = (
        "mean",
        "components",
        "eigenvalues",
        "joint_names",
        "reference_joint_names",
        "reference_parent_ids",
        "reference_orient",
        "reference_orient_parent",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [prefix + key for key in keys if prefix + key not in data]
        if missing:
            raise ValueError(f"Prior is missing required arrays: {', '.join(missing)}")
        if "pose_pca_metadata" not in data:
            raise ValueError("SOMAHand asset is missing pose_pca_metadata.")
        arrays = {key: np.asarray(data[prefix + key]) for key in keys}
        metadata = json.loads(data["pose_pca_metadata"].item())

    if metadata.get("rotation_repr") != "expmap":
        raise ValueError("Prior rotation_repr must be 'expmap'.")
    if not metadata.get("identity_at_bind_pose", False):
        raise ValueError("Prior must use the bind pose as the expmap identity.")
    if arrays["mean"].shape != (24, 3):
        raise ValueError(f"Prior mean must have shape (24, 3), got {arrays['mean'].shape}.")
    if arrays["components"].ndim != 2 or arrays["components"].shape[1] != 72:
        raise ValueError(
            f"Prior components must have shape (K, 72), got {arrays['components'].shape}."
        )
    if arrays["eigenvalues"].shape != (len(arrays["components"]),):
        raise ValueError("Prior eigenvalues do not match the component count.")
    if arrays["reference_orient"].shape != (25, 3, 3):
        raise ValueError("Prior reference_orient must have shape (25, 3, 3).")
    if arrays["reference_orient_parent"].shape != (25, 3, 3):
        raise ValueError("Prior reference_orient_parent must have shape (25, 3, 3).")
    return arrays, metadata


def sample_pose_prior(
    prior: dict[str, np.ndarray],
    *,
    num_poses: int,
    n_components: int,
    seed: int,
    sample_scale: float,
) -> np.ndarray:
    """Draw Gaussian PCA samples as ``(N, 24, 3)`` bind-relative expmaps."""
    available = len(prior["components"])
    if num_poses < 1:
        raise ValueError("num_poses must be at least 1.")
    if not 1 <= n_components <= available:
        raise ValueError(f"n_components must be in [1, {available}], got {n_components}.")
    if sample_scale <= 0.0:
        raise ValueError("sample_scale must be positive.")

    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((num_poses, n_components)) * sample_scale
    components = prior["components"][:n_components].astype(np.float64)
    standard_deviations = np.sqrt(prior["eigenvalues"][:n_components].astype(np.float64))
    mean = prior["mean"].astype(np.float64).reshape(-1)
    samples = mean + (coefficients * standard_deviations) @ components
    return samples.reshape(num_poses, 24, 3)


def bind_relative_to_absolute_local(
    expmaps: np.ndarray,
    reference_orient: np.ndarray,
    reference_orient_parent: np.ndarray,
) -> np.ndarray:
    """Decode wrist-free bind-relative expmaps to 25 absolute local rotations."""
    expmaps = np.asarray(expmaps, dtype=np.float64)
    if expmaps.ndim != 3 or expmaps.shape[1:] != (24, 3):
        raise ValueError(f"expmaps must have shape (N, 24, 3), got {expmaps.shape}.")
    relative = np.tile(np.eye(3, dtype=np.float64), (len(expmaps), 25, 1, 1))
    relative[:, 1:] = (
        Rotation.from_rotvec(expmaps.reshape(-1, 3)).as_matrix().reshape(len(expmaps), 24, 3, 3)
    )
    return reference_orient_parent.swapaxes(-2, -1) @ relative @ reference_orient


def pose_vertices(
    absolute_local: np.ndarray,
    prior: dict[str, np.ndarray],
    *,
    data_root: Path,
    hand_type: str,
    device: torch.device,
    lod: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Skin a neutral SOMAHand identity with absolute local rotations."""
    layer = SOMAHandLayer(
        data_root=data_root,
        hand_type=hand_type,
        device=device,
        mode="torch",
        lod=lod,
        correctives_model_path=None,
    ).to(device)
    runtime_names = np.asarray(layer.rig_data["joint_names"])
    if not np.array_equal(prior["reference_joint_names"], runtime_names):
        raise ValueError("Prior reference joints do not match the SOMAHand runtime ordering.")
    if not np.array_equal(prior["reference_parent_ids"], layer.joint_parent_ids.cpu().numpy()):
        raise ValueError("Prior parent IDs do not match the SOMAHand runtime topology.")
    if not np.allclose(
        prior["reference_orient"],
        layer.bind_pose_world.cpu().numpy()[:, :3, :3],
        atol=1e-5,
    ):
        raise ValueError("Prior bind-reference orientations do not match SOMAHand assets.")

    identity = torch.zeros(1, layer.num_shape_components, device=device)
    rotations = torch.from_numpy(absolute_local).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        layer.prepare_identity(identity)
        output = layer.pose(
            rotations,
            pose2rot=False,
            absolute_pose=True,
        )
    return output["vertices"].cpu().numpy(), layer.faces.cpu().numpy()


def render_media(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    output_prefix: Path,
    fps: float,
    image_size: int,
) -> tuple[Path, Path]:
    """Render one mesh per frame and write matching MP4 and GIF files."""
    import imageio.v2 as imageio

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    flat_vertices = vertices.reshape(-1, 3)
    bbox_min = flat_vertices.min(axis=0)
    bbox_max = flat_vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_extent = float((bbox_max - bbox_min).max())
    if not np.isfinite(bbox_extent) or bbox_extent <= 0.0:
        raise ValueError("Cannot render a degenerate or non-finite vertex sequence.")
    camera_distance = bbox_extent * 1.8
    camera_pose = look_at(
        eye=bbox_center + np.array([0.0, 0.0, camera_distance]),
        target=bbox_center,
        up=np.array([0.0, 1.0, 0.0]),
    )
    focal_length = image_size * camera_distance / (bbox_extent * 1.15)
    renderer = MeshRenderer(
        image_size=image_size,
        light_intensity=5.0,
        focal_length=focal_length,
    )
    renderer.camera.zfar = 500.0
    renderer.setup_mesh(
        faces=faces,
        mesh_color=(0.46, 0.72, 0.16, 1.0),
        cam_pose=camera_pose,
        light_dir=np.array([0.0, -0.3, -1.0]),
        metallic=0.0,
        roughness=0.5,
        base_color_factor=(0.9, 0.9, 0.9, 1.0),
    )

    mp4_path = output_prefix.with_suffix(".mp4")
    gif_path = output_prefix.with_suffix(".gif")
    writer = imageio.get_writer(mp4_path, fps=fps)
    gif_frames = []
    try:
        for frame_vertices in tqdm(vertices, desc="Rendering PCA samples"):
            frame = renderer.render_frame(frame_vertices)
            writer.append_data(frame[..., ::-1])
            gif_frames.append(frame)
    finally:
        writer.close()
        renderer.delete()
    imageio.mimsave(gif_path, gif_frames, duration=1000.0 / fps, loop=0)
    return mp4_path, gif_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-asset", type=Path, default=Path("assets/SOMAHand.npz"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("out/somahand_pose_pca_samples"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("assets"))
    parser.add_argument("--hand-type", choices=("left", "right"), default="left")
    parser.add_argument("--num-poses", type=int, default=60)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--lod", choices=("mid", "low", "xlo"), default="mid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pyopengl-platform", default=default_pyopengl_platform())
    add_logging_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args)
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.image_size < 64:
        parser.error("--image-size must be at least 64")

    set_pyopengl_platform(args.pyopengl_platform)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prior, metadata = load_pose_prior(args.hand_asset, args.hand_type)
    samples = sample_pose_prior(
        prior,
        num_poses=args.num_poses,
        n_components=args.n_components,
        seed=args.seed,
        sample_scale=args.sample_scale,
    )
    absolute_local = bind_relative_to_absolute_local(
        samples,
        prior["reference_orient"],
        prior["reference_orient_parent"],
    )
    vertices, faces = pose_vertices(
        absolute_local,
        prior,
        data_root=args.data_root,
        hand_type=args.hand_type,
        device=device,
        lod=args.lod,
    )
    mp4_path, gif_path = render_media(
        vertices,
        faces,
        output_prefix=args.output_prefix,
        fps=args.fps,
        image_size=args.image_size,
    )
    logger.info(
        "Rendered %d %s-hand samples from schema %s at %.3g FPS",
        len(samples),
        args.hand_type,
        metadata.get("schema_version", "unknown"),
        args.fps,
    )
    logger.info("MP4: %s", mp4_path)
    logger.info("GIF: %s", gif_path)


if __name__ == "__main__":
    main()
