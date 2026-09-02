# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rig a custom SOMA body template mesh and export as a UsdSkel USD file.

The input mesh MUST share the exact topology of the SOMA body template mesh
(``c_skin_mid``): 18 056 vertices with the same vertex ordering and
connectivity as ``assets/SOMA_template_rig.usda``.  The skinning weights are
transferred directly from the template rig, so any mismatch in vertex count
or ordering will produce incorrect deformation.

Supported input formats: OBJ, USD/USDA/USDC.

Usage:
    python -m tools.rig_soma_body_mesh --input my_body.obj --output my_body_rig.usda
    python -m tools.rig_soma_body_mesh --input my_body.usda --output my_body_rig.usda
    python -m tools.rig_soma_body_mesh --input my_body.obj --output my_body_rig.usda --unit meters
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.body import SOMALayer  # noqa: E402
from soma.geometry.rig_utils import joint_world_to_local  # noqa: E402
from soma.io import save_soma_usd  # noqa: E402
from soma.units import Unit  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

# Must match the vertex count of the SOMA body template mesh (c_skin_mid).
_SOMA_BODY_VERTEX_COUNT = 18_056
logger = logging.getLogger(__name__)


def load_mesh_vertices(path):
    """Load vertex positions from an OBJ or USD/USDA/USDC file.

    Args:
        path: Path to the mesh file.

    Returns:
        (V, 3) float32 numpy array of vertex positions.

    Raises:
        ValueError: If the format is unsupported or multiple meshes are found
            with ambiguous vertex counts.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".obj":
        import trimesh

        mesh = trimesh.load(str(path), maintain_order=True, process=False)
        return np.asarray(mesh.vertices, dtype=np.float32)

    if suffix in (".usd", ".usda", ".usdc"):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
        if not mesh_prims:
            raise ValueError(f"No Mesh prim found in '{path}'")

        # If there is only one mesh, use it directly.  Otherwise, look for the
        # one whose vertex count matches the SOMA body template.
        if len(mesh_prims) == 1:
            prim = mesh_prims[0]
        else:
            candidates = [
                p
                for p in mesh_prims
                if len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) == _SOMA_BODY_VERTEX_COUNT
            ]
            if len(candidates) == 1:
                prim = candidates[0]
            elif len(candidates) == 0:
                counts = sorted(
                    set(len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) for p in mesh_prims)
                )
                raise ValueError(
                    f"No mesh with {_SOMA_BODY_VERTEX_COUNT} vertices found in '{path}'. "
                    f"Vertex counts present: {counts}"
                )
            else:
                raise ValueError(
                    f"Multiple meshes with {_SOMA_BODY_VERTEX_COUNT} vertices found in '{path}'. "
                    "Please provide a file with a single body mesh."
                )

        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return np.array(pts, dtype=np.float32)

    raise ValueError(f"Unsupported mesh format '{suffix}'. Expected .obj, .usd, .usda, or .usdc.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rig a custom SOMA body template mesh and export as UsdSkel. "
            f"Input mesh must have exactly {_SOMA_BODY_VERTEX_COUNT} vertices "
            "matching the SOMA body template topology (c_skin_mid)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input mesh file (.obj, .usd, .usda, .usdc). Must match SOMA body template topology.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output UsdSkel file (.usd, .usda, .usdc).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to SOMA assets directory (default: <repo>/assets).",
    )
    parser.add_argument(
        "--unit",
        choices=[u.unit_name for u in Unit],
        default=Unit.CENTIMETERS.unit_name,
        help="Unit of the input mesh coordinates (default: centimeters, matching SOMA template rig).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for skeleton transfer (default: cpu).",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    input_unit = Unit.from_name(args.unit)

    # --- Load input mesh ---
    logger.info(f"Loading mesh: {args.input} ...")
    verts = load_mesh_vertices(args.input)
    logger.info(f"  Vertices: {len(verts)}")

    if len(verts) != _SOMA_BODY_VERTEX_COUNT:
        raise ValueError(
            f"Input mesh has {len(verts)} vertices, expected {_SOMA_BODY_VERTEX_COUNT}. "
            "The mesh must match the SOMA body template topology (c_skin_mid) exactly."
        )

    # --- Initialize SOMA ---
    logger.info("\nInitializing SOMA...")
    soma = SOMALayer(
        data_root,
        identity_model_type="mhr",
        device=args.device,
        mode="pytorch",
        output_unit=input_unit,
    )

    # --- Fit skeleton to the input shape ---
    logger.info("Fitting skeleton...")
    verts_t = torch.from_numpy(verts).to(args.device)
    with torch.no_grad():
        bind_world = soma.skeleton_transfer.fit(verts_t)  # (J, 4, 4)

    bind_local = joint_world_to_local(bind_world, soma.joint_parent_ids)  # (J, 4, 4)

    # --- Export ---
    logger.info(f"\nExporting rig: {args.output}")
    save_soma_usd(
        args.output,
        joint_names=list(soma.rig_data["joint_names"]),
        joint_parent_ids=soma.joint_parent_ids.cpu().numpy(),
        bind_transforms_world=bind_world.cpu().numpy(),
        bind_transforms_local=bind_local.cpu().numpy(),
        rest_shape=verts,
        faces=soma.faces.cpu().numpy(),
        face_vert_indices=soma.rig_data.get("face_vert_indices"),
        face_vert_counts=soma.rig_data.get("face_vert_counts"),
        uv_data=soma.rig_data.get("uv_data"),
        skinning_weights=soma.skinning_weights.cpu().numpy(),
        unit=args.unit,
    )


if __name__ == "__main__":
    main()
