# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MANO rig layer and joint conventions for SOMA-X hand workflows."""

from pathlib import Path

import torch

from ..smpl import _SMPLFamilyLBSLayer
from ..units import Unit
from ._smpl_family_loader import load_mano_pkl, mano_parent_ids

MANO_JOINT_NAMES = [
    "Wrist",
    "Index1",
    "Index2",
    "Index3",
    "Middle1",
    "Middle2",
    "Middle3",
    "Pinky1",
    "Pinky2",
    "Pinky3",
    "Ring1",
    "Ring2",
    "Ring3",
    "Thumb1",
    "Thumb2",
    "Thumb3",
]

# MANO pkl assets expose a kinematic tree for the 16 joints above.  Many
# datasets and visualizers use a 21-joint landmark convention that inserts one
# fingertip landmark at the end of each MANO finger chain.  Those fingertips are
# mesh vertices, not pkl kinematic joints, so keep the derived convention here
# instead of duplicating numeric parent lists in tools.
MANO_FINGERTIP_VERTEX_IDS = {
    "IndexTip": 320,
    "MiddleTip": 443,
    "PinkyTip": 671,
    "RingTip": 554,
    "ThumbTip": 744,
}

MANO_JOINT_NAMES_WITH_FINGERTIPS = [
    "Wrist",
    "Index1",
    "Index2",
    "Index3",
    "IndexTip",
    "Middle1",
    "Middle2",
    "Middle3",
    "MiddleTip",
    "Pinky1",
    "Pinky2",
    "Pinky3",
    "PinkyTip",
    "Ring1",
    "Ring2",
    "Ring3",
    "RingTip",
    "Thumb1",
    "Thumb2",
    "Thumb3",
    "ThumbTip",
]

# Root uses a self-parent (0), matching the rest of the local skeleton utilities.
MANO_JOINT_PARENT_NAMES_WITH_FINGERTIPS = [
    "Wrist",
    "Wrist",
    "Index1",
    "Index2",
    "Index3",
    "Wrist",
    "Middle1",
    "Middle2",
    "Middle3",
    "Wrist",
    "Pinky1",
    "Pinky2",
    "Pinky3",
    "Wrist",
    "Ring1",
    "Ring2",
    "Ring3",
    "Wrist",
    "Thumb1",
    "Thumb2",
    "Thumb3",
]


def _joint_parent_ids_from_names(joint_names: list[str], parent_names: list[str]) -> list[int]:
    if len(joint_names) != len(parent_names):
        raise ValueError(
            f"Expected one parent per joint, got {len(joint_names)} joints and "
            f"{len(parent_names)} parents."
        )
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return [name_to_idx[parent_name] for parent_name in parent_names]


MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS = _joint_parent_ids_from_names(
    MANO_JOINT_NAMES_WITH_FINGERTIPS,
    MANO_JOINT_PARENT_NAMES_WITH_FINGERTIPS,
)

# Backward-compatible aliases for the common "21-joint MANO" wording.
MANO_JOINT_NAMES_21 = MANO_JOINT_NAMES_WITH_FINGERTIPS
MANO_PARENT_IDS_21 = MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS
MANO_TIP_VERTEX_IDS = {
    "index": MANO_FINGERTIP_VERTEX_IDS["IndexTip"],
    "middle": MANO_FINGERTIP_VERTEX_IDS["MiddleTip"],
    "pinky": MANO_FINGERTIP_VERTEX_IDS["PinkyTip"],
    "ring": MANO_FINGERTIP_VERTEX_IDS["RingTip"],
    "thumb": MANO_FINGERTIP_VERTEX_IDS["ThumbTip"],
}


def build_mano_joints_with_fingertips(
    vertices: torch.Tensor,
    joints16: torch.Tensor,
) -> torch.Tensor:
    """Return MANO joints in the 21-landmark convention.

    Args:
        vertices: MANO vertices with shape (..., V, 3).
        joints16: Native MANO joints with shape (..., 16, 3).

    Returns:
        Tensor with shape (..., 21, 3) ordered as
        :data:`MANO_JOINT_NAMES_WITH_FINGERTIPS`.
    """
    if joints16.shape[-2] != len(MANO_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(MANO_JOINT_NAMES)} native MANO joints, got {joints16.shape[-2]}."
        )
    max_tip_id = max(MANO_FINGERTIP_VERTEX_IDS.values())
    if vertices.shape[-2] <= max_tip_id:
        raise ValueError(f"Expected MANO vertices to include vertex {max_tip_id}.")

    tip_ids = MANO_FINGERTIP_VERTEX_IDS
    return torch.cat(
        [
            joints16[..., 0:1, :],
            joints16[..., 1:4, :],
            vertices[..., tip_ids["IndexTip"] : tip_ids["IndexTip"] + 1, :],
            joints16[..., 4:7, :],
            vertices[..., tip_ids["MiddleTip"] : tip_ids["MiddleTip"] + 1, :],
            joints16[..., 7:10, :],
            vertices[..., tip_ids["PinkyTip"] : tip_ids["PinkyTip"] + 1, :],
            joints16[..., 10:13, :],
            vertices[..., tip_ids["RingTip"] : tip_ids["RingTip"] + 1, :],
            joints16[..., 13:16, :],
            vertices[..., tip_ids["ThumbTip"] : tip_ids["ThumbTip"] + 1, :],
        ],
        dim=-2,
    )


class MANOLayer(_SMPLFamilyLBSLayer):
    """MANO LBS rig adapter implementing the PoseInversion layer contract."""

    def __init__(
        self,
        data_root: str | Path,
        hand_type: str,
        device: str | torch.device = "cpu",
        mode: str = "warp",
        output_unit: Unit | str = Unit.METERS,
    ) -> None:
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}.")
        super().__init__(data_root, device=device, mode=mode, output_unit=output_unit)
        self.hand_type = hand_type
        self.model_type = "mano"
        self.model_spec = f"mano-{hand_type}"
        self.topology_family = "hand"
        self.identity_model_type = "mano_native"
        self.identity_model_kwargs = {"hand_type": hand_type}
        self.rig_data = {"joint_names": MANO_JOINT_NAMES}
        self.default_skin_mesh_name = f"{hand_type}_mano"
        self.base_mesh_path = self.data_root / "MANO" / f"base_hand_{hand_type}.obj"
        self.wrap_mesh_path = self.data_root / "MANO" / f"SOMA_wrap_{hand_type}.obj"

        mano = load_mano_pkl(self.data_root, hand_type)
        parent_ids = mano_parent_ids(mano["kintree_table"])
        if len(parent_ids) != len(MANO_JOINT_NAMES):
            raise ValueError(f"Expected 16 MANO joints, got {len(parent_ids)}.")

        self.num_identity_coeffs = mano["shapedirs"].shape[2]
        self.register_buffer(
            "_v_template",
            torch.from_numpy(mano["v_template"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "_shapedirs",
            torch.from_numpy(mano["shapedirs"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "_J_regressor",
            torch.from_numpy(mano["J_regressor"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "skinning_weights",
            torch.from_numpy(mano["weights"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "joint_parent_ids",
            torch.tensor(parent_ids, dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "faces",
            torch.from_numpy(mano["faces"]).to(dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "posedirs",
            torch.from_numpy(mano["posedirs"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "hands_mean",
            torch.from_numpy(mano["hands_mean"]).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "hands_components",
            torch.from_numpy(mano["hands_components"]).to(self.device),
            persistent=False,
        )

        self.prepare_identity(None)

    def _shape_native(self, identity_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blend = torch.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        verts = self._v_template.unsqueeze(0) + blend
        joints = torch.einsum("jv,bvd->bjd", self._J_regressor, verts)
        wrist = joints[:, 0:1]
        return verts - wrist, joints - wrist
