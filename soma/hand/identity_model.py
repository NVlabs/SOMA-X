# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand identity-model backends for SOMA-X."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import trimesh

from ..identity_model import BaseIdentityModel, CoordAxis, NonPersistentModuleWrapper
from ..units import Unit
from ._smpl_family_loader import load_mano_pkl


class BaseHandIdentityModel(BaseIdentityModel):
    """Base class for hand identity models.

    Extends :obj:`~soma.identity_model.BaseIdentityModel` with hand-specific behavior:

    - ``hand_type`` ("left" or "right"): all data is stored left-canonical;
      right-hand output is produced by X-flipping.
    - Wrist centering and X-flip happen in ``forward()``, **after** topology
      transfer and Laplacian blending, so those operations see verts in the
      same frame as the reference meshes (base_body.obj / SOMA_wrap.obj).

    Subclasses implement ``_get_shaped_verts`` returning verts + wrist position
    in the model's native frame.
    """

    def __init__(self, data_root, low_lod, device, hand_type="left", **kwargs):
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got '{hand_type}'")
        self.hand_type = hand_type

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return shaped verts in native frame (same as base_body.obj).

        Wrist centering and X-flip are deferred to ``forward()`` so that
        topology transfer and Laplacian blending operate in the correct frame.
        """
        v_shaped, wrist_pos = self._get_shaped_verts(identity_coeffs)
        self._last_wrist_pos = wrist_pos  # cached for forward()
        return v_shaped

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
        global_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Shape -> topology transfer -> Laplacian blend -> wrist center -> units.

        Per-hand assets (base_hand_{left,right}.obj, SOMA_wrap_{left,right}.obj,
        MANO_{LEFT,RIGHT}.pkl) are loaded independently for each hand, so no
        X-flip is needed.
        """
        identity_rest_shape = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        result = self.identity_model_to_soma(identity_rest_shape)
        result = self._apply_coord_transform(result)
        # Wrist centering (after topology transfer so Laplacian sees native frame)
        wrist_pos = self._last_wrist_pos
        wrist_pos = self._apply_coord_transform(wrist_pos.unsqueeze(1)).squeeze(1)
        result = result - wrist_pos.unsqueeze(1)
        # Unit conversion
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        if isinstance(global_scale, torch.Tensor):
            result = result * global_scale.reshape(-1, 1, 1)
        elif global_scale != 1.0:
            result = result * global_scale
        return result

    def _get_shaped_verts(self, identity_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shaped vertices and wrist position in the model's native frame.

        Args:
            identity_coeffs: (B, K) identity coefficients

        Returns:
            tuple of (v_shaped (B, V, 3), wrist_pos (B, 3))
        """
        raise NotImplementedError


class MANOHandIdentityModel(BaseHandIdentityModel):
    """MANO hand shape model producing SOMAHand-topology rest shapes.

    Loads raw MANO v_template + shapedirs in MANO's native frame (meters),
    shapes them with betas, centers at the wrist joint, and transfers to
    SOMAHand topology via barycentric interpolation.

    Assets required in ``data_root / "MANO"`` (per hand):
      - ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl``   MANO model
      - ``base_hand_left.obj`` / ``base_hand_right.obj``   raw MANO v_template (meters)
      - ``SOMA_wrap_left.obj`` / ``SOMA_wrap_right.obj``   SOMAHand on MANO surface (meters)
    """

    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    _wrist_joint_index = 0

    @property
    def num_identity_coeffs(self) -> int:
        return self._num_betas

    def __init__(self, data_root, low_lod, device, hand_type="left", **kwargs):
        explicit_model_path = kwargs.pop("model_path", None)
        super().__init__(data_root, low_lod, device, hand_type=hand_type, **kwargs)

        mano_dir = self.data_root / "MANO"

        # -- Load per-hand MANO pkl --
        mano_pkl = load_mano_pkl(
            self.data_root,
            hand_type,
            model_path=explicit_model_path,
        )

        v_template = torch.from_numpy(mano_pkl["v_template"]).float().to(device)  # (778, 3)
        shapedirs = (
            torch.from_numpy(np.array(mano_pkl["shapedirs"])).float().to(device)
        )  # (778, 3, 10)
        self._num_betas = shapedirs.shape[2]

        # J_regressor for wrist position
        J_reg = mano_pkl["J_regressor"]
        J_reg = torch.from_numpy(np.array(J_reg)).float().to(device)  # (16, 778)

        self.register_buffer("_v_template", v_template)
        self.register_buffer("_shapedirs", shapedirs)
        self.register_buffer("_J_regressor", J_reg)

        # -- Topology transfer: MANO (778v) -> SOMAHand (2859v) --
        # Laplacian blending at wrist boundary for smooth transition
        mesh_base = trimesh.load(
            mano_dir / f"base_hand_{hand_type}.obj", maintain_order=True, process=False
        )
        V_base = torch.from_numpy(mesh_base.vertices).float().to(device)
        F_base = torch.from_numpy(mesh_base.faces).long().to(device)

        mesh_wrap = trimesh.load(
            mano_dir / f"SOMA_wrap_{hand_type}.obj", maintain_order=True, process=False
        )
        V_wrap = torch.from_numpy(mesh_wrap.vertices).float().to(device)

        # Identify SOMA vertices that have no real MANO correspondence
        # (Laplacian-blended wrist boundary verts far from MANO surface)
        base_trimesh = trimesh.Trimesh(vertices=mesh_base.vertices, faces=mesh_base.faces)
        _, dists, _ = base_trimesh.nearest.on_surface(mesh_wrap.vertices)
        threshold = max(np.median(dists) * 5, 0.002)
        self.no_correspondence_ids = (
            torch.from_numpy(np.where(dists > threshold)[0]).long().to(device)
        )

        self._setup_topology_transfer(V_base, F_base, V_wrap)

    def _get_shaped_verts(self, identity_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape MANO verts with betas, return verts and wrist position.

        Returns:
            (v_shaped (B, 778, 3), wrist_pos (B, 3)) in MANO native meters.
        """
        identity_coeffs.shape[0]
        blend = torch.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        v_shaped = self._v_template.unsqueeze(0) + blend  # (B, 778, 3)
        # Wrist position from shaped verts
        wrist_pos = torch.einsum("jv,bvd->bjd", self._J_regressor, v_shaped)[
            :, self._wrist_joint_index
        ]  # (B, 3)
        return v_shaped, wrist_pos


class MHRHandIdentityModel(BaseHandIdentityModel):
    """MHR hand identity model for SOMAHand.

    Runs the full-body MHR TorchScript model with hand-only parameters
    exposed, then extracts hand vertices from the SOMA-topology output.

    Exposes 5 identity shape coefficients (MHR dims 40-44, hand shape)
    and 26 named scale parameters per hand (overall hand scale + 25
    per-finger segment lengths, offsets, and null transforms).
    """

    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    # Per-hand scale parameter names and their indices into the 68-dim MHR scale vector.
    _SCALE_LAYOUT_RIGHT = [
        (8, "scale_r_hands"),
        (18, "scale_r_index1_length"),
        (19, "scale_r_middle1_length"),
        (20, "scale_r_ring1_length"),
        (21, "scale_r_pinky1_length"),
        (22, "scale_r_thumb1_length"),
        (23, "scale_r_index1_offset"),
        (24, "scale_r_middle1_offset"),
        (25, "scale_r_ring1_offset"),
        (26, "scale_r_pinky1_offset"),
        (27, "scale_r_thumb1_offset"),
        (28, "scale_r_index2_length"),
        (29, "scale_r_middle2_length"),
        (30, "scale_r_ring2_length"),
        (31, "scale_r_pinky2_length"),
        (32, "scale_r_thumb2_length"),
        (33, "scale_r_index3_length"),
        (34, "scale_r_middle3_length"),
        (35, "scale_r_ring3_length"),
        (36, "scale_r_pinky3_length"),
        (37, "scale_r_thumb3_length"),
        (38, "scale_r_index_null_tx"),
        (39, "scale_r_middle_null_tx"),
        (40, "scale_r_ring_null_tx"),
        (41, "scale_r_pinky_null_tx"),
        (42, "scale_r_thumb_null_tx"),
    ]
    _SCALE_LAYOUT_LEFT = [
        (9, "scale_l_hands"),
        (43, "scale_l_index1_length"),
        (44, "scale_l_middle1_length"),
        (45, "scale_l_ring1_length"),
        (46, "scale_l_pinky1_length"),
        (47, "scale_l_thumb1_length"),
        (48, "scale_l_index1_offset"),
        (49, "scale_l_middle1_offset"),
        (50, "scale_l_ring1_offset"),
        (51, "scale_l_pinky1_offset"),
        (52, "scale_l_thumb1_offset"),
        (53, "scale_l_index2_length"),
        (54, "scale_l_middle2_length"),
        (55, "scale_l_ring2_length"),
        (56, "scale_l_pinky2_length"),
        (57, "scale_l_thumb2_length"),
        (58, "scale_l_index3_length"),
        (59, "scale_l_middle3_length"),
        (60, "scale_l_ring3_length"),
        (61, "scale_l_pinky3_length"),
        (62, "scale_l_thumb3_length"),
        (63, "scale_l_index_null_tx"),
        (64, "scale_l_middle_null_tx"),
        (65, "scale_l_ring_null_tx"),
        (66, "scale_l_pinky_null_tx"),
        (67, "scale_l_thumb_null_tx"),
    ]

    # MHR identity shape: dims 40-44 are hand shape (5 dims out of 45).
    _MHR_HAND_IDENTITY_OFFSET = 40
    _MHR_IDENTITY_DIM = 45

    @property
    def num_identity_coeffs(self) -> int:
        return 5

    @property
    def num_scale_params(self) -> int | None:
        return 26

    def __init__(self, data_root, low_lod, device, hand_type="left", **kwargs):
        super().__init__(data_root, low_lod, device, hand_type=hand_type, **kwargs)

        # Scale layout for this hand
        layout = self._SCALE_LAYOUT_LEFT if hand_type == "left" else self._SCALE_LAYOUT_RIGHT
        self._scale_mhr_indices = [idx for idx, _ in layout]
        self.scale_param_names = [name for _, name in layout]

        hand_data = np.load(self.data_root / "SOMAHand.npz", allow_pickle=False)

        # Load MHR TorchScript model
        self._mhr_model = NonPersistentModuleWrapper(
            torch.jit.load(
                self.data_root / "MHR" / "mhr_model_lod1.pt",
                map_location=self.device,
            )
        )

        # Full-body MHR -> SOMA topology transfer
        mesh_mhr = trimesh.load(
            self.data_root / "MHR" / "base_body_lod1.obj",
            maintain_order=True,
            process=False,
        )
        V_mhr = torch.from_numpy(mesh_mhr.vertices).float().to(device)
        F_mhr = torch.from_numpy(mesh_mhr.faces).to(device)
        mesh_soma = trimesh.load(
            self.data_root / "MHR" / "SOMA_wrap_lod1.obj",
            maintain_order=True,
            process=False,
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(V_mhr, F_mhr, V_soma, F_soma, None)

        # Hand vertex IDs and wrist boundary from SOMAHand.npz
        self._hand_vert_ids = hand_data[f"{hand_type}_vert_ids"]
        # Wrist boundary vertex indices in the full SOMA mesh — used for
        # wrist centering after topology transfer.
        boundary_local = hand_data[f"{hand_type}_boundary_loop"]
        self._wrist_boundary_global = hand_data[f"{hand_type}_vert_ids"][boundary_local]

    def _get_shaped_verts(self, identity_coeffs: torch.Tensor) -> torch.Tensor:
        """Run MHR with hand identity, return full-body MHR vertices."""
        B = identity_coeffs.shape[0]
        device = identity_coeffs.device

        # Pad 5-dim hand identity to 45-dim MHR identity.
        # Use F.pad (not in-place assignment) to preserve gradient flow.
        n_pad_left = self._MHR_HAND_IDENTITY_OFFSET
        n_pad_right = self._MHR_IDENTITY_DIM - n_pad_left - 5
        full_identity = torch.nn.functional.pad(
            identity_coeffs, (n_pad_left, n_pad_right)
        )  # (B, 45)

        # Build 68-dim scale from stored 26-dim hand scale.
        # Use scatter (not in-place assignment) to preserve gradient flow.
        full_scale = torch.zeros(B, 68, device=device)
        if self._current_scale_params is not None:
            idx = (
                torch.tensor(
                    self._scale_mhr_indices,
                    device=device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(B, -1)
            )
            full_scale = full_scale.scatter(1, idx, self._current_scale_params)

        # MHR model expects (identity_45, cat(pose_136, scale_68), face_72)
        pose_params = torch.zeros(B, 136, device=device)
        face_params = torch.zeros(B, 72, device=device)
        model_params = torch.cat([pose_params, full_scale], dim=1)

        mhr_verts, _ = self._mhr_model(full_identity, model_params, face_params)
        return mhr_verts  # (B, V_mhr, 3) in cm

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Override to pass scale_params through to _get_shaped_verts."""
        self._current_scale_params = scale_params
        result = self._get_shaped_verts(identity_coeffs)
        self._current_scale_params = None
        return result

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
        global_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Shape -> topology transfer -> hand extraction -> wrist center -> units."""
        # Get full-body MHR verts in native MHR frame
        identity_rest_shape = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        # Topology transfer: MHR mesh -> full SOMA mesh (frame changes here)
        full_soma = self.identity_model_to_soma(identity_rest_shape)
        full_soma = self._apply_coord_transform(full_soma)
        # Wrist position: mean of boundary loop vertices in the transferred mesh
        wrist_pos = full_soma[:, self._wrist_boundary_global].mean(dim=1)  # (B, 3)
        # Extract hand vertices and center at wrist
        result = full_soma[:, self._hand_vert_ids]
        result = result - wrist_pos.unsqueeze(1)
        # Unit conversion
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        if isinstance(global_scale, torch.Tensor):
            result = result * global_scale.reshape(-1, 1, 1)
        elif global_scale != 1.0:
            result = result * global_scale
        return result


class SOMAHandIdentityModel(BaseIdentityModel):
    """Hand-specific PCA identity model.

    Loads the left-hand PCA from SOMAHand.npz.  For the right hand, mirrors
    the left PCA by negating the X component of mean and shapedirs.
    """

    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    def __init__(
        self,
        data_root,
        low_lod,
        device,
        *,
        hand_map,
        hand_type,
        **kwargs,
    ):
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)

        required_keys = ("left_mean", "left_shapedirs", "left_eigenvalues")
        missing_keys = [key for key in required_keys if key not in hand_map]
        if missing_keys:
            raise KeyError(
                f"SOMAHand.npz is missing required hand PCA arrays: {', '.join(missing_keys)}"
            )

        mean = hand_map["left_mean"].astype(np.float64)  # (Vh, 3) wrist-local cm
        sd = hand_map["left_shapedirs"].astype(np.float64)  # (K, Vh*3)
        eigenvalues = hand_map["left_eigenvalues"]  # (K,)

        if hand_type == "right":
            mean = mean.copy()
            mean[:, 0] *= -1
            K_pca = sd.shape[0]
            Vh = mean.shape[0]
            sd = sd.reshape(K_pca, Vh, 3).copy()
            sd[:, :, 0] *= -1
            sd = sd.reshape(K_pca, Vh * 3)

        self.register_buffer(
            "pca_mean", torch.from_numpy(mean.flatten().astype(np.float32)).to(device)
        )
        self.register_buffer("pca_matrix", torch.from_numpy(sd.astype(np.float32)).to(device))
        self.register_buffer(
            "eigenvalues", torch.from_numpy(eigenvalues.astype(np.float32)).to(device)
        )

    @property
    def num_identity_coeffs(self) -> int:
        return self.eigenvalues.shape[0]

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return hand rest shape in wrist-local centimeters.

        Args:
            identity_coeffs: (B, K)
            scale_params: unused; accepted for BaseIdentityModel compatibility.
            kwargs: unused; accepted for BaseIdentityModel compatibility.
        Returns:
            (B, Vh, 3) wrist-local vertices in native SOMA centimeters.
        """
        del scale_params, kwargs
        weighted = identity_coeffs * torch.sqrt(self.eigenvalues)
        shape = weighted @ self.pca_matrix + self.pca_mean.unsqueeze(0)
        return shape.reshape(identity_coeffs.shape[0], -1, 3)
