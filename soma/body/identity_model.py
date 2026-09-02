# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-body identity-model backends (SOMA native, MHR, Anny, SMPL family, GarmentMeasurements)."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import trimesh

from .._smpl_family_loader import load_smpl_family_model
from ..identity_model import BaseIdentityModel, CoordAxis, NonPersistentModuleWrapper
from ..units import Unit

logger = logging.getLogger(__name__)


class SMPLSimplified(nn.Module):
    """Minimal SMPL-family shape model for identity-only forward passes."""

    def __init__(self, model_data: dict[str, np.ndarray], device):
        super().__init__()
        self.device = device
        self.faces = model_data["faces"]
        self.v_template = torch.from_numpy(model_data["v_template"]).float().to(device)
        self.shape_dirs = torch.from_numpy(model_data["shapedirs"]).float().to(device)
        self.num_betas = self.shape_dirs.shape[2]

    def forward(self, betas=None):
        blend_shape = torch.einsum("bl,mkl->bmk", [betas, self.shape_dirs])
        v_shaped = self.v_template + blend_shape
        return v_shaped


class AnnySimplified(nn.Module):
    """Wrapper around Anny to simplify the forward pass"""

    def __init__(self, anny_model, device):
        super().__init__()
        self.anny_model = anny_model.to(device=device, dtype=torch.float32)
        self.device = device
        # ignore some local change labels
        full_local_change_labels = self.anny_model.local_change_labels
        self.phenotype_labels = ["gender", "age", "muscle", "weight", "height", "proportions"]

        ignore_names = ["mouth", "eye", "nipple", "cheek", "chin", "ear", "lip", "nose"]
        self.local_change_labels = []
        self.ignore_change_labels = []
        for label in full_local_change_labels:
            keep_label = True
            for ignore_label in ignore_names:
                if ignore_label in label:
                    keep_label = False
                    break
            if keep_label and label not in self.local_change_labels:
                self.local_change_labels.append(label)
            elif not keep_label and label not in self.ignore_change_labels:
                self.ignore_change_labels.append(label)

    def forward(self, phenotype_kwargs=None, local_changes_kwargs=None):
        if (
            isinstance(phenotype_kwargs, torch.Tensor)
            and phenotype_kwargs.ndim == 2
            and phenotype_kwargs.shape[1] == len(self.phenotype_labels)
        ):
            phenotype_kwargs = {
                label: phenotype_kwargs[:, idx] for idx, label in enumerate(self.phenotype_labels)
            }
        if local_changes_kwargs is None:
            local_changes_kwargs = {}
        elif (
            isinstance(local_changes_kwargs, torch.Tensor)
            and local_changes_kwargs.ndim == 2
            and local_changes_kwargs.shape[1] == len(self.local_change_labels)
        ):
            local_changes_kwargs = {
                label: local_changes_kwargs[:, idx]
                for idx, label in enumerate(self.local_change_labels)
            }
        phenotype_kwargs = self.anny_model.parse_phenotype_kwargs(phenotype_kwargs)
        assert set(phenotype_kwargs) <= set(self.anny_model.phenotype_labels), (
            f"Invalid phenotype: {set(phenotype_kwargs) - set(self.anny_model.phenotype_labels)}; available: {self.anny_model.phenotype_labels}"
        )
        blendshape_coeffs = self.anny_model.get_phenotype_blendshape_coefficients(
            **phenotype_kwargs, local_changes=local_changes_kwargs
        )
        rest_vertices = self.anny_model.get_rest_vertices(blendshape_coeffs)
        return rest_vertices


class MHRIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self) -> int:
        return 45

    @property
    def num_scale_params(self) -> int | None:
        # 68 body-part scales.  The MHR TorchScript model expects 204
        # model_parameters = 136 pose + 68 scale.
        return 68

    def __init__(self, data_root, low_lod, device, **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)

        lod = "lod1" if not low_lod else "lod6"
        self.identity_model = NonPersistentModuleWrapper(
            torch.jit.load(
                self.data_root / "MHR" / f"mhr_model_{lod}.pt",
                map_location=self.device,
            )
        )

        mesh_mhr = trimesh.load(
            self.data_root / "MHR" / f"base_body_{lod}.obj", maintain_order=True, process=False
        )
        V_mhr = torch.from_numpy(mesh_mhr.vertices).float().to(device)
        F_mhr = torch.from_numpy(mesh_mhr.faces).to(device)
        mesh_soma = trimesh.load(
            self.data_root / "MHR" / "SOMA_wrap_lod1.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_mhr, F_mhr, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return the rest shape in centimeters (native MHR unit).

        Args:
            identity_coeffs: (B, 45) shape coefficients.
            scale_params: (B, 68) body-part scales (required).
            kwargs: optional dict. Supports:
                ``bone_length_flexibles``: (B, 6) tensor injected into
                pose_params[130:136]. These modify skeleton bone lengths
                (spine, neck, shoulder, arm, hip, leg) and are identity-like
                but stored in MHR's pose vector.
        """
        assert scale_params is not None, "scale_params is required for MHR"
        B = identity_coeffs.shape[0]
        pose_params = torch.zeros(B, 136).to(identity_coeffs.device)
        if kwargs is not None and "bone_length_flexibles" in kwargs:
            pose_params[:, 130:136] = kwargs["bone_length_flexibles"]
        face_expr_params = torch.zeros(B, 72).to(identity_coeffs.device)
        identity_rest_shape, _ = self.identity_model(
            identity_coeffs,
            torch.cat([pose_params, scale_params], dim=1),
            face_expr_params,
        )
        return identity_rest_shape


class AnnyIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Z
    NATIVE_FORWARD = CoordAxis.NEG_Y

    @property
    def num_identity_coeffs(self) -> int:
        return len(self.identity_model.phenotype_labels)

    @property
    def num_scale_params(self) -> int | None:
        return len(self.identity_model.local_change_labels)

    def __init__(self, data_root, low_lod, device, **kwargs):
        # Anny mesh has mouth bag and eye bags so no need to exclude them
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)

        # TODO: reduce Anny's forward pass to just shape parameters
        import anny

        anny_model = anny.create_fullbody_model(
            all_phenotypes=True, local_changes=True, remove_unattached_vertices=True
        )
        self.identity_model = AnnySimplified(anny_model, device)
        mesh_anny = trimesh.load(
            self.data_root / "Anny" / "base_body.obj", maintain_order=True, process=False
        )
        V_anny = torch.from_numpy(mesh_anny.vertices).float().to(device)
        F_anny = torch.from_numpy(mesh_anny.faces).to(device)
        mesh_soma = trimesh.load(
            self.data_root / "Anny" / "SOMA_wrap.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        V_soma, _ = self._apply_soma_lod(V_soma)
        self._setup_topology_transfer(V_anny, F_anny, V_soma)
        self.scale_param_names = tuple(self.identity_model.local_change_labels)

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        rest_shape = self.identity_model(
            phenotype_kwargs=identity_coeffs, local_changes_kwargs=scale_params
        )
        return rest_shape


class SMPLIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self) -> int:
        return self.identity_model.num_betas

    def __init__(self, data_root, low_lod, device, model_type="smpl", **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        imt = model_type
        gender = kwargs.pop("gender", "neutral")
        explicit_model_path = kwargs.pop("model_path", None)
        num_betas = int(kwargs.pop("num_betas", 10))
        super().__init__(data_root, low_lod, device, **kwargs)

        if explicit_model_path is not None:
            model_path = Path(explicit_model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL model not found at '{model_path}'")
            logger.info("Loading %s model from %s", imt.upper(), model_path)
        else:
            model_dir = self.data_root / imt.upper()
            model_path_npz = model_dir / f"{imt.upper()}_{gender.upper()}.npz"
            model_path_pkl = model_dir / f"{imt.upper()}_{gender.upper()}.pkl"
            if model_path_npz.exists():
                model_path = model_path_npz
                logger.info("Loading %s model from %s", imt.upper(), model_path_npz)
            elif model_path_pkl.exists():
                model_path = model_path_pkl
                logger.info("Loading %s model from %s", imt.upper(), model_path_pkl)
            else:
                raise FileNotFoundError(
                    f"Neither {model_path_npz} nor {model_path_pkl} found. Cannot load {imt.upper()} model.\n"
                    "Pass model_path via identity_model_kwargs, or place the file in "
                    f"<data_root>/{imt.upper()}/."
                )

        model_data = load_smpl_family_model(
            model_path,
            model_type=imt,
            num_betas=num_betas,
        )
        self.identity_model = SMPLSimplified(model_data, self.device)

        mesh_smpl = trimesh.load(
            self.data_root / imt.upper() / "base_body.obj",
            maintain_order=True,
            process=False,
        )
        V_smpl = torch.from_numpy(mesh_smpl.vertices).float().to(self.device)
        F_smpl = torch.from_numpy(mesh_smpl.faces).to(self.device)
        mesh_soma = trimesh.load(
            self.data_root / imt.upper() / "SOMA_wrap.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(self.device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(self.device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_smpl, F_smpl, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        rest_shape = self.identity_model(identity_coeffs)
        return rest_shape


class GarmentMeasurementIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self) -> int:
        return self.eigenvalues.shape[0]

    def __init__(self, data_root, low_lod, device, **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "GarmentMeasurements" / "point.npz"

        data = np.load(self.pca_npz_file, allow_pickle=False)
        self.pca_matrix = torch.from_numpy(data["pca_matrix"]).float().to(device)
        self.pca_mean = torch.from_numpy(data["pca_mean"]).float().to(device)
        self.eigenvalues = torch.from_numpy(data["eigenvalues"]).float().to(device)

        mesh_garment = trimesh.load(
            self.data_root / "GarmentMeasurements" / "mean.obj", maintain_order=True, process=False
        )
        F_garment = torch.from_numpy(mesh_garment.faces).to(device)

        V_garment_from_pca = self.pca_mean.reshape(-1, 3)

        mesh_soma = trimesh.load(
            self.data_root / "GarmentMeasurements" / "SOMA_wrap.obj",
            maintain_order=True,
            process=False,
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_garment_from_pca, F_garment, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        weighted_coeffs = identity_coeffs * torch.sqrt(self.eigenvalues)
        weighted_pcas = torch.matmul(weighted_coeffs, self.pca_matrix.T)
        shape_garment = self.pca_mean.unsqueeze(0) + weighted_pcas
        shape_garment = shape_garment.reshape(identity_coeffs.shape[0], -1, 3)
        return shape_garment


class SOMAIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self) -> int:
        return self.eigenvalues.shape[0]

    def __init__(self, data_root, low_lod, device, **kwargs):
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "SOMA_neutral.npz"

        data = np.load(self.pca_npz_file, allow_pickle=False)
        mean = data["mean"]
        shapedirs = data["shapedirs"].reshape(data["shapedirs"].shape[0], -1, 3)
        self._pca_is_lod_subset = self._nv_lod_mid_to_low is not None
        if self._pca_is_lod_subset:
            lod_indices = self._nv_lod_mid_to_low.detach().cpu().numpy()
            mean = mean[lod_indices]
            shapedirs = shapedirs[:, lod_indices, :]
        shapedirs = np.ascontiguousarray(shapedirs.reshape(shapedirs.shape[0], -1))
        mean = np.ascontiguousarray(mean)
        self.register_buffer(
            "pca_matrix",
            torch.from_numpy(shapedirs).float().to(device).T,
            persistent=False,
        )
        self.register_buffer(
            "pca_mean",
            torch.from_numpy(mean).float().to(device).flatten(),
            persistent=False,
        )
        self.register_buffer(
            "eigenvalues",
            torch.from_numpy(data["eigenvalues"]).float().to(device),
            persistent=False,
        )

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        weighted_coeffs = identity_coeffs * torch.sqrt(self.eigenvalues)
        weighted_pcas = torch.matmul(weighted_coeffs, self.pca_matrix.T)
        shape_soma = self.pca_mean.unsqueeze(0) + weighted_pcas
        shape_soma = shape_soma.reshape(identity_coeffs.shape[0], -1, 3)
        if self._nv_lod_mid_to_low is not None and not self._pca_is_lod_subset:
            shape_soma = shape_soma[:, self._nv_lod_mid_to_low, :]
        return shape_soma


def create_identity_model(
    identity_model_type: str,
    data_root,
    low_lod: bool,
    device,
    output_unit: Unit = Unit.METERS,
    **kwargs: Any,
) -> BaseIdentityModel:
    """Factory function to create the appropriate identity model.

    Args:
        output_unit: Desired unit for the model's ``forward()`` output.
            Internally the model operates in its native units; conversion
            to *output_unit* is applied at the output boundary.
    """
    identity_model_type = identity_model_type.lower()

    if identity_model_type == "soma":
        return SOMAIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    if identity_model_type == "mhr":
        return MHRIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    elif identity_model_type == "anny":
        return AnnyIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    elif identity_model_type in ["smplx", "smplh", "smpl"]:
        return SMPLIdentityModel(
            data_root,
            low_lod,
            device,
            model_type=identity_model_type,
            output_unit=output_unit,
            **kwargs,
        )
    elif identity_model_type == "garment":
        return GarmentMeasurementIdentityModel(
            data_root, low_lod, device, output_unit=output_unit, **kwargs
        )
    else:
        raise ValueError(f"Invalid identity model: {identity_model_type}")
