# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-model base classes shared by the body and hand packages.

The full-body backends live in :mod:`soma.body.identity_model` and the hand
backends in :mod:`soma.hand.identity_model`. The body backend names are still
resolvable from this module (lazily, to avoid an import cycle) so existing
imports and serialized class references keep working.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .geometry.barycentric_interp import BarycentricInterpolator
from .geometry.laplacian import LaplacianMesh
from .units import Unit


class CoordAxis:
    """Named axis constants for declaring a model's native coordinate convention.

    Each constant is a ``(axis_index, sign)`` tuple, where ``axis_index`` is
    0=X, 1=Y, 2=Z and ``sign`` is +1 or -1.

    SOMA standard: Y+ up (``Y``), Z+ forward (``Z``).
    """

    X = (0, +1)
    Y = (1, +1)
    Z = (2, +1)
    NEG_X = (0, -1)
    NEG_Y = (1, -1)
    NEG_Z = (2, -1)


# Parity of each (right_idx, up_idx, fwd_idx) permutation of (0,1,2).
# Used by _apply_coord_transform to derive the right-axis sign so that
# the resulting transform always has determinant +1 (proper rotation).
_PERM_PARITY = {
    (0, 1, 2): +1,
    (1, 2, 0): +1,
    (2, 0, 1): +1,
    (0, 2, 1): -1,
    (2, 1, 0): -1,
    (1, 0, 2): -1,
}


class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """Delegate the forward pass to the wrapped module."""
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return an empty state dict so wrapped weights stay non-persistent."""
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        """Pretend-load state without reporting missing wrapped-module weights."""
        return torch.nn.modules.module._IncompatibleKeys([], [])


class BaseIdentityModel(nn.Module, ABC):
    """Abstract base class for identity models.

    Each subclass **must** declare its native length unit via a ``NATIVE_UNIT``
    class attribute (a :obj:`~soma.units.Unit` enum member).  Omitting it raises
    ``TypeError`` at construction time.

    All internal computation (``get_rest_shape``, ``identity_model_to_soma``,
    LaplacianMesh) operates in native units.  The conversion to the caller's
    desired ``output_unit`` happens once, at the output boundary of ``forward()``.
    """

    NATIVE_UNIT: Unit
    NATIVE_UP: tuple = CoordAxis.Y  # SOMA standard: Y+ up
    NATIVE_FORWARD: tuple = CoordAxis.Z  # SOMA standard: Z+ forward

    def __init__(self, data_root, low_lod, device, output_unit=Unit.METERS, **kwargs):
        nv_lod_mid_to_low = kwargs.pop("nv_lod_mid_to_low", None)
        soma_low_lod_faces = kwargs.pop("soma_low_lod_faces", None)
        super().__init__()
        if not hasattr(self, "NATIVE_UNIT") or not isinstance(self.NATIVE_UNIT, Unit):
            raise TypeError(
                f"{type(self).__name__} must define a NATIVE_UNIT class attribute (a Unit enum member)"
            )
        self.data_root = Path(data_root)
        self.low_lod = low_lod
        self.device = device
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit
        if nv_lod_mid_to_low is not None:
            self.register_buffer("_nv_lod_mid_to_low", nv_lod_mid_to_low, persistent=False)
        else:
            self._nv_lod_mid_to_low = None
        self._soma_low_lod_faces = soma_low_lod_faces

    def _apply_soma_lod(self, V_soma, F_soma=None):
        """Subset SOMA-topology vertices/faces for low LOD.

        Returns (V_soma, F_soma) unchanged when not in low-LOD mode, or the
        subsetted pair when ``_nv_lod_mid_to_low`` is available.
        """
        if self._nv_lod_mid_to_low is None:
            return V_soma, F_soma
        V_low = V_soma[self._nv_lod_mid_to_low]
        F_low = self._soma_low_lod_faces if F_soma is not None else None
        return V_low, F_low

    @property
    @abstractmethod
    def num_identity_coeffs(self) -> int:
        """Number of identity coefficients expected by ``get_rest_shape``."""
        ...

    @property
    def num_scale_params(self) -> int | None:
        """Number of scale parameters expected by ``get_rest_shape``, or ``None`` if unused."""
        return None

    @abstractmethod
    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return the rest shape in NATIVE_UNIT scale."""
        pass

    def _setup_topology_transfer(self, V_source, F_source, V_soma):
        """Set up barycentric interpolation from source topology to SOMA topology.

        Use this when the source mesh already has well-defined inner-face
        geometry (e.g. Anny) and no Laplacian blending is needed.
        All vertex data must be in the model's native units.
        """
        self._to_soma_interp = BarycentricInterpolator(V_source, F_source, V_soma)
        self._laplacian_mesh = None

    def _setup_topology_transfer_with_blending(
        self, V_source, F_source, V_soma, F_soma, vertex_ids_to_exclude
    ):
        """Set up barycentric interpolation plus Laplacian blending.

        Use this when the source mesh lacks inner-face geometry (e.g. eye bags,
        mouth bag) and the excluded vertices need to be solved via a Laplacian
        system to blend smoothly with the surrounding surface.
        All vertex data must be in the model's native units.

        When *vertex_ids_to_exclude* is ``None`` or empty, no Laplacian
        blending is needed and only barycentric interpolation is used.
        """
        self._to_soma_interp = BarycentricInterpolator(V_source, F_source, V_soma)
        if vertex_ids_to_exclude is None or (
            hasattr(vertex_ids_to_exclude, "__len__") and len(vertex_ids_to_exclude) == 0
        ):
            self._laplacian_mesh = None
        else:
            mask_anchors = torch.ones(V_soma.shape[0], dtype=torch.bool, device=self.device)
            mask_anchors[vertex_ids_to_exclude] = False
            self._laplacian_mesh = LaplacianMesh(V_soma, F_soma, mask_anchors=mask_anchors)

    def identity_model_to_soma(self, identity_rest_shape: torch.Tensor) -> torch.Tensor:
        """Transform from source topology to SOMA topology (with optional Laplacian blending)."""
        if hasattr(self, "_to_soma_interp"):
            soma_verts = self._to_soma_interp(identity_rest_shape)
            if self._laplacian_mesh is not None:
                soma_verts = self._laplacian_mesh.solve(soma_verts)
            return soma_verts
        return identity_rest_shape

    def _apply_coord_transform(self, verts: torch.Tensor) -> torch.Tensor:
        """Reorder/negate axes from the model's native convention to SOMA (Y+ up, Z+ forward)."""
        if self.NATIVE_UP == CoordAxis.Y and self.NATIVE_FORWARD == CoordAxis.Z:
            return verts  # already in SOMA frame, no-op
        up_idx, up_sign = self.NATIVE_UP
        fwd_idx, fwd_sign = self.NATIVE_FORWARD
        right_idx = 3 - up_idx - fwd_idx  # 0+1+2=3, so the remaining index
        parity = _PERM_PARITY[(right_idx, up_idx, fwd_idx)]
        right_sign = parity * up_sign * fwd_sign  # ensures det(transform) = +1
        right = verts[..., right_idx : right_idx + 1] * right_sign
        up = verts[..., up_idx : up_idx + 1] * up_sign
        fwd = verts[..., fwd_idx : fwd_idx + 1] * fwd_sign
        return torch.cat([right, up, fwd], dim=-1)

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
        global_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Generate a SOMA-topology rest shape in the requested output unit."""
        identity_rest_shape = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        result = self.identity_model_to_soma(identity_rest_shape)
        result = self._apply_coord_transform(result)
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        if isinstance(global_scale, torch.Tensor):
            result = result * global_scale.reshape(-1, 1, 1)
        elif global_scale != 1.0:
            result = result * global_scale
        return result


_BODY_BACKEND_EXPORTS = frozenset(
    {
        "AnnyIdentityModel",
        "AnnySimplified",
        "GarmentMeasurementIdentityModel",
        "MHRIdentityModel",
        "SMPLIdentityModel",
        "SMPLSimplified",
        "SOMAIdentityModel",
        "create_identity_model",
    }
)


def __getattr__(name: str):
    """Resolve legacy ``soma.identity_model.<body backend>`` references lazily."""
    if name in _BODY_BACKEND_EXPORTS:
        from . import body

        return getattr(body.identity_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseIdentityModel",
    "CoordAxis",
    "NonPersistentModuleWrapper",
    *sorted(_BODY_BACKEND_EXPORTS),
]
