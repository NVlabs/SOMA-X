import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .units import Unit

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | torch.Tensor
_DEFAULT_CORRECTIVES_MODEL_FILENAME = "correctives_model.pt"


class _DefaultCorrectivesModelPath:
    def __repr__(self) -> str:
        return "_DEFAULT_CORRECTIVES_MODEL_PATH"


_DEFAULT_CORRECTIVES_MODEL_PATH = _DefaultCorrectivesModelPath()


def _resolve_correctives_model_path(
    *,
    data_root: str | Path,
    correctives_model_path: str | Path | None,
    load_correctives_model: bool | None,
    default_enabled: bool = True,
) -> Path | None:
    """Resolve the corrective checkpoint path while preserving the legacy flag."""
    path_is_default = correctives_model_path is _DEFAULT_CORRECTIVES_MODEL_PATH
    path_is_disabled = correctives_model_path is None
    path_is_explicit = not path_is_default and not path_is_disabled

    if load_correctives_model is not None:
        warnings.warn(
            "load_correctives_model is deprecated; use correctives_model_path=None "
            "to disable corrective loading, or pass a checkpoint path.",
            DeprecationWarning,
            stacklevel=3,
        )
        if path_is_explicit:
            raise ValueError(
                "Do not pass load_correctives_model together with an explicit "
                "correctives_model_path."
            )
        if load_correctives_model:
            if path_is_disabled:
                raise ValueError(
                    "load_correctives_model=True conflicts with correctives_model_path=None."
                )
            if not default_enabled:
                raise ValueError(
                    "Correctives require procedural transforms; construct with "
                    "enable_procedural_transforms=True or disable correctives."
                )
            return Path(data_root) / _DEFAULT_CORRECTIVES_MODEL_FILENAME
        return None

    if path_is_disabled:
        return None

    if path_is_default:
        if not default_enabled:
            return None
        return Path(data_root) / _DEFAULT_CORRECTIVES_MODEL_FILENAME

    if not default_enabled:
        raise ValueError(
            "Correctives require procedural transforms; construct with "
            "enable_procedural_transforms=True or disable correctives."
        )

    path = Path(correctives_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Correctives model checkpoint not found: {path}")
    return path


def _as_float_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def _as_index_tensor(indices: ArrayLike, name: str, size: int) -> torch.Tensor:
    idx = torch.as_tensor(indices)
    if idx.dtype == torch.bool:
        if idx.ndim != 1 or idx.numel() != size:
            raise ValueError(
                f"{name} boolean mask must have shape ({size},), got {tuple(idx.shape)}"
            )
        idx = torch.nonzero(idx, as_tuple=False).flatten()
    else:
        if idx.ndim != 1:
            raise ValueError(f"{name} must be a 1D index array, got shape {tuple(idx.shape)}")
        if torch.is_floating_point(idx) or torch.is_complex(idx):
            raise TypeError(f"{name} must contain integer indices")
        idx = idx.to(dtype=torch.long)

    if idx.numel() > 0 and ((idx < 0).any() or (idx >= size).any()):
        raise IndexError(f"{name} contains indices outside [0, {size})")

    return idx.cpu()


def _slice_checkpoint_tensors(
    *,
    bindpose: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    M1_mask: torch.Tensor | None,
    M2_mask: torch.Tensor | None,
    cors_per_joint: int,
    v_index_map: ArrayLike | None = None,
    joint_indices: ArrayLike | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if W2.shape[1] % 3 != 0:
        raise ValueError(f"W2 width must be divisible by 3, got {W2.shape[1]}")

    num_joints = int(bindpose.shape[0])
    num_verts = int(W2.shape[1] // 3)

    if v_index_map is not None:
        v_idx = _as_index_tensor(v_index_map, "v_index_map", num_verts)
        xyz = torch.arange(3, dtype=torch.long)
        col_idx = (v_idx[:, None] * 3 + xyz).reshape(-1)

        W2 = W2[:, col_idx.to(W2.device)]
        M2_mask = M2_mask[:, v_idx.to(M2_mask.device)] if M2_mask is not None else None

    if joint_indices is not None:
        j_idx = _as_index_tensor(joint_indices, "joint_indices", num_joints)
        dof_idx = (j_idx[:, None] * 6 + torch.arange(6, dtype=torch.long)).reshape(-1)
        corrective_idx = (
            j_idx[:, None] * cors_per_joint
            + torch.arange(cors_per_joint, dtype=torch.long)
        ).reshape(-1)

        bindpose = bindpose[j_idx.to(bindpose.device)]
        W1 = W1[dof_idx.to(W1.device)][:, corrective_idx.to(W1.device)]
        W2 = W2[corrective_idx.to(W2.device)]
        if M1_mask is not None:
            m1_j_idx = j_idx.to(M1_mask.device)
            M1_mask = M1_mask[m1_j_idx][:, m1_j_idx]
        M2_mask = M2_mask[j_idx.to(M2_mask.device)] if M2_mask is not None else None

    return bindpose, W1, W2, M1_mask, M2_mask


class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        # ScriptModules don't support requires_grad_(); plain nn.Modules do.
        if not isinstance(module, torch.jit.ScriptModule):
            module.requires_grad_(False)  # freeze: never updated by an optimizer

    def train(self, mode=True):
        # Keep wrapper itself in the requested mode but always hold the
        # inner module in eval so its forward() stays deterministic and
        # never returns the (pred, reg) training-only tuple.
        super().train(mode)
        self.module.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])


class CorrectivesMLP(nn.Module):
    def __init__(
        self,
        *,
        bindpose: ArrayLike,  # (J, 4, 4) or (J, 3, 3)
        cors_per_joint: int,  # Num correctives per joint
        num_verts: int,  # number of vertices in the target mesh
        M1_mask: ArrayLike | None = None,  # (J,J)
        M2_mask: ArrayLike | None = None,  # (J,V)
        W1_init: torch.Tensor | None = None,
        W2_init: torch.Tensor | None = None,
        dropout_p: float = 0.0,  # dropout probability on hidden activations
        use_tanh: bool = True,  # apply tanh after relu on hidden activations
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.J = bindpose.shape[0]  # num joints
        self.C = int(cors_per_joint)  # num correctives per joint
        self.K = self.J * self.C  # total num correctives
        self.I = 6  # num input features per joint
        self.D = self.I * self.J  # total num input features
        self.V = num_verts  # number of vertices in the target mesh

        # ---- non persistent buffers ----
        if M1_mask is not None:
            assert M1_mask.shape[0] == M1_mask.shape[1] and self.J == M1_mask.shape[0], (
                "M1_mask needs to be of shape (J,J)"
            )
            M1_prior = _as_float_tensor(M1_mask).repeat_interleave(self.I, dim=0)
            M1_prior = M1_prior.repeat_interleave(self.C, dim=1)
            self.register_buffer("M1_prior", M1_prior, persistent=False)
            self.register_buffer("M1_mask", _as_float_tensor(M1_mask), persistent=False)
        else:
            self.M1_prior = None
            self.M1_mask = None

        if M2_mask is not None:
            assert M2_mask.shape[0] == self.J and M2_mask.shape[1] == self.V, (
                "M2_mask needs to be of shape (J, V)"
            )
            M2_prior = _as_float_tensor(M2_mask).repeat_interleave(self.C, dim=0)
            M2_prior = M2_prior.repeat_interleave(3, dim=1)
            self.register_buffer("M2_prior", M2_prior, persistent=False)
            self.register_buffer("M2_mask", _as_float_tensor(M2_mask), persistent=False)
        else:
            self.M2_prior = None
            self.M2_mask = None

        self.register_buffer("bindpose", _as_float_tensor(bindpose[:, :3, :3]), persistent=False)

        self.dropout_p = dropout_p

        # ---- parameters ----
        if W1_init is None:
            self.W1 = nn.Parameter(torch.empty(self.D, self.K))
            nn.init.xavier_normal_(self.W1, gain=1.0)
        else:
            self.W1 = nn.Parameter(W1_init)

        if W2_init is None:
            self.W2 = nn.Parameter(torch.zeros(self.K, 3 * self.V))
        else:
            self.W2 = nn.Parameter(W2_init)

    def forward(self, x: torch.Tensor, V: torch.Tensor = None):

        B = x.shape[0]

        x = self.bindpose.transpose(-1, -2).unsqueeze(0) @ x[..., :3, :3]

        x[:, :, 0, 0] -= 1
        x[:, :, 1, 1] -= 1
        input = x[..., :, :2].reshape(B, -1)

        W1 = (self.W1 * self.M1_prior) if self.M1_prior is not None else self.W1
        W2 = (self.W2 * self.M2_prior) if self.M2_prior is not None else self.W2

        z = input @ W1
        z = nn.functional.relu(z)
        if self.use_tanh:
            z = nn.functional.tanh(z)
        z = nn.functional.dropout(z, p=self.dropout_p, training=self.training)
        y = z @ W2

        output = {
            "out": y.view(B, -1, 3),
            "z": z,
            "W2": self.W2,  # raw W2 (used for regularization)
            "W2_masked": W2,  # W2 * M2_prior (actual geometric contribution)
        }

        return output

    def apply_w2_vertex_mask(self, vertex_mask: torch.Tensor) -> None:
        if not isinstance(vertex_mask, torch.Tensor):
            raise TypeError("vertex_mask must be a torch.Tensor")
        if vertex_mask.shape != (self.V,):
            raise ValueError(f"vertex_mask must have shape ({self.V},), got {vertex_mask.shape}")

        mask = vertex_mask.to(dtype=self.W2.dtype, device=self.W2.device)
        mask = mask.reshape(1, self.V, 1).expand(1, self.V, 3).reshape(1, 3 * self.V)
        with torch.no_grad():
            self.W2.mul_(mask)

    # ------------------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        *,
        native_unit: Unit = Unit.CENTIMETERS,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        meta: dict[str, Any] | None = None,
        save_masks=True,
    ) -> None:
        payload = {
            "C_max": self.C,
            "use_tanh": self.use_tanh,
            "bindpose": self.bindpose.cpu(),
            "W1": (
                (self.W1 * self.M1_prior)
                if (self.M1_prior is not None and not save_masks)
                else self.W1
            )
            .detach()
            .to_sparse()
            .cpu(),
            "W2": (
                (self.W2 * self.M2_prior)
                if (self.M2_prior is not None and not save_masks)
                else self.W2
            )
            .detach()
            .to_sparse()
            .cpu(),
            "meta": meta or {},
        }
        if save_masks:
            payload["M1_mask"] = self.M1_mask.to_sparse().cpu()
            payload["M2_mask"] = self.M2_mask.to_sparse().cpu()

        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()

        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()

        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(
        path: str,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
        joint_indices: ArrayLike | None = None,
        v_index_map: ArrayLike | None = None,
        output_unit: Unit = Unit.METERS,
    ):
        if not os.path.exists(path):
            return None

        # Always load to CPU first.  The caller moves the resulting module to
        # the target device afterwards via .to(device).  Loading directly to
        # GPU fails in forked DataLoader workers where the CUDA context is
        # unavailable.
        if torch.__version__ < "2.8" and torch.cuda._is_in_bad_fork():
            # Before PyTorch 2.8, sparse validation checks pinned memory and
            # reinitializes CUDA in forked workers (pytorch/pytorch#153143).
            # Retain the legacy bypass only there. Newer PyTorch avoids that
            # pinning check and must retain its native validation behavior.
            _orig_validate = torch._utils._validate_loaded_sparse_tensors
            torch._utils._validate_loaded_sparse_tensors = lambda: None
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            finally:
                torch._utils._validate_loaded_sparse_tensors = _orig_validate
                # The skipped validator normally releases these references.
                torch._utils._sparse_tensors_to_validate.clear()
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        native_unit = Unit.from_name(ckpt["unit"]) if "unit" in ckpt else Unit.CENTIMETERS
        model_scale = native_unit.meters_per_unit / output_unit.meters_per_unit

        cors_per_joint = ckpt["C_max"]
        use_tanh = ckpt.get("use_tanh", True)
        bindpose = ckpt["bindpose"]
        W1 = ckpt["W1"].to_dense()
        W2 = ckpt["W2"].to_dense() * model_scale

        M1_mask = ckpt["M1_mask"].to_dense() if "M1_mask" in ckpt else None
        M2_mask = ckpt["M2_mask"].to_dense() if "M2_mask" in ckpt else None

        bindpose, W1, W2, M1_mask, M2_mask = _slice_checkpoint_tensors(
            bindpose=bindpose,
            W1=W1,
            W2=W2,
            M1_mask=M1_mask,
            M2_mask=M2_mask,
            cors_per_joint=cors_per_joint,
            v_index_map=v_index_map,
            joint_indices=joint_indices,
        )

        num_verts = int(W2.shape[1] // 3)
        model = CorrectivesMLP(
            cors_per_joint=cors_per_joint,
            bindpose=bindpose,
            num_verts=num_verts,
            M1_mask=M1_mask,
            M2_mask=M2_mask,
            W1_init=W1,
            W2_init=W2,
            use_tanh=use_tanh,
        )
        model = NonPersistentModuleWrapper(model)
        model = model.to(torch.device(map_location))

        if optimizer is not None and "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                logger.warning("Optimizer state not loaded: %s", e)

        if scheduler is not None and "scheduler_state" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                logger.warning("Scheduler state not loaded: %s", e)

        return model
