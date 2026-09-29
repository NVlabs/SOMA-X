# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real sparse corrective checkpoints load without model assets or downloads."""

from unittest.mock import Mock

import pytest
import torch

from soma.correctives_model import CorrectivesMLP
from soma.units import Unit


@pytest.fixture
def checkpoint(tmp_path):
    model = CorrectivesMLP(
        bindpose=torch.eye(3).repeat(2, 1, 1),
        cors_per_joint=1,
        num_verts=2,
        M1_mask=torch.eye(2),
        M2_mask=torch.ones(2, 2),
        W1_init=torch.arange(24, dtype=torch.float32).reshape(12, 2),
        W2_init=torch.arange(12, dtype=torch.float32).reshape(2, 6),
    )
    path = tmp_path / "correctives.pt"
    model.save_checkpoint(path)
    return path, model


@pytest.mark.parametrize("save_masks", [False, True])
def test_sparse_checkpoint_roundtrip(checkpoint, save_masks, monkeypatch):
    path, original = checkpoint
    original.save_checkpoint(path, save_masks=save_masks)
    validate = Mock(wraps=torch._utils._validate_loaded_sparse_tensors)
    monkeypatch.setattr(torch._utils, "_validate_loaded_sparse_tensors", validate)

    loaded = CorrectivesMLP.load_checkpoint(path, output_unit=Unit.CENTIMETERS)

    validate.assert_called_once()
    assert torch._utils._validate_loaded_sparse_tensors is validate
    assert not torch._utils._sparse_tensors_to_validate
    poses = torch.eye(3).repeat(1, 2, 1, 1)
    poses[:, :, 0, 0] = 1.5
    expected = original(poses)["out"]
    assert expected.abs().max() > 0
    torch.testing.assert_close(loaded(poses)["out"], expected)
    assert all(p.device.type == "cpu" for p in loaded.parameters())


@pytest.mark.parametrize(
    "version,bad_fork",
    [
        ("2.6.0+cu124", False),
        ("2.7.1+cu126", False),
        ("2.8.0+cu128", True),
        ("2.10.0+cu130", True),
        ("2.14.0+cu130", True),
    ],
)
def test_native_validation_outside_legacy_cuda_fork(checkpoint, monkeypatch, version, bad_fork):
    path, _ = checkpoint
    monkeypatch.setattr(torch, "__version__", type(torch.__version__)(version))
    monkeypatch.setattr(torch.cuda, "_is_in_bad_fork", lambda: bad_fork)
    validate = Mock(wraps=torch._utils._validate_loaded_sparse_tensors)
    monkeypatch.setattr(torch._utils, "_validate_loaded_sparse_tensors", validate)

    assert CorrectivesMLP.load_checkpoint(path) is not None

    validate.assert_called_once()
    assert torch._utils._validate_loaded_sparse_tensors is validate
    assert not torch._utils._sparse_tensors_to_validate


@pytest.mark.parametrize("version", ["2.6.0+cu124", "2.7.1+cu126"])
@pytest.mark.parametrize("load_fails", [False, True])
def test_legacy_fork_restores_validator_and_clears_pending(
    checkpoint, monkeypatch, version, load_fails
):
    path, _ = checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    monkeypatch.setattr(torch, "__version__", type(torch.__version__)(version))
    monkeypatch.setattr(torch.cuda, "_is_in_bad_fork", lambda: True)
    validate = Mock(side_effect=RuntimeError("legacy CUDA pinning check"))
    monkeypatch.setattr(torch._utils, "_validate_loaded_sparse_tensors", validate)
    pending = []
    monkeypatch.setattr(torch._utils, "_sparse_tensors_to_validate", pending)

    def legacy_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        pending.append(payload["W1"])
        if load_fails:
            raise OSError("checkpoint read failed")
        # Pre-2.8 serialization calls this hook without arguments.
        torch._utils._validate_loaded_sparse_tensors()
        return payload

    monkeypatch.setattr(torch, "load", legacy_load)
    if load_fails:
        with pytest.raises(OSError, match="checkpoint read failed"):
            CorrectivesMLP.load_checkpoint(path)
    else:
        assert CorrectivesMLP.load_checkpoint(path) is not None

    validate.assert_not_called()
    assert torch._utils._validate_loaded_sparse_tensors is validate
    assert not pending


def test_rejects_invalid_sparse_checkpoint(checkpoint):
    path, _ = checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # Duplicate in-bounds indices falsely marked coalesced: invalid, but safe
    # to exercise even against the old validation-bypassing implementation.
    payload["W1"] = torch.sparse_coo_tensor(
        torch.zeros(2, 2, dtype=torch.int64),
        torch.ones(2),
        size=(12, 2),
        is_coalesced=True,
        check_invariants=False,
    )
    torch.save(payload, path)

    with torch.sparse.check_sparse_tensor_invariants():
        with pytest.raises(RuntimeError, match="coalesced"):
            CorrectivesMLP.load_checkpoint(path)


def test_missing_checkpoint_returns_none(tmp_path):
    assert CorrectivesMLP.load_checkpoint(tmp_path / "missing.pt") is None
