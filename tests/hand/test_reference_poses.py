# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical hand references preserve the current hand coordinate frame."""

import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from soma import SOMAHandLayer
from soma.geometry.lbs import batch_rodrigues

ASSETS = Path(__file__).resolve().parents[2] / "assets"
SELECTOR = {"version": "v0.3.0", "data_key": "t_pose_world"}
pytestmark = pytest.mark.asset_heavy


class _PoseCorrectives(torch.nn.Module):
    def __init__(self, num_vertices):
        super().__init__()
        self.num_vertices = num_vertices

    def forward(self, rotations):
        offsets = rotations[:, :, :, 0].mean(dim=1) * 0.002
        return {"out": offsets[:, None].expand(-1, self.num_vertices, -1)}


@pytest.fixture(
    scope="module",
    params=[pytest.param("cpu", marks=pytest.mark.cpu)]
    + ([pytest.param("cuda", marks=pytest.mark.gpu)] if torch.cuda.is_available() else []),
)
def device(request):
    return request.param


@pytest.fixture(scope="module", params=["left", "right"])
def hand(request, device):
    layer = SOMAHandLayer(
        data_root=ASSETS,
        hand_type=request.param,
        device=device,
        mode="dense",
        correctives_model_path=None,
    ).double()
    layer.correctives_model = _PoseCorrectives(len(layer.bind_shape))
    return layer


@pytest.fixture
def prepared_hand(hand):
    hand.prepare_identity(hand.bind_shape.new_zeros(1, hand.num_shape_components))
    return hand


def _angles(layer, batch=2):
    return torch.linspace(
        -0.25,
        0.35,
        batch * 25 * 3,
        device=layer.bind_shape.device,
        dtype=layer.bind_shape.dtype,
    ).reshape(batch, 25, 3)


def _rotations(angles):
    return batch_rodrigues(angles.reshape(-1, 3), dtype=angles.dtype).reshape(
        *angles.shape[:-1], 3, 3
    )


def _absolute_oracle(rotations, reference, parents):
    """Accumulate motion in the reference wrist frame, then recover local rotations."""
    motion = [rotations[:, 0]]
    world = [torch.linalg.solve(reference[0], motion[0] @ reference[0])]
    for joint in range(1, len(parents)):
        motion.append(motion[parents[joint]] @ rotations[:, joint])
        world.append(torch.linalg.solve(reference[0], motion[joint] @ reference[joint]))
    local = [world[0]]
    local.extend(torch.linalg.solve(world[parents[j]], world[j]) for j in range(1, len(parents)))
    return torch.stack(local, dim=1), torch.stack(world, dim=1)


def _assert_output_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-6)


def test_historical_getter_is_offline_and_uses_current_hand_frame(hand, monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("Historical references must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    with np.load(ASSETS / "SOMA_neutral.npz", allow_pickle=False) as data:
        for version, key in [("0.1.0", "t_pose_world__v0.1.0"), ("v0.3.0", "t_pose_world__v0.2.0")]:
            body = torch.as_tensor(
                data[key], dtype=hand.bind_pose_world.dtype, device=hand.bind_pose_world.device
            )
            expected = hand._correctives_to_hand_frame @ body[hand.hand_joint_ids_global]
            actual = hand.get_reference_pose(version=version, data_key="t_pose_world")
            assert actual.shape == (25, 3, 3)
            assert actual.device == hand.bind_pose_world.device
            assert actual.dtype == hand.bind_pose_world.dtype
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(hand.get_reference_pose(key), actual)
            actual.zero_()
            torch.testing.assert_close(hand.get_reference_pose(key), expected, atol=2e-6, rtol=2e-6)
    current = hand.get_reference_pose(**SELECTOR)
    torch.testing.assert_close(current, hand.t_pose_world[:, :3, :3], atol=2e-6, rtol=2e-6)
    records = hand.list_reference_poses()
    assert {record["id"] for record in records} >= {"t_pose_world__v0.1.0", "t_pose_world__v0.2.0"}
    assert {record["version"] for record in records if "version" in record} == {"0.1.0", "0.2.0"}
    assert all("soma_version" not in record for record in records)
    records[0]["description"] = "changed by caller"
    assert hand.list_reference_poses()[0].get("description") != "changed by caller"


@pytest.mark.parametrize("entry_point", ["getter", "pose", "forward"])
def test_obsolete_version_keyword_is_rejected(prepared_hand, entry_point):
    hand = prepared_hand
    selector = {"soma_version": "0.1.0"}
    with pytest.raises(TypeError, match="soma_version"):
        if entry_point == "getter":
            hand.get_reference_pose(**selector)
        elif entry_point == "pose":
            hand.pose(_angles(hand), reference_pose=selector)
        else:
            hand(
                _angles(hand),
                hand.bind_shape.new_zeros(1, hand.num_shape_components),
                reference_pose=selector,
            )


@pytest.mark.parametrize("apply_correctives", [False, True])
@pytest.mark.parametrize("matrix_input", [False, True])
def test_reference_tensor_dict_and_absolute_pose_agree(
    prepared_hand, apply_correctives, matrix_input
):
    hand = prepared_hand
    angles = _angles(hand)
    rotations = _rotations(angles)
    selector = {"version": "0.1.0", "data_key": "t_pose_world"}
    reference = hand.get_reference_pose(**selector)
    absolute, expected_world = _absolute_oracle(
        rotations, reference, hand.joint_parent_ids.tolist()
    )
    translation = angles.new_tensor([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])
    options = {"apply_correctives": apply_correctives, "global_translation": translation}
    expected = hand.pose(absolute, pose2rot=False, absolute_pose=True, **options)
    poses = rotations if matrix_input else angles
    for reference_pose in (reference, selector):
        actual = hand.pose(
            poses, pose2rot=not matrix_input, reference_pose=reference_pose, **options
        )
        _assert_output_close(actual, expected)
        torch.testing.assert_close(
            actual["transforms"][..., :3, :3], expected_world, atol=2e-6, rtol=2e-6
        )
        fk = hand.pose(
            poses, pose2rot=not matrix_input, reference_pose=reference_pose, fk_only=True, **options
        )
        assert "vertices" not in fk
        torch.testing.assert_close(fk["transforms"], actual["transforms"], atol=2e-6, rtol=2e-6)
    assert selector == {"version": "0.1.0", "data_key": "t_pose_world"}


@pytest.mark.parametrize("apply_correctives", [False, True])
def test_forward_matches_prepared_pose(prepared_hand, apply_correctives):
    hand = prepared_hand
    angles = _angles(hand, batch=1)
    identity = hand.bind_shape.new_zeros(1, hand.num_shape_components)
    hand.prepare_identity(identity, repose_to_bind_pose=apply_correctives)
    expected = hand.pose(angles, reference_pose=SELECTOR, apply_correctives=apply_correctives)
    _assert_output_close(
        hand(angles, identity, reference_pose=SELECTOR, apply_correctives=apply_correctives),
        expected,
    )


def test_default_reference_and_cached_template_are_unchanged(prepared_hand):
    hand = prepared_hand
    angles = _angles(hand)
    template = hand.t_pose_world.clone()
    expected = hand.pose(angles)
    _assert_output_close(hand.pose(angles, reference_pose=None), expected)
    _assert_output_close(hand.pose(angles, reference_pose=SELECTOR), expected)
    translated = template.clone()
    translated[:, :3, 3] += 100
    _assert_output_close(hand.pose(angles, reference_pose=translated), expected)
    hand.pose(angles, reference_pose={"version": "0.1.0"})
    _assert_output_close(hand.pose(angles), expected)
    torch.testing.assert_close(hand.t_pose_world, template, atol=0, rtol=0)


def test_custom_reference_and_pose_gradients(prepared_hand):
    hand = prepared_hand
    angles = _angles(hand, batch=1).requires_grad_()
    reference_angles = (
        torch.linspace(-0.2, 0.3, 75, dtype=angles.dtype, device=angles.device)
        .reshape(25, 3)
        .requires_grad_()
    )
    reference = _rotations(reference_angles)
    actual = hand.pose(angles, reference_pose=reference, apply_correctives=True)
    weights = torch.linspace(
        0.1, 1.0, actual["vertices"].numel(), dtype=angles.dtype, device=angles.device
    )
    loss = (actual["vertices"].reshape(-1) * weights).sum()
    gradients = torch.autograd.grad(loss, (angles, reference_angles), retain_graph=True)
    absolute, _ = _absolute_oracle(_rotations(angles), reference, hand.joint_parent_ids.tolist())
    expected = hand.pose(absolute, pose2rot=False, absolute_pose=True, apply_correctives=True)
    expected_loss = (expected["vertices"].reshape(-1) * weights).sum()
    expected_gradients = torch.autograd.grad(expected_loss, (angles, reference_angles))
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().max() > 1e-4
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "problem", ["batch", "count", "shape", "nan", "reflection", "scale", "type"]
)
def test_invalid_reference_tensor(prepared_hand, problem):
    hand = prepared_hand
    reference = hand.get_reference_pose(**SELECTOR)
    if problem == "batch":
        reference = reference[None]
    elif problem == "count":
        reference = reference[1:]
    elif problem == "shape":
        reference = reference[..., :2]
    elif problem == "nan":
        reference[2, 0, 0] = float("nan")
    elif problem == "reflection":
        reference[2, :, 0] *= -1
    elif problem == "scale":
        reference[2] *= 1.2
    else:
        reference = reference.cpu().numpy()
    with pytest.raises(TypeError if problem == "type" else ValueError, match="reference_pose"):
        hand.pose(_angles(hand), reference_pose=reference)


@pytest.mark.parametrize(
    "selector,error",
    [
        ({}, ValueError),
        ({"version": "0.0.1"}, KeyError),
        ({"version": "not-a-version"}, ValueError),
        ({"version": "0.1.0", "data_key": "missing"}, KeyError),
        ({"version": "0.1.0", "reference_id": "t_pose_world__v0.1.0"}, ValueError),
        ({"unknown": "value"}, TypeError),
    ],
)
def test_invalid_selector_fails_before_identity_mutation(prepared_hand, selector, error):
    hand = prepared_hand
    rest = hand._cached_rest_shape
    bind = hand._cached_bind_transforms_world
    with pytest.raises(error):
        hand(
            _angles(hand),
            hand.bind_shape.new_zeros(2, hand.num_shape_components),
            reference_pose=selector,
        )
    assert hand._cached_rest_shape is rest
    assert hand._cached_bind_transforms_world is bind


def test_absolute_reference_conflict_fails_before_identity_mutation(prepared_hand):
    hand = prepared_hand
    rest = hand._cached_rest_shape
    bind = hand._cached_bind_transforms_world
    with pytest.raises(ValueError, match="absolute_pose"):
        hand(
            _angles(hand),
            hand.bind_shape.new_zeros(2, hand.num_shape_components),
            reference_pose=SELECTOR,
            absolute_pose=True,
        )
    assert hand._cached_rest_shape is rest
    assert hand._cached_bind_transforms_world is bind


def test_float32_getter_can_be_passed_to_pose(hand):
    # Getter output must remain valid with reduced-precision CUDA matmul enabled.
    precision = "tf32" if hand.bind_pose_world.device.type == "cuda" else "none"
    hand.float()
    try:
        with torch.backends.flags(fp32_precision=precision):
            hand.prepare_identity(hand.bind_shape.new_zeros(1, hand.num_shape_components))
            reference = hand.get_reference_pose(**SELECTOR)
            assert reference.dtype == torch.float32
            assert reference.device == hand.bind_pose_world.device
            angles = _angles(hand)
            actual = hand.pose(angles, reference_pose=reference)
            expected = hand.pose(angles)
            _assert_output_close(actual, expected)
    finally:
        hand.double()
