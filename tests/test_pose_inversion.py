# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PoseInversion.

Tests both the default ``fit()`` solver (analytical inverse-LBS warm start plus
Lie-GN refinement) and ``fit(autograd_iters=...)`` (FK-based gradient
optimization) against ground-truth posed vertices from example_animation.npy.

Pose conventions
~~~~~~~~~~~~~~~~
- example_animation.npy stores local rotations *relative to T-pose*
  (joint orient not applied).  demo_soma_vis.py applies a t-pose
  correction before passing to ``soma.pose(absolute_pose=False)``.

- Both ``fit()`` and ``fit(autograd_iters=...)`` return *absolute*
  local rotations (joint orient already baked in), suitable for
  ``soma.pose(absolute_pose=True)`` or direct LBS via
  ``BatchedSkinning.pose(absolute_pose=True)``.

Requires CUDA and assets/.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
MOTION_FILE = ASSETS_DIR / "example_animation.npy"

# 94-joint skeleton to 77-joint mapping (from demo_soma_vis.py)
# fmt: off
_NVSKEL93_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "LeftForeArmTwist1", "LeftForeArmTwist2", "LeftArmTwist1", "LeftArmTwist2",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "RightForeArmTwist1", "RightForeArmTwist2", "RightArmTwist1", "RightArmTwist2",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "LeftShinTwist1", "LeftShinTwist2", "LeftLegTwist1", "LeftLegTwist2",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    "RightShinTwist1", "RightShinTwist2", "RightLegTwist1", "RightLegTwist2",
]
_NVSKEL77_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
# fmt: on
_93TO77_IDX = [_NVSKEL93_NAMES.index(n) for n in _NVSKEL77_NAMES]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_joint_pose_prior_weights_overrides_named_joints():
    """Per-joint pose-prior weights should override only named joints."""
    from soma.fitting.pose_inversion import _joint_pose_prior_weights

    weights = _joint_pose_prior_weights(
        ["Root", "Hips", "LeftShin"],
        {"Hips": 0.25, "LeftShin": 5.0},
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert weights.tolist() == [1.0, 0.25, 5.0]


def test_joint_pose_prior_weights_rejects_unknown_joint():
    """Typos in explicit pose-prior weights should fail loudly."""
    from soma.fitting.pose_inversion import _joint_pose_prior_weights

    with pytest.raises(ValueError, match="Unknown joint"):
        _joint_pose_prior_weights(
            ["Root", "Hips"],
            {"NoSuchJoint": 2.0},
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_lie_gn_solve_falls_back_for_singular_batch_element():
    """Singular normal-equation batches should return finite deterministic steps."""
    from soma.fitting.pose_inversion import _solve_lie_gn_normal_equations

    JtJ = torch.eye(3).repeat(2, 1, 1)
    JtJ[1] = 0.0
    rhs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    delta = _solve_lie_gn_normal_equations(JtJ, rhs)

    assert torch.allclose(delta[0], rhs[0])
    assert torch.allclose(delta[1], torch.zeros(3))
    assert torch.isfinite(delta).all()


def test_lie_gn_active_joints_exclude_virtual_root_but_include_hand_root():
    """Full-body virtual roots are fixed; a hand-only wrist root remains active."""
    from soma.fitting.pose_inversion import _active_lie_joint_indices

    descendant_weights = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    assert _active_lie_joint_indices(descendant_weights, root_idx=1).tolist() == [1]
    assert _active_lie_joint_indices(descendant_weights, root_idx=0).tolist() == [0, 1]


def test_pose_inversion_exposes_skeleton_transfer_rotation_method():
    from soma.fitting.pose_inversion import PoseInversion

    class FakeSoma:
        low_lod = True
        bind_shape = torch.empty(0, 3)
        nv_lod_mid_to_low = None
        rig_data = {"bind_shape": torch.empty(0, 3)}
        root_joint_idx = 1
        output_unit = None
        identity_model_type = "soma"

    inv = PoseInversion(FakeSoma(), skeleton_transfer_rotation_method="auto")

    assert inv.skeleton_transfer_rotation_method == "auto"
    inv.skeleton_transfer_rotation_method = "newton-schulz"
    assert inv.skeleton_transfer_rotation_method == "newton-schulz"
    with pytest.raises(ValueError, match="Unknown skeleton_transfer_rotation_method"):
        inv.skeleton_transfer_rotation_method = "svd"


def test_pose_inversion_exposes_refit_rotation_method():
    from soma.fitting.pose_inversion import PoseInversion

    class FakeSoma:
        low_lod = True
        bind_shape = torch.empty(0, 3)
        nv_lod_mid_to_low = None
        rig_data = {"bind_shape": torch.empty(0, 3)}
        root_joint_idx = 1
        output_unit = None
        identity_model_type = "soma"

    inv = PoseInversion(FakeSoma(), refit_rotation_method="auto")

    assert inv.refit_rotation_method == "auto"
    with pytest.raises(ValueError, match="Unknown refit_rotation_method"):
        inv.refit_rotation_method = "svd"


def test_pose_inversion_uses_automatic_default_rotation_methods():
    from soma.fitting.pose_inversion import PoseInversion

    class FakeSoma:
        low_lod = True
        bind_shape = torch.empty(0, 3)
        nv_lod_mid_to_low = None
        rig_data = {"bind_shape": torch.empty(0, 3)}
        root_joint_idx = 1
        output_unit = None
        identity_model_type = "soma"

    inv = PoseInversion(FakeSoma())

    assert inv.skeleton_transfer_rotation_method == "auto"
    assert inv.refit_rotation_method == "auto"


def _load_motion(soma, frames):
    """Load example_animation.npy frames, return ground-truth posed vertices.

    Follows the same pipeline as tools/demo_soma_vis.py:
    1. Remap 94-joint → 78-joint (root + 77)
    2. Apply t-pose correction
    3. Forward pass through soma.pose()

    Returns (posed_vertices, root_translation).
    """
    from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local

    device = soma.device
    motion_full = torch.from_numpy(np.load(MOTION_FILE)).float().to(device)
    rot_local = motion_full[..., :3, :3]
    root_trans = motion_full[:, 1, :3, 3]

    # Remap 94 → 78 joints (root + 77)
    if rot_local.shape[1] == 94:
        subset_idx = [0] + [i + 1 for i in _93TO77_IDX]
        rot_local = rot_local[:, subset_idx]

    # T-pose correction: animation data is in a different skeleton
    # convention; rotate world transforms to match SOMA's joint orient.
    public_rig = soma.public_rig_view()
    public_parent_ids = public_rig.joint_parent_ids
    correction = public_rig.t_pose_world[:, :3, :3].transpose(-2, -1)
    rot_world = joint_local_to_world(rot_local, public_parent_ids)
    rot_world = rot_world @ correction
    rot_local = joint_world_to_local(rot_world, public_parent_ids)

    # Build pose: global_orient (Hips=joint 1) + body (joints 2:)
    global_orient = rot_local[:, 1]
    body_pose = rot_local[:, 2:]
    pose = torch.cat([global_orient.unsqueeze(1), body_pose], dim=1)
    transl = root_trans

    # Select frames
    pose = pose[frames]
    transl = transl[frames]

    # Forward pass — these rotations are relative to T-pose. Pose inversion fits
    # raw LBS, so keep correctives out of the target vertices.
    with torch.no_grad():
        out = soma.pose(
            pose,
            transl=transl,
            pose2rot=False,
            absolute_pose=False,
            apply_correctives=False,
        )

    return out["vertices"], transl


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.xlo
@pytest.mark.asset_heavy
@requires_cuda
def test_xlo_layer_default_inversion_uses_xlo_topology():
    """PoseInversion(xlo_layer) should not try to downsample xlo to low LOD."""
    from soma.body import SOMALayer
    from soma.fitting.pose_inversion import PoseInversion

    soma = SOMALayer(
        data_root=str(ASSETS_DIR),
        identity_model_type="soma",
        device="cuda",
        mode="warp",
        lod="xlo",
    )
    identity_coeffs = torch.zeros(1, soma.identity_model.num_identity_coeffs, device="cuda")
    soma.prepare_identity(identity_coeffs)

    inv = PoseInversion(soma)
    assert inv.soma is soma
    inv.prepare_identity(identity_coeffs)

    pose = torch.eye(3, device="cuda").reshape(1, 1, 3, 3).expand(1, 77, 3, 3).contiguous()
    transl = torch.zeros(1, 3, device="cuda")
    with torch.no_grad():
        target = soma.pose(
            pose,
            transl=transl,
            pose2rot=False,
            absolute_pose=False,
            apply_correctives=False,
        )["vertices"]

    # This test validates topology selection, not minimum-iteration convergence.
    # XLO has sparse support, so its inverse fit varies more across GPU runtimes
    # than the denser mid/low LOD tests below.
    xlo_mean_error_limit = 0.02
    result = inv.fit(target)
    assert result["per_vertex_error"].shape == (1, soma.bind_shape.shape[0])
    mean_err = result["per_vertex_error"].mean().item()
    assert mean_err < xlo_mean_error_limit, f"XLO topology inversion error too high: {mean_err:.6f}"


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_procedural_pose_inversion_uses_public_view_without_public_layer_clone():
    from soma.body import SOMALayer
    from soma.fitting.pose_inversion import PoseInversion

    soma = SOMALayer(
        data_root=str(ASSETS_DIR),
        identity_model_type="soma",
        device="cpu",
        mode="dense",
        lod="low",
        enable_procedural_transforms=True,
    )
    inv = PoseInversion(soma, low_lod=False)

    assert inv.soma is soma
    assert inv._autograd_soma is soma

    identity_coeffs = torch.zeros(1, soma.identity_model.num_identity_coeffs)
    inv.prepare_identity(identity_coeffs, repose_to_bind_pose=False)

    assert inv.joint_names == list(soma.public_joint_names)
    assert inv._cache["parent_ids"].shape[0] == len(soma.public_joint_names)
    assert inv._cache["skinning_weights"].shape[1] == len(soma.public_joint_names)
    assert inv._cache["bone_indices"].max().item() < len(soma.public_joint_names)


@pytest.fixture(scope="module")
def soma_and_inv():
    """Create SOMALayer + PoseInversion, prepare mean-shape identity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not ASSETS_DIR.is_dir():
        pytest.fail(f"Assets directory not found: {ASSETS_DIR}")
    if not MOTION_FILE.is_file():
        pytest.fail(f"Motion file not found: {MOTION_FILE}")

    from soma.body import SOMALayer
    from soma.fitting.pose_inversion import PoseInversion

    device = "cuda"
    soma = SOMALayer(
        data_root=str(ASSETS_DIR),
        identity_model_type="soma",
        device=device,
        mode="warp",
        low_lod=True,
    )

    # Prepare mean shape
    n_id = soma.identity_model.num_identity_coeffs
    identity_coeffs = torch.zeros(1, n_id, device=device)
    soma.prepare_identity(identity_coeffs)

    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(identity_coeffs)

    return soma, inv


@pytest.fixture(scope="module")
def soma_and_inv_no_procedural():
    """Create legacy public-rig SOMALayer + PoseInversion."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not ASSETS_DIR.is_dir():
        pytest.fail(f"Assets directory not found: {ASSETS_DIR}")
    if not MOTION_FILE.is_file():
        pytest.fail(f"Motion file not found: {MOTION_FILE}")

    from soma.body import SOMALayer
    from soma.fitting.pose_inversion import PoseInversion

    device = "cuda"
    soma = SOMALayer(
        data_root=str(ASSETS_DIR),
        identity_model_type="soma",
        device=device,
        mode="warp",
        low_lod=True,
        enable_procedural_transforms=False,
    )

    n_id = soma.identity_model.num_identity_coeffs
    identity_coeffs = torch.zeros(1, n_id, device=device)
    soma.prepare_identity(identity_coeffs)

    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(identity_coeffs)

    return soma, inv


@requires_cuda
class TestInvert:
    """Tests for PoseInversion.fit() default solver."""

    def test_single_frame_roundtrip(self, soma_and_inv):
        """Single frame: fit recovers pose with low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(verts)

        J = result["rotations"].shape[1]  # 78 (root + 77 joints)
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1

        expected_root = torch.eye(
            3,
            device=result["rotations"].device,
            dtype=result["rotations"].dtype,
        ).unsqueeze(0)
        torch.testing.assert_close(result["rotations"][:, 0], expected_root, rtol=0, atol=1e-6)

        mean_err = result["per_vertex_error"].mean().item()
        max_err = result["per_vertex_error"].max().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"
        assert max_err < 0.05, f"Max vertex error too high: {max_err:.6f} m"

    def test_single_frame_roundtrip_without_procedural_transforms(
        self,
        soma_and_inv_no_procedural,
    ):
        """Default solver should also fit the non-procedural public rig."""
        soma, inv = soma_and_inv_no_procedural
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(verts)

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1

        mean_err = result["per_vertex_error"].mean().item()
        max_err = result["per_vertex_error"].max().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"
        assert max_err < 0.05, f"Max vertex error too high: {max_err:.6f} m"

    def test_batch_roundtrip(self, soma_and_inv):
        """Multiple diverse frames: consistent low error across batch."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result = inv.fit(verts)

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (4, J, 3, 3)
        assert result["per_vertex_error"].shape[0] == 4

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_roundtrip_forward_pass(self, soma_and_inv):
        """Verify inverted rotations reproduce vertices via soma.pose().

        fit returns absolute local rotations for 78 joints
        (root + 77).  Strip the root (index 0) and pass to
        soma.pose(absolute_pose=True) to reconstruct.
        """
        soma, inv = soma_and_inv
        verts_gt, _ = _load_motion(soma, frames=[50, 200])

        result = inv.fit(verts_gt)

        # Strip root joint (index 0) — soma.pose() expects 77 joints
        rotations_no_root = result["rotations"][:, 1:]
        # fit uses raw LBS without correctives, so disable
        # correctives in the forward pass for a fair comparison.
        with torch.no_grad():
            out = soma.pose(
                rotations_no_root,
                transl=result["root_translation"],
                pose2rot=False,
                absolute_pose=True,
                apply_correctives=False,
            )
        verts_recon = out["vertices"]

        err = torch.norm(verts_recon - verts_gt, dim=-1)
        mean_err = err.mean().item()
        # Slightly higher threshold than internal per_vertex_error because
        # soma.pose() uses full skinning weights while fit
        # uses sparse top-K weights internally.
        assert mean_err < 0.02, f"Forward-pass roundtrip error too high: {mean_err:.6f} m"

    def test_batch_size_chunking(self, soma_and_inv):
        """batch_size parameter produces comparable results to all-at-once."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 50, 100, 150])

        result_all = inv.fit(verts)
        result_chunked = inv.fit(verts, batch_size=2)

        assert result_chunked["rotations"].shape == result_all["rotations"].shape

        # Analytical is deterministic, so results should be very close
        err_all = result_all["per_vertex_error"].mean().item()
        err_chunked = result_chunked["per_vertex_error"].mean().item()
        assert abs(err_all - err_chunked) < 0.005, (
            f"Chunked vs all-at-once error mismatch: {err_all:.6f} vs {err_chunked:.6f}"
        )

    def test_identity_pose_near_zero_error(self, soma_and_inv):
        """Rest pose (identity rotations) should fit with near-zero error."""
        soma, inv = soma_and_inv
        device = soma.device

        J = 77
        rot_mats = torch.eye(3, device=device).expand(1, J, 3, 3).clone()

        transl = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            out = soma.pose(
                rot_mats,
                transl=transl,
                pose2rot=False,
                apply_correctives=False,
            )
        verts = out["vertices"]

        result = inv.fit(verts)

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.02, f"Identity pose error too high: {mean_err:.6f} m"


@requires_cuda
class TestInvertAutogradFK:
    """Tests for PoseInversion.fit(autograd_iters=...)."""

    def test_single_frame_roundtrip(self, soma_and_inv):
        """Single frame: fit(autograd_iters) recovers pose with low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(
            verts,
            body_iters=0,
            full_iters=0,
            lie_iters=0,
            autograd_iters=20,
            autograd_lr=5e-3,
        )

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1
        assert result["local_rotation_drift"].shape == (1, J)
        assert result["root_translation_drift"].shape == (1,)

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_batch_roundtrip(self, soma_and_inv):
        """Multiple diverse frames: consistent low error across batch."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result = inv.fit(
            verts,
            body_iters=0,
            full_iters=0,
            lie_iters=0,
            autograd_iters=20,
            autograd_lr=5e-3,
        )

        assert result["rotations"].shape[0] == 4
        assert result["local_rotation_drift"].shape[0] == 4
        assert result["root_translation_drift"].shape == (4,)
        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_heel_weight_targets_rear_foot_vertices(self, soma_and_inv):
        """Heel weights should select a smaller rear-foot subset."""
        soma, inv = soma_and_inv

        from soma.fitting.pose_inversion import (
            _bind_joint_positions_from_cache,
            _normalized_vertex_weights,
        )

        cache = inv._cache
        bind_shape = soma._cached_rest_shape.detach()
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=bind_shape.dtype,
            device=bind_shape.device,
        )
        foot_weights = _normalized_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            {"feet": 10.0},
            dtype=torch.float32,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )
        heel_weights = _normalized_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            {"heels": 10.0},
            dtype=torch.float32,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )
        assert foot_weights is not None
        assert heel_weights is not None

        foot_mask = foot_weights > 1.0
        heel_mask = heel_weights > 1.0
        assert heel_mask.any()
        assert torch.all(foot_mask[heel_mask])
        assert heel_mask.sum() < foot_mask.sum()


@requires_cuda
class TestLieAlgebraGN:
    """Tests for PoseInversion.fit(lie_iters=...)."""

    def test_single_frame_roundtrip(self, soma_and_inv):
        """Single frame: standalone Lie-GN recovers pose with low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(verts, body_iters=0, full_iters=0, lie_iters=5)

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1

        expected_root = torch.eye(
            3,
            device=result["rotations"].device,
            dtype=result["rotations"].dtype,
        ).unsqueeze(0)
        torch.testing.assert_close(result["rotations"][:, 0], expected_root, rtol=0, atol=1e-6)

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_batch_roundtrip(self, soma_and_inv):
        """Multiple diverse frames: Lie-GN gives consistent low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result = inv.fit(verts, body_iters=0, full_iters=0, lie_iters=5)

        assert result["rotations"].shape[0] == 4
        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_heel_weight_changes_lie_gn_objective(self, soma_and_inv):
        """Lie-GN should honor heel weights, not only the analytical warm start."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        from soma.fitting.pose_inversion import (
            _bind_joint_positions_from_cache,
            _heel_vertex_ids,
        )

        cache = inv._cache
        bind_shape = soma._cached_rest_shape.detach()
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=bind_shape.dtype,
            device=bind_shape.device,
        )
        heel_ids = _heel_vertex_ids(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            bind_shape,
            bind_joint_positions,
        )
        assert heel_ids

        target = verts.clone()
        target[:, heel_ids] += torch.tensor([0.0, 0.0, 0.03], device=target.device)

        unweighted = inv.fit(target, body_iters=0, full_iters=0, lie_iters=3)
        weighted = inv.fit(
            target,
            body_iters=0,
            full_iters=0,
            lie_iters=3,
            leaf_weight={"heels": 50.0},
        )

        unweighted_heel_err = unweighted["per_vertex_error"][:, heel_ids].mean()
        weighted_heel_err = weighted["per_vertex_error"][:, heel_ids].mean()
        assert weighted_heel_err < 0.5 * unweighted_heel_err

    def test_lie_gn_after_analytical(self, soma_and_inv):
        """Lie-GN warm-started by analytical solver equals or beats analytical alone."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result_analytical = inv.fit(verts, body_iters=2, full_iters=1, lie_iters=0)
        result_combined = inv.fit(verts, body_iters=2, full_iters=1, lie_iters=3)

        J = result_combined["rotations"].shape[1]
        assert result_combined["rotations"].shape == (4, J, 3, 3)

        mean_err = result_combined["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

        # Combined should not be significantly worse than analytical alone.
        # Lie-GN at near-convergence can shuffle small amounts of error, so allow
        # 1 mm of slack. The line search prevents divergence; this bound catches
        # real regressions.
        mean_err_analytical = result_analytical["per_vertex_error"].mean().item()
        assert mean_err <= mean_err_analytical + 1e-3, (
            f"Combined ({mean_err:.6f}) worse than analytical ({mean_err_analytical:.6f})"
        )
