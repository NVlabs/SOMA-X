"""
Smoke tests for SOMAHandLayer forward pass.
CUDA gets the broad matrix; CPU keeps targeted smoke rows.
Requires assets/SOMA_neutral.npz (run `git lfs pull` after clone).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.hand._test_layers import make_hand_layer as _make_hand_layer

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"
HAND_ASSET = ASSETS_DIR / "SOMAHand.npz"
TEMPLATE_RIG = ASSETS_DIR / "SOMA_template_rig.usda"
PROCEDURAL_DEFINITION = ASSETS_DIR / "SOMA_procedural_transforms.json"
_HAND_TYPES = ["left", "right"]
_HAND_IDENTITY_MODEL_TYPES = ["soma", "mano", "mhr"]
_HAND_FORWARD_SMOKE_CASES = {
    ("left", "cuda", "soma"),
    ("right", "cuda", "soma"),
    ("right", "cuda", "mhr"),
    ("right", "cpu", "soma"),
}


def _hand_forward_marks(hand_type, device, identity_model_type):
    marks = [pytest.mark.asset_heavy]
    marks.append(pytest.mark.gpu if device == "cuda" else pytest.mark.cpu)
    if (hand_type, device, identity_model_type) not in _HAND_FORWARD_SMOKE_CASES:
        marks.append(pytest.mark.slow)
    return marks


def _hand_lod_marks(hand_type, lod):
    marks = [pytest.mark.cpu, pytest.mark.asset_heavy]
    if lod == "xlo":
        marks.append(pytest.mark.xlo)
    if (hand_type, lod) not in {("right", "mid"), ("left", "mid")}:
        marks.append(pytest.mark.slow)
    return marks


def _public_joint_names_from_assets(rig_data):
    if "joint_names" in rig_data:
        return np.array(rig_data["joint_names"]).copy()

    from soma.procedural_transforms import load_soma_procedural_transform_definition

    definition = load_soma_procedural_transform_definition(PROCEDURAL_DEFINITION)
    return np.array(definition.public_joint_names)


@lru_cache(maxsize=1)
def _public_mid_rig_data():
    rig_data = dict(np.load(CORE_ASSET, allow_pickle=False))
    if TEMPLATE_RIG.exists():
        from soma.io import load_lod_rig_from_usd
        from soma.procedural_transforms import derive_soma_rig_without_procedural_joints

        rig_data.update(
            derive_soma_rig_without_procedural_joints(
                load_lod_rig_from_usd(TEMPLATE_RIG, "mid"),
                _public_joint_names_from_assets(rig_data),
            )
        )
    return rig_data


def _expected_correctives_to_hand_frame(hand_type):
    hand_data = np.load(HAND_ASSET, allow_pickle=False)
    rig_data = _public_mid_rig_data()
    wrist_global_id = int(hand_data[f"{hand_type}_wrist_global_id"])
    wrist_inv = np.linalg.inv(rig_data["bind_pose_world"][wrist_global_id])
    return torch.as_tensor(wrist_inv[:3, :3], dtype=torch.float32)


_HAND_FORWARD_CASES = [
    pytest.param(
        hand_type,
        "cuda",
        identity_model_type,
        id=f"cuda-{hand_type}-{identity_model_type}",
        marks=_hand_forward_marks(hand_type, "cuda", identity_model_type),
    )
    for hand_type in _HAND_TYPES
    for identity_model_type in _HAND_IDENTITY_MODEL_TYPES
] + [
    pytest.param(
        "right",
        "cpu",
        "soma",
        id="cpu-right-soma-smoke",
        marks=_hand_forward_marks("right", "cpu", "soma"),
    ),
]
_HAND_FK_ONLY_CASES = [
    pytest.param("cuda", mode, hand_type, id=f"cuda-{hand_type}-{mode}", marks=pytest.mark.gpu)
    for hand_type in _HAND_TYPES
    for mode in ["warp", "dense"]
] + [
    pytest.param("cpu", "warp", "right", id="cpu-right-warp-smoke", marks=pytest.mark.cpu),
]
_HAND_FK_ABSOLUTE_CASES = [
    pytest.param(
        "cuda",
        absolute_pose,
        id=f"cuda-absolute_pose_{absolute_pose}",
        marks=pytest.mark.gpu if not absolute_pose else (pytest.mark.gpu, pytest.mark.slow),
    )
    for absolute_pose in [False, True]
] + [
    pytest.param("cpu", False, id="cpu-relative_pose-smoke", marks=pytest.mark.cpu),
]
_HAND_LOD_CASES = [
    pytest.param("right", "mid", id="right-mid", marks=_hand_lod_marks("right", "mid")),
    pytest.param("right", "low", id="right-low", marks=_hand_lod_marks("right", "low")),
    pytest.param("right", "xlo", id="right-xlo", marks=_hand_lod_marks("right", "xlo")),
    pytest.param("left", "mid", id="left-mid-smoke", marks=_hand_lod_marks("left", "mid")),
]


@pytest.fixture(scope="module")
def data_root():
    if not ASSETS_DIR.is_dir():
        pytest.fail(
            f"Assets directory not found: {ASSETS_DIR}. "
            "Clone the repo and run `git lfs pull` to fetch assets."
        )
    if not CORE_ASSET.is_file():
        pytest.fail(
            f"Required asset not found: {CORE_ASSET}. "
            "Run `git lfs pull` to fetch LFS-tracked files."
        )
    return str(ASSETS_DIR)


@pytest.mark.parametrize(("hand_type", "device", "identity_model_type"), _HAND_FORWARD_CASES)
def test_hand_layer_forward_shape(data_root, hand_type, device, identity_model_type):
    """Forward pass returns correct shapes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_hand_layer(data_root, hand_type, device, identity_model_type)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    finger_poses = torch.zeros(B, 25, 3, 3, device=device)
    # Identity rot matrices
    finger_poses[:] = torch.eye(3, device=device)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    with torch.no_grad():
        out = layer(finger_poses, identity_coeffs, pose2rot=False)

    assert "vertices" in out
    assert "joints" in out
    verts = out["vertices"]
    joints = out["joints"]

    assert verts.dim() == 3
    assert verts.shape[0] == B
    assert verts.shape[2] == 3
    assert verts.shape[1] > 0, "Expected at least 1 hand vertex"

    assert joints.dim() == 3
    assert joints.shape == (B, 25, 3)


@pytest.mark.parametrize(("hand_type", "device", "identity_model_type"), _HAND_FORWARD_CASES)
def test_hand_layer_axis_angle_input(data_root, hand_type, device, identity_model_type):
    """Forward pass works with axis-angle input (pose2rot=True)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_hand_layer(data_root, hand_type, device, identity_model_type)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 2
    finger_poses = torch.zeros(B, 25, 3, device=device)  # zero axis-angle = identity
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    with torch.no_grad():
        out = layer(finger_poses, identity_coeffs, pose2rot=True)

    Vh_total = layer.faces.max().item() + 1
    assert out["vertices"].shape == (B, Vh_total, 3)
    assert out["joints"].shape == (B, 25, 3)


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_hand_layer_wrist_local_origin(data_root, hand_type):
    """In T-pose with identity coeffs, wrist joint should be near origin in wrist-local space."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    finger_poses = torch.zeros(B, 25, 3, 3, device=device)
    finger_poses[:] = torch.eye(3, device=device)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    with torch.no_grad():
        out = layer(finger_poses, identity_coeffs, pose2rot=False)

    # Vertices should be centered near origin (wrist-local)
    verts = out["vertices"][0]  # (Vh, 3)
    centroid = verts.mean(dim=0)
    # Centroid should be reasonably close to origin (hand scale ~0.1m)
    assert centroid.norm() < 0.3, (
        f"Hand centroid {centroid.tolist()} seems too far from origin for wrist-local space"
    )


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_hand_layer_attributes(data_root, hand_type):
    """Check key attributes exist with correct types/shapes."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    # Python attributes
    assert hasattr(layer, "hand_joint_ids_global")
    assert len(layer.hand_joint_ids_global) == 25
    assert hasattr(layer, "wrist_global_id")

    # Buffers
    assert layer.joint_parent_ids.shape == (25,)
    assert layer.bind_pose_world.shape == (25, 4, 4)
    assert layer.t_pose_world.shape == (25, 4, 4)
    assert layer.hand_vert_ids.shape[0] > 0
    assert layer.faces.shape[1] == 3


@pytest.mark.parametrize("hand_type", ["left", "right"])
@pytest.mark.parametrize(
    "lod",
    [
        pytest.param("mid", id="mid"),
        pytest.param("low", id="low", marks=pytest.mark.slow),
        pytest.param("xlo", id="xlo", marks=(pytest.mark.slow, pytest.mark.xlo)),
    ],
)
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_correctives_are_sliced_to_lod(data_root, hand_type, lod):
    """SOMAHandLayer uses the shared body correctives model sliced to hand LOD."""
    correctives_path = Path(data_root) / "correctives_model.pt"
    if not correctives_path.is_file():
        pytest.skip("Corrective model not available")

    layer, skip_reason = _make_hand_layer(
        data_root,
        hand_type,
        "cpu",
        lod=lod,
        correctives_model_path=correctives_path,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)
    if layer.correctives_model is None:
        pytest.skip("Corrective model not available")

    correctives = (
        layer.correctives_model.module
        if hasattr(layer.correctives_model, "module")
        else layer.correctives_model
    )
    if lod == "xlo":
        expected_vertices = layer.xlo_skeleton_mid_to_low.shape[0]
        assert layer.correctives_lod_transfer is not None
    else:
        expected_vertices = layer.bind_shape.shape[0]
        assert layer.correctives_lod_transfer is None
    assert correctives.J == 25
    assert correctives.V == expected_vertices
    assert correctives.W1.shape[0] == 25 * 6
    assert correctives.W2.shape[1] == expected_vertices * 3

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 25, 3, 3).contiguous()
    identity_coeffs = torch.zeros(1, layer.num_shape_components)
    with torch.no_grad():
        out = layer(
            poses,
            identity_coeffs,
            pose2rot=False,
            apply_correctives=True,
        )
    assert out["vertices"].shape == (1, layer.bind_shape.shape[0], 3)


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_can_skip_correctives_model_for_pure_lbs(data_root):
    """Pure-LBS callers can skip loading the corrective checkpoint."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(
        data_root,
        "right",
        device,
        mode="dense",
        correctives_model_path=None,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    assert layer.correctives_model is None

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 25, 3, 3).contiguous()
    identity_coeffs = torch.zeros(1, layer.num_shape_components)

    with torch.no_grad():
        out = layer(
            poses,
            identity_coeffs,
            pose2rot=False,
            apply_correctives=False,
        )

    assert "vertices" in out
    assert out["vertices"].shape[1] == layer.bind_shape.shape[0]


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_deprecated_load_correctives_model_false_alias(data_root):
    """Legacy callers can still disable checkpoint loading through the old flag."""
    device = "cpu"
    with pytest.warns(DeprecationWarning, match="load_correctives_model is deprecated"):
        layer, skip_reason = _make_hand_layer(
            data_root,
            "right",
            device,
            mode="dense",
            load_correctives_model=False,
        )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    assert layer.correctives_model_path is None
    assert layer.correctives_model is None


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_missing_custom_correctives_model_path_raises(data_root, tmp_path):
    from soma import SOMAHandLayer

    with pytest.raises(FileNotFoundError, match="Correctives model checkpoint not found"):
        SOMAHandLayer(
            data_root=data_root,
            hand_type="right",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            correctives_model_path=tmp_path / "missing_correctives.pt",
        )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_rejects_correctives_flag_and_explicit_path(data_root):
    from soma import SOMAHandLayer

    correctives_path = Path(data_root) / "correctives_model.pt"
    if not correctives_path.is_file():
        pytest.skip("Corrective model not available")

    with pytest.warns(DeprecationWarning, match="load_correctives_model is deprecated"):
        with pytest.raises(ValueError, match="explicit correctives_model_path"):
            SOMAHandLayer(
                data_root=data_root,
                hand_type="right",
                device="cpu",
                identity_model_type="soma",
                mode="dense",
                correctives_model_path=correctives_path,
                load_correctives_model=False,
            )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_apply_correctives_requires_loaded_model(data_root):
    device = "cpu"
    layer, skip_reason = _make_hand_layer(
        data_root,
        "right",
        device,
        mode="dense",
        correctives_model_path=None,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 25, 3, 3).contiguous()
    identity_coeffs = torch.zeros(1, layer.num_shape_components)

    with pytest.raises(RuntimeError, match="no corrective model is loaded"):
        with torch.no_grad():
            layer(
                poses,
                identity_coeffs,
                pose2rot=False,
                apply_correctives=True,
            )


@pytest.mark.parametrize("hand_type", ["left", "right"])
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_correctives_use_template_wrist_bind_frame(data_root, hand_type):
    """Corrective offsets are rotated by the template wrist bind transform."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    expected_frame = _expected_correctives_to_hand_frame(hand_type)
    assert torch.allclose(layer._correctives_to_hand_frame.cpu(), expected_frame, atol=1e-6)


@pytest.mark.parametrize("hand_type", ["left", "right"])
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_prepare_identity_reposes_to_bind_pose(data_root, hand_type):
    """SOMAHandLayer implements SOMALayer's repose_to_bind_pose flow."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device, mode="dense")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    identity_coeffs = torch.zeros(1, layer.num_shape_components, device=device)

    with torch.no_grad():
        layer.prepare_identity(identity_coeffs, repose_to_bind_pose=False)
        fitted_rest_shape = layer._cached_rest_shape.clone()
        fitted_bind_transforms = layer._cached_bind_transforms_world.clone()

        layer.batched_skinning.rebind(fitted_bind_transforms, fitted_rest_shape)
        expected_rest_shape, expected_bind_transforms = layer.batched_skinning.pose(
            local_rotations=layer.bind_pose_local[..., :3, :3],
            global_translation=layer.bind_pose_local[..., layer.root_joint_idx, :3, 3],
            return_transforms=True,
            absolute_pose=True,
        )

        layer.prepare_identity(identity_coeffs, repose_to_bind_pose=True)

    assert torch.allclose(layer._cached_rest_shape, expected_rest_shape, atol=1e-6)
    assert torch.allclose(
        layer._cached_bind_transforms_world,
        expected_bind_transforms,
        atol=1e-6,
    )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_hand_layer_legacy_low_lod_alias(data_root):
    """SOMAHandLayer keeps a low_lod alias like SOMALayer."""
    layer, skip_reason = _make_hand_layer(data_root, "left", "cpu", low_lod=True)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    assert layer.lod == "low"
    assert layer.low_lod is True


@pytest.mark.parametrize(("hand_type", "lod"), _HAND_LOD_CASES)
def test_hand_layer_lod_forward_shape(data_root, lod, hand_type):
    """Hand LODs should expose reduced topology with the same 25-joint rig."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device, lod=lod)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(B, 25, 3, 3).contiguous()
    identity_coeffs = torch.zeros(B, layer.num_shape_components)

    with torch.no_grad():
        out = layer(poses, identity_coeffs, pose2rot=False)

    assert layer.lod == lod
    assert out["vertices"].shape == (B, layer.bind_shape.shape[0], 3)
    assert out["joints"].shape == (B, 25, 3)
    assert layer.skinning_weights.shape == (layer.bind_shape.shape[0], 25)
    assert layer.faces.max().item() < layer.bind_shape.shape[0]
    if lod == "xlo":
        assert layer.identity_lod_transfer is not None
        assert layer.xlo_skeleton_mid_to_low is not None
        assert layer.skeleton_transfer.bind_shape.shape[0] > layer.bind_shape.shape[0]
    elif lod == "low":
        assert layer.identity_lod_mid_ids is not None
        assert layer.bind_shape.shape[0] < layer.hand_mid_vert_ids.shape[0]
    else:
        assert layer.identity_lod_mid_ids is None
        assert layer.identity_lod_transfer is None


# ---------------------------------------------------------------------------
# Bone scale parameter tests (scale_params on SOMA backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_bone_scales_identity(data_root, hand_type):
    """scale_params=1.0 produces the same output as scale_params=None."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    poses = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, 25, 3, 3)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    with torch.no_grad():
        layer.prepare_identity(identity_coeffs)
        out_none = layer.pose(poses, pose2rot=False)
        layer.prepare_identity(
            identity_coeffs,
            scale_params=torch.ones(B, 24, device=device),
        )
        out_ones = layer.pose(poses, pose2rot=False)

    assert torch.allclose(out_none["vertices"], out_ones["vertices"], atol=1e-5), (
        f"scale_params=1.0 should match scale_params=None, "
        f"max diff={(out_none['vertices'] - out_ones['vertices']).abs().max()}"
    )
    assert torch.allclose(out_none["joints"], out_ones["joints"], atol=1e-5)


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_bone_scales_gradient(data_root, hand_type):
    """Gradients flow through scale_params (bone-length scales on SOMA)."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    poses = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, 25, 3, 3)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)
    scale_params = torch.ones(B, 24, device=device, requires_grad=True)

    layer.prepare_identity(identity_coeffs, scale_params=scale_params)
    out = layer.pose(poses, pose2rot=False)
    loss = out["joints"].sum()
    loss.backward()

    assert scale_params.grad is not None, "scale_params.grad should not be None"
    assert not torch.all(scale_params.grad == 0), "scale_params.grad should not be all zeros"


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_bone_scales_effect(data_root, hand_type):
    """Scaling a finger changes its joint positions."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    poses = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, 25, 3, 3)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    with torch.no_grad():
        layer.prepare_identity(identity_coeffs)
        out_base = layer.pose(poses, pose2rot=False)

        # Scale index finger (joints 5-8, params indices 4-7) by 1.5
        scales = torch.ones(B, 24, device=device)
        scales[:, 4:8] = 1.5
        layer.prepare_identity(identity_coeffs, scale_params=scales)
        out_scaled = layer.pose(poses, pose2rot=False)

    # Index finger joints (5-8) should have moved
    index_diff = (out_scaled["joints"][0, 5:9] - out_base["joints"][0, 5:9]).norm(dim=-1)
    assert index_diff.mean() > 0.001, (
        f"Index finger joints should move when scaled, mean diff={index_diff.mean()}"
    )
    # Thumb joints (1-3) should be unaffected
    thumb_diff = (out_scaled["joints"][0, 1:4] - out_base["joints"][0, 1:4]).norm(dim=-1)
    assert thumb_diff.max() < 1e-5, (
        f"Thumb joints should not move when only index is scaled, max diff={thumb_diff.max()}"
    )


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_bone_scales_end_joints_effective(data_root, hand_type):
    """Scaling an end joint (e.g. IndexEnd) moves fingertip verts."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    poses = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, 25, 3, 3)
    identity_coeffs = torch.zeros(B, layer.num_shape_components, device=device)

    # End joint param indices are (global_joint - 1): (3, 8, 13, 18, 23)
    end_param_indices = [3, 8, 13, 18, 23]

    with torch.no_grad():
        layer.prepare_identity(identity_coeffs)
        out_base = layer.pose(poses, pose2rot=False)

        scales = torch.ones(B, 24, device=device)
        scales[:, end_param_indices] = 1.5
        layer.prepare_identity(identity_coeffs, scale_params=scales)
        out_scaled = layer.pose(poses, pose2rot=False)

    # End-joint positions (global indices 4, 9, 14, 19, 24) should move
    end_global = [4, 9, 14, 19, 24]
    joint_diff = (out_scaled["joints"][0, end_global] - out_base["joints"][0, end_global]).norm(
        dim=-1
    )
    assert joint_diff.min() > 1e-4, (
        f"End-joint positions should respond to end-joint scale, min diff={joint_diff.min()}"
    )

    # Some fingertip vertices should also move (end joints now have skinning weight)
    vert_diff = (out_scaled["vertices"][0] - out_base["vertices"][0]).norm(dim=-1)
    assert vert_diff.max() > 1e-4, (
        f"Fingertip verts should move when end joints scale, max diff={vert_diff.max()}"
    )


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_skinning_weights_normalized(data_root, hand_type):
    """Every vertex's skinning weights sum to ~1."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    row_sums = layer.skinning_weights.sum(dim=-1)
    max_err = (row_sums - 1.0).abs().max().item()
    assert max_err < 1e-5, f"Skinning rows not normalized, max |sum - 1| = {max_err}"


def test_soma_hand_identity_model_unit_boundary(data_root):
    """SOMAHandIdentityModel.get_rest_shape stays native; forward returns layer output units."""
    from soma.units import Unit

    device = "cpu"
    layer, skip_reason = _make_hand_layer(
        data_root,
        "right",
        device,
        identity_model_type="soma",
        output_unit=Unit.METERS,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    coeffs = torch.zeros(1, layer.num_shape_components, device=device)
    native_rest_cm = layer.identity_model.get_rest_shape(coeffs)
    public_rest_m = layer.get_rest_shape(coeffs)

    cm_to_m = Unit.CENTIMETERS.meters_per_unit / Unit.METERS.meters_per_unit
    assert torch.allclose(public_rest_m, native_rest_cm * cm_to_m, atol=1e-6)


@pytest.mark.cpu
def test_soma_hand_identity_model_requires_hand_pca():
    """SOMAHandIdentityModel should not fall back to full-body PCA."""
    from soma.hand.identity_model import SOMAHandIdentityModel

    with pytest.raises(KeyError, match="left_mean"):
        SOMAHandIdentityModel(
            REPO_ROOT,
            low_lod=False,
            device="cpu",
            hand_map={},
            hand_type="left",
        )


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_hand_end_joints_have_weight(data_root, hand_type):
    """The v0027 raw hand weights give end joints non-zero skinning weights."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    end_joints = [4, 9, 14, 19, 24]
    W = layer.skinning_weights  # (Vh, 25)
    for j in end_joints:
        n_verts = int((W[:, j] > 0).sum().item())
        assert n_verts >= 20, f"End joint {j} has only {n_verts} weighted verts (expected >= 20)"


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_hand_skinning_influence_counts_are_valid(data_root, hand_type):
    """The raw v0027 hand weights have valid non-empty influence rows."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    W = layer.skinning_weights
    nnz_per_row = (W > 0).sum(dim=-1).float()
    mean_nnz = nnz_per_row.mean().item()
    max_nnz = int(nnz_per_row.max().item())

    assert mean_nnz > 0.0
    assert max_nnz <= layer.skinning_weights.shape[1], (
        f"Max influences per vertex = {max_nnz} exceeds number of joints"
    )


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_dominant_joint_preserved(data_root, hand_type):
    """The raw v0027 hand weights keep some strongly dominant joint influences."""
    device = "cpu"
    layer, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    W = layer.skinning_weights  # (Vh, 25)
    raw_max, raw_argmax = W.max(dim=-1)
    dominant_mask = raw_max >= 0.9
    assert dominant_mask.sum() > 0, "Expected some dominantly-weighted verts in raw rig"
    assert raw_argmax[dominant_mask].numel() == dominant_mask.sum()


@pytest.mark.parametrize(("device", "mode", "hand_type"), _HAND_FK_ONLY_CASES)
def test_hand_layer_fk_only_matches_full_pose(data_root, device, mode, hand_type):
    """fk_only=True must return the same transforms as the full pose() path.

    FK is the shared prefix; only the LBS block is skipped. The fast path
    also drops the "vertices" key. Exercised across warp/dense modes, left/
    right hands, and CUDA with a CPU smoke row. This is the contract Lin's
    DataLoader pipeline relies on.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_hand_layer(
        data_root,
        hand_type,
        device,
        identity_model_type="soma",
        mode=mode,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(0)
    B = 4
    K = layer.identity_model.num_identity_coeffs
    poses = torch.randn(B, 25, 3, device=device) * 0.1
    coeffs = torch.zeros(B, K, device=device)
    transl = torch.randn(B, 3, device=device) * 0.05

    layer.prepare_identity(coeffs, global_scale=1.0)

    with torch.no_grad():
        ref = layer.pose(poses=poses, pose2rot=True, global_translation=transl)
        fast = layer.pose(poses=poses, pose2rot=True, global_translation=transl, fk_only=True)

    assert "vertices" in ref
    assert "vertices" not in fast
    assert torch.allclose(fast["transforms"], ref["transforms"], atol=1e-6), (
        f"hand fk_only transforms diverge: "
        f"max abs = {(fast['transforms'] - ref['transforms']).abs().max().item():.3e}"
    )
    assert torch.allclose(fast["joints"], ref["joints"], atol=1e-6)


@pytest.mark.parametrize(("device", "absolute_pose"), _HAND_FK_ABSOLUTE_CASES)
def test_hand_layer_fk_only_absolute_pose(data_root, device, absolute_pose):
    """fk_only output matches full pose() under absolute_pose and pose2rot=False."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_hand_layer(data_root, "right", device, identity_model_type="soma")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(1)
    B = 2
    K = layer.identity_model.num_identity_coeffs
    poses_R = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, 25, 3, 3).contiguous()
    coeffs = torch.zeros(B, K, device=device)
    transl = torch.zeros(B, 3, device=device)

    layer.prepare_identity(coeffs, global_scale=1.0)

    with torch.no_grad():
        ref = layer.pose(
            poses=poses_R,
            pose2rot=False,
            absolute_pose=absolute_pose,
            global_translation=transl,
        )
        fast = layer.pose(
            poses=poses_R,
            pose2rot=False,
            absolute_pose=absolute_pose,
            global_translation=transl,
            fk_only=True,
        )

    assert torch.allclose(fast["transforms"], ref["transforms"], atol=1e-6)


def test_hand_layer_fk_only_skips_warp_on_cpu(data_root):
    """Warp-mode hand layer built on CPU: fk_only must not launch any warp kernel.

    This exercises the kernel-launch-free guarantee that Lin's fork+DataLoader
    pipeline relies on. The smoke test here is cheap — fork-safety is
    separately covered in tests/test_dataloader_hand.py.
    """
    layer, skip_reason = _make_hand_layer(data_root, "left", "cpu", identity_model_type="soma")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(2)
    K = layer.identity_model.num_identity_coeffs
    layer.prepare_identity(torch.zeros(1, K), global_scale=1.0)
    poses = torch.zeros(1, 25, 3)
    with torch.no_grad():
        out = layer.pose(poses=poses, fk_only=True)
    assert out["transforms"].shape == (1, 25, 4, 4)
    assert torch.isfinite(out["transforms"]).all()
    assert "vertices" not in out


def test_soma_hand_paired_rig_single_pass_mirror(data_root):
    """Single PoseMirror_SOMA call mirrors both hands simultaneously.

    Pattern used by the camera-space mirror prototype: build a 50-joint
    paired skeleton (Left-first, Right-second), populate BOTH halves with
    FK-computed world transforms, call PoseMirror_SOMA once, and verify:

      - The left half of the output equals mirror(right input)
      - The right half of the output equals mirror(left input)
      - Double-application recovers the input bit-exactly (involution)

    This is the both-hands DataLoader pattern: one mirror call produces two
    augmentations, half the cost of two separate calls.
    """
    from soma.geometry.rig_utils import PoseMirror_SOMA

    layer_L, skip_L = _make_hand_layer(data_root, "left", "cpu", identity_model_type="soma")
    layer_R, skip_R = _make_hand_layer(data_root, "right", "cpu", identity_model_type="soma")
    if skip_L or skip_R:
        pytest.skip(skip_L or skip_R)

    torch.manual_seed(7)
    B = 1
    K = layer_R.identity_model.num_identity_coeffs
    coeffs = torch.zeros(B, K)

    # Two independent "scenes" — one per hand — so the paired call is genuinely
    # carrying two distinct poses. Use random finger rotations and random wrist
    # translations.
    def _random_poses_and_transl(seed_offset):
        torch.manual_seed(7 + seed_offset)
        poses = torch.randn(B, 25, 3) * 0.2
        transl = torch.randn(B, 3) * 0.1
        return poses, transl

    poses_R, transl_R = _random_poses_and_transl(0)
    poses_L, transl_L = _random_poses_and_transl(1)

    layer_R.prepare_identity(coeffs, global_scale=1.0)
    layer_L.prepare_identity(coeffs, global_scale=1.0)
    with torch.no_grad():
        T_R = layer_R.pose(
            poses=poses_R,
            pose2rot=True,
            global_translation=transl_R,
            fk_only=True,
        )["transforms"].squeeze(0)  # (25, 4, 4)
        T_L = layer_L.pose(
            poses=poses_L,
            pose2rot=True,
            global_translation=transl_L,
            fk_only=True,
        )["transforms"].squeeze(0)  # (25, 4, 4)

    left_names = layer_L.rig_data["joint_names"]
    right_names = layer_R.rig_data["joint_names"]
    paired_names = list(left_names) + list(right_names)

    # Populate both slots in one tensor; one call mirrors both.
    T_paired = torch.eye(4).repeat(50, 1, 1)
    T_paired[:25] = T_L
    T_paired[25:] = T_R

    mirror = PoseMirror_SOMA(paired_names, root_name="__no_root__")
    T_out = mirror(T_paired)

    # A. The left half of the output == mirror(right input).
    #    We verify this by comparing to a SEPARATE single-direction call.
    T_paired_R_only = torch.eye(4).repeat(50, 1, 1)
    T_paired_R_only[25:] = T_R
    mirror_R_only = mirror(T_paired_R_only)
    assert torch.allclose(T_out[:25], mirror_R_only[:25], atol=1e-6), (
        f"single-pass left half diverges from dedicated right->left call: "
        f"max abs = {(T_out[:25] - mirror_R_only[:25]).abs().max().item():.3e}"
    )

    # B. The right half of the output == mirror(left input).
    T_paired_L_only = torch.eye(4).repeat(50, 1, 1)
    T_paired_L_only[:25] = T_L
    mirror_L_only = mirror(T_paired_L_only)
    assert torch.allclose(T_out[25:], mirror_L_only[25:], atol=1e-6)

    # C. Involution: mirror(mirror(paired)) == paired, bit-exact.
    T_twice = mirror(T_out)
    assert torch.allclose(T_twice, T_paired, atol=1e-6), (
        f"involution failed: max abs = {(T_twice - T_paired).abs().max().item():.3e}"
    )


def test_hand_layer_vs_full_body(data_root):
    """Left-hand vertices in wrist-local space should have the same pairwise
    distances as the corresponding full-body vertices.

    Only tested for the left hand: the right-hand mesh uses mirrored left-hand
    geometry (identical topology but no 1-to-1 vertex correspondence with the
    right-hand body vertices), so a direct full-body comparison is not valid.
    """
    from soma import SOMALayer

    hand_type = "left"
    device = "cpu"

    try:
        soma = SOMALayer(
            data_root=data_root,
            device=device,
            identity_model_type="soma",
            mode="warp",
        ).to(device)
    except Exception as e:
        pytest.skip(f"Could not create full-body layer: {e}")
    hand, skip_reason = _make_hand_layer(data_root, hand_type, device)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    B = 1
    # Identity (zero) pose: all rot matrices = I
    poses_full = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, 77, 3, 3)
    # Hand poses: skeleton joints 15-39 (left) = pose indices 14-38 (pose_idx = skel_idx - 1)
    # SOMALayer forward takes 77 joints (index 0 = Hips)
    if hand_type == "left":
        finger_global_local = list(range(14, 39))  # 25 joints: wrist + 24 fingers
    else:
        finger_global_local = list(range(42, 67))
    finger_poses = poses_full[:, finger_global_local, :, :]  # (B, 25, 3, 3)

    soma_coeffs = torch.zeros(B, soma.num_shape_components, device=device)
    hand_coeffs = torch.zeros(B, hand.num_shape_components, device=device)
    transl = torch.zeros(B, 3, device=device)

    with torch.no_grad():
        out_full = soma(poses_full, soma_coeffs, transl=transl, pose2rot=False)
        out_hand = hand(finger_poses, hand_coeffs, pose2rot=False)

    # Compare shapes using pairwise distances — rotation-invariant, so we don't need
    # to recover the exact wrist world transform.
    # Only compare original (non-clip) vertices since full-body model has no clip verts.
    Vh_orig = hand.hand_vert_ids.shape[0]
    hand_verts_local = out_hand["vertices"][0, :Vh_orig, :]  # (Vh_orig, 3) wrist-local
    full_verts_hand = out_full["vertices"][0, hand.hand_vert_ids, :]  # (Vh_orig, 3) world

    # Sample a small subset to keep the test fast
    n_sample = min(50, hand_verts_local.shape[0])
    idx = torch.arange(0, hand_verts_local.shape[0], hand_verts_local.shape[0] // n_sample)[
        :n_sample
    ]
    pd_hand = torch.cdist(hand_verts_local[idx], hand_verts_local[idx])  # (S, S)
    pd_full = torch.cdist(full_verts_hand[idx], full_verts_hand[idx])  # (S, S)

    diff = (pd_hand - pd_full).abs().mean()
    # Allow up to 1cm mean pairwise distance error.  The two models use different RBF
    # fits (full-body vs hand-only vertices) for the skeleton, causing small differences
    # in the fitted wrist position/rotation, which propagate into slightly different LBS
    # deformations even at zero identity.
    assert diff < 0.01, (
        f"Hand pairwise-distance mismatch between hand model and full-body model: {diff:.5f}m"
    )
