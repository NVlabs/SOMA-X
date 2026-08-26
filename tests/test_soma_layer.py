# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests: SOMALayer forward pass for each identity model (soma, mhr, anny, smpl, garment)
as in tools/demo_soma_vis.py. CUDA gets the broad matrix; CPU keeps targeted smoke rows.
CUDA is skipped when unavailable. Fails if assets/SOMA_neutral.npz is not present
(e.g. run `git lfs pull` after clone). Optional models (smpl, anny) are skipped when
their assets or dependencies are missing.
"""

import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._optional_assets import body_identity_skip_reason

# Repo root = parent of tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"
TEMPLATE_RIG = ASSETS_DIR / "SOMA_template_rig.usda"
PROCEDURAL_DEFINITION = ASSETS_DIR / "SOMA_procedural_transforms.json"
_LAYER_CACHE_MAX_SIZE = 2
# Keep only adjacent no/correctives pairs so CUDA layers are not retained across the matrix.
_LAYER_CACHE = OrderedDict()
_BODY_IDENTITY_MODEL_TYPES = ["soma", "mhr", "anny", "smpl", "smplx", "garment"]
_BODY_LODS = ["mid", "low", "xlo"]
_BODY_CORRECTIVE_CASES = [
    (False, "no_correctives"),
    (True, "correctives"),
]
_BODY_FORWARD_SMOKE_CASES = {
    ("cuda", "soma", "mid", False),
    ("cuda", "soma", "mid", True),
    ("cuda", "soma", "low", False),
    ("cuda", "soma", "xlo", False),
    ("cuda", "mhr", "mid", False),
    ("cuda", "anny", "mid", False),
    ("cpu", "soma", "mid", False),
}


def _body_forward_marks(device, identity_model_type, lod, apply_correctives):
    marks = [pytest.mark.asset_heavy]
    marks.append(pytest.mark.gpu if device == "cuda" else pytest.mark.cpu)
    if lod == "xlo":
        marks.append(pytest.mark.xlo)
    if (device, identity_model_type, lod, apply_correctives) not in _BODY_FORWARD_SMOKE_CASES:
        marks.append(pytest.mark.slow)
    return marks


_SOMA_LAYER_FORWARD_CASES = [
    pytest.param(
        device,
        identity_model_type,
        lod,
        apply_correctives,
        id=f"{device}-{identity_model_type}-{lod}_lod-{corrective_id}",
        marks=_body_forward_marks(device, identity_model_type, lod, apply_correctives),
    )
    for device in ["cuda"]
    for identity_model_type in _BODY_IDENTITY_MODEL_TYPES
    for lod in _BODY_LODS
    for apply_correctives, corrective_id in _BODY_CORRECTIVE_CASES
] + [
    pytest.param(
        "cpu",
        "soma",
        "mid",
        False,
        id="cpu-soma-mid_lod-no_correctives-smoke",
        marks=_body_forward_marks("cpu", "soma", "mid", False),
    ),
    pytest.param(
        "cpu",
        "soma",
        "mid",
        True,
        id="cpu-soma-mid_lod-correctives-smoke",
        marks=_body_forward_marks("cpu", "soma", "mid", True),
    ),
]
_SOMA_FK_ONLY_CASES = [
    pytest.param("cuda", mode, id=f"cuda-{mode}", marks=pytest.mark.gpu)
    for mode in ["warp", "dense"]
] + [
    pytest.param("cpu", "warp", id="cpu-warp-smoke", marks=pytest.mark.cpu),
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
            "Run `git lfs pull` (or `git lfs pull assets/`) to fetch LFS-tracked files."
        )
    return str(ASSETS_DIR)


def _make_layer(data_root, identity_model_type, device, lod="mid"):
    """Create SOMALayer; return (layer, skip_reason). skip_reason is non-None to skip the test."""
    cache_key = (str(Path(data_root).resolve()), identity_model_type, str(device), lod)
    if cache_key in _LAYER_CACHE:
        _LAYER_CACHE.move_to_end(cache_key)
        return _LAYER_CACHE[cache_key]

    skip_reason = body_identity_skip_reason(data_root, identity_model_type, lod=lod)
    if skip_reason is not None:
        return _remember_layer(cache_key, (None, skip_reason))

    from soma import SOMALayer

    try:
        layer = SOMALayer(
            data_root=data_root,
            lod=lod,
            device=device,
            identity_model_type=identity_model_type,
            mode="warp",
        ).to(device)
        return _remember_layer(cache_key, (layer, None))
    except (FileNotFoundError, ImportError) as e:
        return _remember_layer(cache_key, (None, f"Missing asset or dependency: {e}"))


def _remember_layer(cache_key, value):
    _LAYER_CACHE[cache_key] = value
    _LAYER_CACHE.move_to_end(cache_key)
    while len(_LAYER_CACHE) > _LAYER_CACHE_MAX_SIZE:
        _LAYER_CACHE.popitem(last=False)
    return value


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _make_slim_data_root(tmp_path: Path, *, include_template: bool = True) -> Path:
    from soma.io import SOMA_NEUTRAL_RIG_KEYS

    data_root = tmp_path / "assets"
    data_root.mkdir()
    with np.load(CORE_ASSET, allow_pickle=False) as data:
        slim = {key: data[key] for key in data.files if key not in SOMA_NEUTRAL_RIG_KEYS}
    np.savez_compressed(data_root / CORE_ASSET.name, allow_pickle=False, **slim)
    _link_or_copy(PROCEDURAL_DEFINITION, data_root / PROCEDURAL_DEFINITION.name)
    if include_template:
        _link_or_copy(TEMPLATE_RIG, data_root / TEMPLATE_RIG.name)
    return data_root


def _make_inputs(layer, identity_model_type, device, batch_size=1):
    """Build identity_coeffs and scale_params for the given identity model type."""
    if identity_model_type == "anny":
        ann = layer.identity_model.identity_model
        identity_coeffs = {
            k: torch.ones(batch_size, device=device) * 0.5 for k in ann.phenotype_labels
        }
        scale_params = {k: torch.zeros(batch_size, device=device) for k in ann.local_change_labels}
    elif identity_model_type == "mhr":
        n_id = layer.identity_model.num_identity_coeffs
        n_scale = layer.identity_model.num_scale_params
        identity_coeffs = torch.zeros(batch_size, n_id, device=device)
        scale_params = torch.zeros(batch_size, n_scale, device=device)
    else:
        n_id = layer.identity_model.num_identity_coeffs
        identity_coeffs = torch.zeros(batch_size, n_id, device=device)
        scale_params = None
    return identity_coeffs, scale_params


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_legacy_low_lod_positional_device(data_root):
    """The new lod selector must not break legacy positional device calls."""
    from soma import SOMALayer

    try:
        layer = SOMALayer(
            data_root,
            True,
            "cpu",
            identity_model_type="mhr",
            mode="dense",
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")
    assert layer.low_lod is True
    assert layer.lod == "low"


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_loads_slim_npz_with_template_rig(tmp_path):
    from soma import SOMALayer

    data_root = _make_slim_data_root(tmp_path)
    layer = SOMALayer(
        data_root=data_root,
        lod="mid",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
    )

    assert layer.bind_shape.shape[0] == layer.shape_mean.shape[0]
    assert layer.procedural_transforms is not None
    assert len(layer.public_joint_names) == 78
    assert len(layer.rig_data["joint_names"]) > len(layer.public_joint_names)
    assert layer.skinning_weights.shape == (
        layer.bind_shape.shape[0],
        len(layer.rig_data["joint_names"]),
    )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_slim_npz_requires_template_rig(tmp_path):
    from soma import SOMALayer

    data_root = _make_slim_data_root(tmp_path, include_template=False)
    with pytest.raises(FileNotFoundError, match="template rig"):
        SOMALayer(
            data_root=data_root,
            lod="mid",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
        )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.xlo
@pytest.mark.asset_heavy
def test_soma_layer_xlo_uses_low_lod_skeleton_transfer(data_root):
    """XLO should fit identity skeletons on low LOD, not sparse xlo vertices."""
    from soma import SOMALayer

    try:
        xlo_layer = SOMALayer(
            data_root=data_root,
            lod="xlo",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")

    assert xlo_layer.xlo_skeleton_transfer is not None
    assert xlo_layer.xlo_skeleton_mid_to_low is not None
    assert xlo_layer.skeleton_transfer is xlo_layer.xlo_skeleton_transfer
    assert xlo_layer.bind_shape.shape[0] != xlo_layer.xlo_skeleton_transfer.bind_shape.shape[0]

    fit_call = {}
    original_xlo_fit = xlo_layer.xlo_skeleton_transfer.fit

    def record_xlo_skeleton_fit(rest_shape):
        fit_call["shape"] = rest_shape.shape
        return original_xlo_fit(rest_shape)

    xlo_layer.xlo_skeleton_transfer.fit = record_xlo_skeleton_fit

    identity_coeffs = torch.zeros(1, xlo_layer.identity_model.num_identity_coeffs)
    with torch.no_grad():
        xlo_layer.prepare_identity(identity_coeffs)

    assert fit_call["shape"][1] == xlo_layer.xlo_skeleton_mid_to_low.shape[0]


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.xlo
@pytest.mark.asset_heavy
def test_soma_layer_virtual_root_stays_pinned_for_anny(data_root):
    """Anny identity fitting must not leave an offset on the dummy Root joint."""
    layer, skip_reason = _make_layer(data_root, "anny", "cpu", lod="xlo")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    identity_coeffs, scale_params = _make_inputs(layer, "anny", "cpu", batch_size=1)
    with torch.no_grad():
        layer.prepare_identity(identity_coeffs, scale_params=scale_params)

    assert torch.allclose(
        layer._cached_bind_transforms_world[0, 0],
        torch.eye(4),
        atol=1e-6,
    )

    pose = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 77, 3, 3).contiguous()
    transl = torch.zeros(1, 3)
    with torch.no_grad():
        out = layer.pose(
            pose,
            transl=transl,
            pose2rot=False,
            absolute_pose=True,
            apply_correctives=False,
        )

    assert abs(out["joints"][0, 0, 1].item()) < 1e-6
    assert out["vertices"][0, :, 1].min().item() < 0.0


@pytest.mark.parametrize(
    ("device", "identity_model_type", "lod", "apply_correctives"),
    _SOMA_LAYER_FORWARD_CASES,
)
def test_soma_layer_forward(data_root, identity_model_type, device, lod, apply_correctives):
    """SOMALayer forward pass for each identity model, LOD, and corrective mode."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_layer(data_root, identity_model_type, device, lod=lod)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    if apply_correctives and layer.correctives_model is None:
        pytest.skip("Corrective model not available")

    if lod == "low":
        assert layer.nv_lod_mid_to_low is not None
        expected_num_verts = layer.nv_lod_mid_to_low.shape[0]
        assert layer.bind_shape.shape[0] == expected_num_verts
        if layer.correctives_model is not None:
            assert layer.correctives_model.module.V == expected_num_verts
    elif lod == "xlo":
        assert layer.nv_lod_mid_to_low is None
        assert layer.identity_lod_transfer is not None
        assert layer.xlo_skeleton_transfer is not None
        if layer.correctives_model is not None:
            assert layer.correctives_model.module.V == layer.xlo_skeleton_mid_to_low.shape[0]
        expected_num_verts = layer.bind_shape.shape[0]

    batch_size = 1
    num_pose_joints = 77
    pose = torch.zeros(batch_size, num_pose_joints, 3, 3, device=device)
    transl = torch.zeros(batch_size, 3, device=device)
    identity_coeffs, scale_params = _make_inputs(layer, identity_model_type, device, batch_size)

    with torch.no_grad():
        out = layer(
            pose,
            identity_coeffs,
            scale_params=scale_params,
            transl=transl,
            pose2rot=False,
            apply_correctives=apply_correctives,
        )

    assert "vertices" in out
    assert "joints" in out
    verts = out["vertices"]
    joints = out["joints"]
    assert verts.dim() == 3 and verts.shape[0] == batch_size and verts.shape[2] == 3
    if lod in {"low", "xlo"}:
        assert verts.shape[1] == expected_num_verts, (
            f"Expected {expected_num_verts} {lod}-LOD vertices, got {verts.shape[1]}"
        )
    assert joints.dim() == 3 and joints.shape[0] == batch_size and joints.shape[2] == 3
    assert joints.shape[1] == num_pose_joints


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_can_skip_correctives_model_for_pure_lbs(data_root):
    """Pure-LBS callers can skip loading the corrective checkpoint."""
    from soma import SOMALayer

    try:
        layer = SOMALayer(
            data_root=data_root,
            lod="low",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            correctives_model_path=None,
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")

    assert layer.correctives_model is None

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 77, 3, 3).contiguous()
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
def test_soma_layer_deprecated_load_correctives_model_false_alias(data_root):
    """Legacy callers can still disable checkpoint loading through the old flag."""
    from soma import SOMALayer

    try:
        with pytest.warns(DeprecationWarning, match="load_correctives_model is deprecated"):
            layer = SOMALayer(
                data_root=data_root,
                lod="low",
                device="cpu",
                identity_model_type="soma",
                mode="dense",
                load_correctives_model=False,
            )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")

    assert layer.correctives_model_path is None
    assert layer.correctives_model is None


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_accepts_custom_correctives_model_path(data_root):
    """Callers can load a corrective checkpoint outside the default resolver."""
    from soma import SOMALayer

    correctives_path = Path(data_root) / "correctives_model.pt"
    if not correctives_path.is_file():
        pytest.skip("Corrective model not available")

    try:
        layer = SOMALayer(
            data_root=data_root,
            lod="low",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            correctives_model_path=correctives_path,
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")

    assert layer.correctives_model_path == correctives_path
    assert layer.correctives_model is not None


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_missing_custom_correctives_model_path_raises(data_root, tmp_path):
    from soma import SOMALayer

    with pytest.raises(FileNotFoundError, match="Correctives model checkpoint not found"):
        SOMALayer(
            data_root=data_root,
            lod="low",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            correctives_model_path=tmp_path / "missing_correctives.pt",
        )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_rejects_correctives_flag_and_explicit_path(data_root):
    from soma import SOMALayer

    correctives_path = Path(data_root) / "correctives_model.pt"
    if not correctives_path.is_file():
        pytest.skip("Corrective model not available")

    with pytest.warns(DeprecationWarning, match="load_correctives_model is deprecated"):
        with pytest.raises(ValueError, match="explicit correctives_model_path"):
            SOMALayer(
                data_root=data_root,
                lod="low",
                device="cpu",
                identity_model_type="soma",
                mode="dense",
                correctives_model_path=correctives_path,
                load_correctives_model=False,
            )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_correctives_require_procedural_transforms(data_root):
    from soma import SOMALayer

    correctives_path = Path(data_root) / "correctives_model.pt"
    if not correctives_path.is_file():
        pytest.skip("Corrective model not available")

    with pytest.raises(ValueError, match="Correctives require procedural transforms"):
        SOMALayer(
            data_root=data_root,
            lod="low",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            enable_procedural_transforms=False,
            correctives_model_path=correctives_path,
        )


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_apply_correctives_requires_loaded_model(data_root):
    from soma import SOMALayer

    try:
        layer = SOMALayer(
            data_root=data_root,
            lod="low",
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            correctives_model_path=None,
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"Missing asset or dependency: {e}")

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 77, 3, 3).contiguous()
    identity_coeffs = torch.zeros(1, layer.num_shape_components)

    with pytest.raises(RuntimeError, match="no corrective model is loaded"):
        with torch.no_grad():
            layer(
                poses,
                identity_coeffs,
                pose2rot=False,
                apply_correctives=True,
            )


@pytest.mark.parametrize(("device", "mode"), _SOMA_FK_ONLY_CASES)
def test_soma_layer_fk_only_matches_full_pose(data_root, device, mode):
    """fk_only=True must return the same transforms as the full pose() path.

    FK is the shared prefix; only the LBS block is skipped. The fast path
    also drops the "vertices" key and pose-dependent correctives.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma import SOMALayer

    layer = SOMALayer(
        data_root=data_root,
        device=device,
        identity_model_type="mhr",
        mode=mode,
    )
    im = layer.identity_model
    torch.manual_seed(0)
    B = 4
    poses = torch.randn(B, 77, 3, device=device) * 0.1
    id_coeffs = torch.zeros(B, im.num_identity_coeffs, device=device)
    scale_params = torch.zeros(B, im.num_scale_params, device=device)
    transl = torch.randn(B, 3, device=device) * 0.2

    layer.prepare_identity(id_coeffs, scale_params=scale_params)

    with torch.no_grad():
        # apply_correctives=False so the reference doesn't add per-vertex
        # corrective offsets; fk_only auto-skips them. transforms must match.
        ref = layer.pose(poses=poses, transl=transl, apply_correctives=False)
        fast = layer.pose(poses=poses, transl=transl, fk_only=True)

    assert "vertices" in ref
    assert "vertices" not in fast, "fk_only=True must not return vertices"
    assert torch.allclose(fast["transforms"], ref["transforms"], atol=1e-6), (
        f"fk_only transforms diverge from pose() transforms: "
        f"max abs = {(fast['transforms'] - ref['transforms']).abs().max().item():.3e}"
    )
    assert torch.allclose(fast["joints"], ref["joints"], atol=1e-6)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_soma_layer_pose_uses_explicit_fk_lbs_pipeline(data_root, monkeypatch):
    from soma import SOMALayer
    from soma.geometry.batched_skinning import BatchedSkinning

    layer = SOMALayer(
        data_root=data_root,
        device="cpu",
        identity_model_type="soma",
        mode="dense",
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, 3, 0] = 0.1
    layer.prepare_identity(identity)

    def fail_pose(*_args, **_kwargs):
        raise AssertionError("SOMALayer.pose should call FK and LBS explicitly")

    monkeypatch.setattr(BatchedSkinning, "pose", fail_pose)

    out = layer.pose(poses, apply_correctives=False)
    fk_only = layer.pose(poses, apply_correctives=False, fk_only=True)

    assert out["vertices"].shape[0] == 1
    assert out["transforms"].shape == (1, 78, 4, 4)
    assert "vertices" not in fk_only


@pytest.mark.parametrize("enable_procedural_transforms", [False, True])
def test_soma_layer_soma_bone_scaling_scales_public_bones(
    data_root,
    enable_procedural_transforms,
):
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=data_root,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=enable_procedural_transforms,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    ones = torch.ones(1, layer.num_scale_params)

    assert layer.NUM_BONE_SCALE_PARAMS == layer.num_scale_params
    assert len(layer.scale_param_names) == layer.num_scale_params
    assert len(layer.scale_param_segments) == layer.num_scale_params
    assert layer.soma_bone_scale_param_names == layer.scale_param_names
    assert layer.soma_bone_scale_param_segments == layer.scale_param_segments
    for active_name in (
        "LeftHand",
        "LeftShin",
        "RightHandIndexEnd",
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
    ):
        assert active_name in layer.scale_param_names
    for inactive_name in (
        "Head",
        "LeftShoulder",
        "RightShoulder",
        "LeftLeg",
        "RightLeg",
        "LeftToeEnd",
        "RightToeEnd",
    ):
        assert inactive_name not in layer.scale_param_names
    assert layer.scale_param_names[-4:] == (
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
    )
    assert layer.scale_param_segments[-4:] == (
        ("LeftShin", "LeftFoot"),
        ("RightShin", "RightFoot"),
        ("LeftFoot", "LeftToeBase"),
        ("RightFoot", "RightToeBase"),
    )
    assert layer.scale_param_names[layer.LEGACY_NUM_BONE_SCALE_PARAMS - 1] == "RightShin"

    with torch.no_grad():
        layer.prepare_identity(identity)
        ref = layer.pose(poses, apply_correctives=False)
        layer.prepare_identity(identity, scale_params=ones)
        unchanged = layer.pose(poses, apply_correctives=False)

    torch.testing.assert_close(unchanged["joints"], ref["joints"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(unchanged["vertices"], ref["vertices"], atol=1e-6, rtol=1e-6)

    legacy_scales = ones[:, : layer.LEGACY_NUM_BONE_SCALE_PARAMS].clone()
    legacy_scales[:, -1] = 1.1
    current_equivalent_scales = ones.clone()
    current_equivalent_scales[:, layer.scale_param_names.index("RightShin")] = 1.1
    with torch.no_grad():
        layer.prepare_identity(identity, scale_params=legacy_scales)
        legacy_output = layer.pose(poses, apply_correctives=False)
        layer.prepare_identity(identity, scale_params=current_equivalent_scales)
        current_equivalent_output = layer.pose(poses, apply_correctives=False)
    torch.testing.assert_close(
        legacy_output["joints"],
        current_equivalent_output["joints"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        legacy_output["vertices"],
        current_equivalent_output["vertices"],
        atol=1e-6,
        rtol=1e-6,
    )

    public_names = layer.public_joint_names
    left_forearm_joint = public_names.index("LeftForeArm") - 1
    left_hand_joint = public_names.index("LeftHand") - 1
    left_hand_scale_idx = layer.scale_param_names.index("LeftHand")
    assert layer.scale_param_segments[left_hand_scale_idx] == ("LeftForeArm", "LeftHand")
    scaled = ones.clone()
    scaled[:, left_hand_scale_idx] = 1.5

    with torch.no_grad():
        layer.prepare_identity(identity, scale_params=scaled)
        scaled_out = layer.pose(poses, apply_correctives=False)

    base_length = torch.linalg.norm(
        unchanged["joints"][0, left_hand_joint] - unchanged["joints"][0, left_forearm_joint]
    )
    scaled_length = torch.linalg.norm(
        scaled_out["joints"][0, left_hand_joint] - scaled_out["joints"][0, left_forearm_joint]
    )
    assert scaled_length.item() == pytest.approx(base_length.item() * 1.5, rel=1e-4)
    assert not torch.allclose(scaled_out["vertices"], unchanged["vertices"])

    left_shin_joint = public_names.index("LeftShin") - 1
    left_foot_joint = public_names.index("LeftFoot") - 1
    left_toe_joint = public_names.index("LeftToeBase") - 1
    left_toe_end_joint = public_names.index("LeftToeEnd") - 1
    left_foot_scale_idx = layer.scale_param_names.index("LeftFoot")
    lower_leg_scaled = ones.clone()
    lower_leg_scaled[:, left_foot_scale_idx] = 1.25
    with torch.no_grad():
        layer.prepare_identity(identity, scale_params=lower_leg_scaled)
        lower_leg_scaled_out = layer.pose(poses, apply_correctives=False)
    base_lower_leg_length = torch.linalg.norm(
        unchanged["joints"][0, left_foot_joint] - unchanged["joints"][0, left_shin_joint]
    )
    scaled_lower_leg_length = torch.linalg.norm(
        lower_leg_scaled_out["joints"][0, left_foot_joint]
        - lower_leg_scaled_out["joints"][0, left_shin_joint]
    )
    base_foot_length = torch.linalg.norm(
        unchanged["joints"][0, left_toe_joint] - unchanged["joints"][0, left_foot_joint]
    )
    lower_leg_scaled_foot_length = torch.linalg.norm(
        lower_leg_scaled_out["joints"][0, left_toe_joint]
        - lower_leg_scaled_out["joints"][0, left_foot_joint]
    )
    assert scaled_lower_leg_length.item() == pytest.approx(
        base_lower_leg_length.item() * 1.25,
        rel=1e-4,
    )
    torch.testing.assert_close(
        lower_leg_scaled_foot_length,
        base_foot_length,
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.allclose(lower_leg_scaled_out["vertices"], unchanged["vertices"])

    left_toe_scale_idx = layer.scale_param_names.index("LeftToeBase")
    toe_scaled = ones.clone()
    toe_scaled[:, left_toe_scale_idx] = 1.25
    with torch.no_grad():
        layer.prepare_identity(identity, scale_params=toe_scaled)
        toe_scaled_out = layer.pose(poses, apply_correctives=False)
    scaled_foot_length = torch.linalg.norm(
        toe_scaled_out["joints"][0, left_toe_joint] - toe_scaled_out["joints"][0, left_foot_joint]
    )
    base_toe_length = torch.linalg.norm(
        unchanged["joints"][0, left_toe_end_joint] - unchanged["joints"][0, left_toe_joint]
    )
    scaled_toe_length = torch.linalg.norm(
        toe_scaled_out["joints"][0, left_toe_end_joint]
        - toe_scaled_out["joints"][0, left_toe_joint]
    )
    assert scaled_foot_length.item() == pytest.approx(base_foot_length.item() * 1.25, rel=1e-4)
    torch.testing.assert_close(scaled_toe_length, base_toe_length, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(toe_scaled_out["vertices"], unchanged["vertices"])

    trainable_scales = ones.clone().requires_grad_(True)
    layer.prepare_identity(identity, scale_params=trainable_scales)
    trainable_out = layer.pose(poses, apply_correctives=False)
    trainable_out["vertices"].square().sum().backward()
    lower_limb_gradients = trainable_scales.grad[
        0,
        [
            layer.scale_param_names.index("LeftFoot"),
            layer.scale_param_names.index("RightFoot"),
            layer.scale_param_names.index("LeftToeBase"),
            layer.scale_param_names.index("RightToeBase"),
        ],
    ]
    assert torch.isfinite(lower_limb_gradients).all()
    assert torch.count_nonzero(lower_limb_gradients) == 4

    with pytest.raises(
        ValueError,
        match=(
            rf"\(B, {layer.num_scale_params}\).*legacy shape "
            rf"\(B, {layer.LEGACY_NUM_BONE_SCALE_PARAMS}\)"
        ),
    ):
        layer.prepare_identity(identity, scale_params=torch.ones(1, 77))
