from types import SimpleNamespace

import numpy as np
import torch

from soma.io import load_soma_npz, save_soma_npz
from tools import identity_conversion
from tools.identity_conversion import (
    _poses_for_resave,
    convert_identity_parameters,
    convert_soma_npz,
    neutral_scale_params,
)


class _FakeLayer:
    def __init__(self, backend, *, hand=False):
        self.identity_model_type = backend
        if hand:
            self.hand_type = "right"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_shape_components = 1
        self.num_scale_params = 1
        self.identity_model = SimpleNamespace(num_identity_coeffs=1, num_scale_params=1)
        self.joint_parent_ids = torch.tensor([-1, 0])
        self.bind_pose_local = torch.eye(4).repeat(2 if hand else 3, 1, 1)
        self.public_transform_joint_indices = torch.arange(3)
        self._cached_scale_params = None

    def prepare_identity(
        self,
        identity_coeffs,
        scale_params=None,
        repose_to_bind_pose=True,
        global_scale=1.0,
    ):
        del repose_to_bind_pose
        base = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        identity_direction = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]])
        scale_direction = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        scale_value = scale_params[:, :1] if scale_params is not None else 0.0
        if self.identity_model_type == "soma" and scale_params is not None:
            scale_value = scale_value - 1.0
        self._cached_rest_shape = global_scale * (
            base
            + identity_coeffs[:, :1, None] * identity_direction
            + scale_value[:, :, None] * scale_direction
        )
        self._cached_scale_params = scale_params

    def pose(self, rotations, **kwargs):
        del rotations, kwargs
        return {"vertices": self._cached_rest_shape}


def test_neutral_soma_scale_params_are_one():
    scales = neutral_scale_params(_FakeLayer("soma", hand=True), 2)
    assert scales.shape == (2, 1)
    assert torch.equal(scales, torch.ones_like(scales))


def test_convert_identity_optimizes_soma_coefficients_and_bone_scale():
    result = convert_identity_parameters(
        _FakeLayer("mhr"),
        _FakeLayer("soma"),
        np.array([[0.7]], dtype=np.float32),
        source_scale_params=np.array([[0.2]], dtype=np.float32),
        iterations=500,
        learning_rate=0.05,
        regularization=0.0,
    )

    np.testing.assert_allclose(result.identity_coeffs, [[0.7]], atol=2e-3)
    np.testing.assert_allclose(result.scale_params, [[1.2]], atol=2e-3)
    np.testing.assert_allclose(result.global_scale, 1.0)
    assert result.vertex_error[0] < 1e-3


def test_global_scale_is_fixed_by_default():
    result = convert_identity_parameters(
        _FakeLayer("mhr"),
        _FakeLayer("soma"),
        np.array([[0.3]], dtype=np.float32),
        source_scale_params=np.array([[0.0]], dtype=np.float32),
        global_scale=1.4,
        iterations=10,
    )
    assert result.global_scale == 1.4


def test_poses_for_resave_restores_root_for_save_soma_npz(tmp_path):
    input_path = tmp_path / "input.npz"
    save_soma_npz(
        input_path,
        np.zeros((1, 3, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        joint_names=["Root", "Hips", "Spine"],
        identity_model_type="mhr",
        identity_coeffs=np.zeros((1, 1), dtype=np.float32),
        scale_params=np.zeros((1, 1), dtype=np.float32),
    )
    loaded = load_soma_npz(input_path)

    poses, names = _poses_for_resave(loaded)

    assert poses.shape == (1, 3, 3)
    assert names == ["Root", "Hips", "Spine"]


def test_convert_soma_npz_preserves_animation_and_records_source(tmp_path, monkeypatch):
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "output.npz"
    poses = np.zeros((2, 3, 3), dtype=np.float32)
    transl = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_soma_npz(
        input_path,
        poses,
        transl,
        joint_names=["Root", "Hips", "Spine"],
        identity_model_type="mhr",
        identity_coeffs=np.array([[0.4]], dtype=np.float32),
        scale_params=np.array([[0.1]], dtype=np.float32),
        extra_arrays={"custom": np.array([3, 4])},
    )
    input_data = load_soma_npz(input_path)
    input_data["allow_pickle"] = np.array(False)
    monkeypatch.setattr(identity_conversion, "load_soma_npz", lambda _: input_data)

    def layer_factory(backend, unit, role):
        assert unit == "meters"
        assert role in {"source", "target"}
        return _FakeLayer(backend)

    convert_soma_npz(
        input_path,
        output_path,
        target_backend="soma",
        layer_factory=layer_factory,
        iterations=400,
        learning_rate=0.05,
        regularization=0.0,
    )
    converted = load_soma_npz(output_path)

    assert converted.identity_model_type == "soma"
    np.testing.assert_array_equal(converted.poses, poses[:, 1:])
    np.testing.assert_array_equal(converted.transl, transl)
    np.testing.assert_array_equal(converted.custom, [3, 4])
    assert str(converted.conversion_source_identity_model_type) == "mhr"
    np.testing.assert_allclose(converted.conversion_source_identity_coeffs, [[0.4]])
