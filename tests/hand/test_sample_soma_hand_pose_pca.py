from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.hand.sample_soma_hand_pose_pca import (
    bind_relative_to_absolute_local,
    load_pose_prior,
    sample_pose_prior,
)

pytestmark = pytest.mark.hand

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prior(num_components=3):
    components = np.zeros((num_components, 72), dtype=np.float32)
    components[np.arange(num_components), np.arange(num_components)] = 1.0
    return {
        "mean": np.full((24, 3), 0.1, dtype=np.float32),
        "components": components,
        "eigenvalues": np.arange(1, num_components + 1, dtype=np.float32),
    }


def test_sample_pose_prior_uses_standardized_coefficient_convention():
    prior = _prior()
    expected_coefficients = np.random.default_rng(17).standard_normal((2, 2))

    samples = sample_pose_prior(
        prior,
        num_poses=2,
        n_components=2,
        seed=17,
        sample_scale=0.5,
    )

    expected = np.full((2, 72), 0.1)
    expected[:, :2] += expected_coefficients * 0.5 * np.sqrt([1.0, 2.0])
    np.testing.assert_allclose(samples.reshape(2, 72), expected, atol=1e-7)


@pytest.mark.asset_heavy
def test_load_pose_prior_from_checked_in_hand_asset():
    hand_asset = REPO_ROOT / "assets/SOMAHand.npz"
    if not hand_asset.is_file():
        pytest.skip("SOMAHand asset is not installed")

    prior, metadata = load_pose_prior(hand_asset, "left")

    assert prior["mean"].shape == (24, 3)
    assert prior["components"].shape == (32, 72)
    assert prior["eigenvalues"].shape == (32,)
    assert metadata["identity_at_bind_pose"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_poses": 0, "n_components": 2, "sample_scale": 1.0}, "num_poses"),
        ({"num_poses": 2, "n_components": 4, "sample_scale": 1.0}, "n_components"),
        ({"num_poses": 2, "n_components": 2, "sample_scale": 0.0}, "sample_scale"),
    ],
)
def test_sample_pose_prior_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sample_pose_prior(_prior(), seed=1, **kwargs)


def test_bind_relative_decoder_has_identity_at_bind_and_round_trips():
    rng = np.random.default_rng(9)
    reference_orient = Rotation.from_rotvec(rng.normal(scale=0.2, size=(25, 3))).as_matrix()
    parent_ids = np.asarray(
        [0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23]
    )
    reference_orient_parent = reference_orient[parent_ids]
    samples = rng.normal(scale=0.1, size=(3, 24, 3))

    absolute = bind_relative_to_absolute_local(
        samples,
        reference_orient,
        reference_orient_parent,
    )
    recovered_relative = reference_orient_parent @ absolute @ reference_orient.swapaxes(-2, -1)
    recovered = Rotation.from_matrix(recovered_relative[:, 1:].reshape(-1, 3, 3)).as_rotvec()

    np.testing.assert_allclose(recovered.reshape(3, 24, 3), samples, atol=1e-12)
    bind_absolute = bind_relative_to_absolute_local(
        np.zeros((1, 24, 3)),
        reference_orient,
        reference_orient_parent,
    )
    expected_bind = reference_orient_parent.swapaxes(-2, -1) @ reference_orient
    np.testing.assert_allclose(bind_absolute[0], expected_bind, atol=1e-12)
