from pathlib import Path

import numpy as np
import pytest

from tests.hand._test_layers import make_hand_layer

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"
HAND_ASSET = ASSETS_DIR / "SOMAHand.npz"
TEMPLATE_RIG = ASSETS_DIR / "SOMA_template_rig.usda"
PROCEDURAL_DEFINITION = ASSETS_DIR / "SOMA_procedural_transforms.json"
_HAND_TEMPLATE_REGRESSION_CASES = [
    pytest.param("left", "mid", id="left-mid", marks=(pytest.mark.cpu, pytest.mark.asset_heavy)),
    pytest.param("right", "mid", id="right-mid", marks=(pytest.mark.cpu, pytest.mark.asset_heavy)),
    pytest.param(
        "right",
        "xlo",
        id="right-xlo",
        marks=(pytest.mark.slow, pytest.mark.cpu, pytest.mark.xlo, pytest.mark.asset_heavy),
    ),
]


@pytest.fixture(scope="module")
def data_root():
    if not ASSETS_DIR.is_dir():
        pytest.fail(
            f"Assets directory not found: {ASSETS_DIR}. "
            "Clone the repo and run `git lfs pull` to fetch assets."
        )
    for asset in (CORE_ASSET, HAND_ASSET, TEMPLATE_RIG):
        if not asset.is_file():
            pytest.fail(f"Required asset not found: {asset}. Run `git lfs pull`.")
    return str(ASSETS_DIR)


@pytest.fixture(scope="module")
def expanded_template_mid_rig():
    from soma.io import load_lod_rig_from_usd

    public_joint_names = _public_joint_names()
    template_rig = load_lod_rig_from_usd(TEMPLATE_RIG, "mid")
    if len(template_rig["joint_names"]) <= len(public_joint_names):
        pytest.skip("Template rig is not expanded relative to the public SOMA rig")
    return template_rig


def _public_joint_names():
    from soma.procedural_transforms import load_soma_procedural_transform_definition

    definition = load_soma_procedural_transform_definition(PROCEDURAL_DEFINITION)
    return list(definition.public_joint_names)


def _expected_hand_joint_names(hand_type):
    hand_map = np.load(HAND_ASSET, allow_pickle=False)
    hand_joint_ids = hand_map[f"{hand_type}_hand_joint_ids_global"]
    public_joint_names = _public_joint_names()
    return [public_joint_names[idx] for idx in hand_joint_ids]


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_template_hand_indices_resolve_against_derived_public_rig(
    expanded_template_mid_rig,
    hand_type,
):
    from soma.procedural_transforms import derive_soma_rig_without_procedural_joints

    hand_map = np.load(HAND_ASSET, allow_pickle=False)
    hand_joint_ids = hand_map[f"{hand_type}_hand_joint_ids_global"]
    public_rig = derive_soma_rig_without_procedural_joints(
        expanded_template_mid_rig,
        _public_joint_names(),
    )

    public_hand_names = [str(public_rig["joint_names"][idx]) for idx in hand_joint_ids]

    assert public_hand_names == _expected_hand_joint_names(hand_type)


@pytest.mark.parametrize(("hand_type", "lod"), _HAND_TEMPLATE_REGRESSION_CASES)
def test_hand_layer_derives_public_hand_names_for_expanded_template(
    data_root,
    expanded_template_mid_rig,
    hand_type,
    lod,
):
    layer, skip_reason = make_hand_layer(
        data_root,
        hand_type,
        "cpu",
        lod=lod,
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    assert layer.rig_data["joint_names"] == _expected_hand_joint_names(hand_type)
    assert layer.joint_parent_ids.shape == (25,)
    assert len(layer.hand_joint_ids_global) == 25
