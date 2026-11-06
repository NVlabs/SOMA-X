# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from soma import body
from soma.fitting.pose_inversion import PoseInversion


@pytest.mark.parametrize("procedural", [False, True])
def test_mid_lod_inversion_constructs_low_lod_body_layer(monkeypatch, tmp_path, procedural):
    original = SimpleNamespace(
        lod="mid",
        low_lod=False,
        data_root=tmp_path,
        device=torch.device("cpu"),
        identity_model_type="mhr",
        mode="warp",
        output_unit="cm",
        identity_model_kwargs={"example_option": 3},
        procedural_template_rig_path=tmp_path / "template.usdc",
        procedural_transforms_enabled=procedural,
    )
    low = SimpleNamespace(
        low_lod=True,
        bind_shape=torch.empty(5, 3),
        nv_lod_mid_to_low=None,
        identity_model_type="mhr",
    )
    calls = []

    def make_layer(data_root, **kwargs):
        calls.append((data_root, kwargs))
        return low

    monkeypatch.setattr(body, "SOMALayer", make_layer)

    inversion = PoseInversion(original, low_lod=True)

    assert inversion.soma is low
    assert inversion._soma_orig is original
    assert inversion._autograd_soma is (low if procedural else None)
    assert inversion._soma_num_verts == 5
    assert calls == [
        (
            tmp_path,
            {
                "lod": "low",
                "device": original.device,
                "identity_model_type": "mhr",
                "mode": original.mode,
                "output_unit": original.output_unit,
                "identity_model_kwargs": original.identity_model_kwargs,
                "template_rig_path": original.procedural_template_rig_path,
                "enable_procedural_transforms": procedural,
            },
        )
    ]
