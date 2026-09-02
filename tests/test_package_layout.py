# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest


def test_body_layer_import_paths_resolve_to_one_implementation() -> None:
    from soma import SOMALayer, SOMAPoseOutput
    from soma.body import SOMALayer as BodySOMALayer
    from soma.body import SOMAPoseOutput as BodySOMAPoseOutput
    from soma.body import SOMAPublicRigView as BodySOMAPublicRigView
    from soma.body.soma import SOMALayer as ImplementationSOMALayer
    from soma.body.soma import SOMAPoseOutput as ImplementationSOMAPoseOutput
    from soma.body.soma import SOMAPublicRigView as ImplementationSOMAPublicRigView
    from soma.soma import SOMALayer as LegacySOMALayer
    from soma.soma import SOMAPoseOutput as LegacySOMAPoseOutput
    from soma.soma import SOMAPublicRigView as LegacySOMAPublicRigView

    assert SOMALayer is BodySOMALayer is ImplementationSOMALayer is LegacySOMALayer
    assert (
        SOMAPoseOutput is BodySOMAPoseOutput is ImplementationSOMAPoseOutput is LegacySOMAPoseOutput
    )
    assert BodySOMAPublicRigView is ImplementationSOMAPublicRigView is LegacySOMAPublicRigView


def test_legacy_body_module_resolves_serialized_class_references() -> None:
    from soma.body import SOMALayer, SOMAPoseOutput, SOMAPublicRigView

    for class_type in (SOMALayer, SOMAPoseOutput, SOMAPublicRigView):
        canonical_payload = pickle.dumps(class_type, protocol=0)
        assert b"soma.body.soma" in canonical_payload

        legacy_payload = canonical_payload.replace(b"soma.body.soma", b"soma.soma")
        assert pickle.loads(legacy_payload) is class_type


def test_identity_model_import_paths_resolve_to_one_implementation() -> None:
    import soma
    import soma.identity_model as legacy_module
    from soma.body import identity_model as body_module

    assert soma.create_identity_model is body_module.create_identity_model
    assert legacy_module.create_identity_model is body_module.create_identity_model
    for name in (
        "AnnyIdentityModel",
        "GarmentMeasurementIdentityModel",
        "MHRIdentityModel",
        "SMPLIdentityModel",
        "SOMAIdentityModel",
    ):
        assert getattr(legacy_module, name) is getattr(body_module, name)
        assert issubclass(getattr(body_module, name), legacy_module.BaseIdentityModel)
    # The hand package shares the base classes without importing from soma.body.
    from soma.hand.identity_model import BaseHandIdentityModel

    assert issubclass(BaseHandIdentityModel, legacy_module.BaseIdentityModel)
    with pytest.raises(AttributeError):
        legacy_module.DoesNotExist  # noqa: B018


def test_legacy_identity_module_resolves_serialized_class_references() -> None:
    from soma.body.identity_model import MHRIdentityModel, SOMAIdentityModel

    for class_type in (MHRIdentityModel, SOMAIdentityModel):
        canonical_payload = pickle.dumps(class_type, protocol=0)
        assert b"soma.body.identity_model" in canonical_payload

        legacy_payload = canonical_payload.replace(
            b"soma.body.identity_model", b"soma.identity_model"
        )
        assert pickle.loads(legacy_payload) is class_type


def test_fitting_import_paths_resolve_to_one_implementation() -> None:
    from soma.fitting import MHRPoseInversion, PoseInversion, smooth_pose
    from soma.fitting.pose_inversion import PoseInversion as ImplementationPoseInversion
    from soma.fitting.pose_inversion_mhr import MHRPoseInversion as ImplementationMHRPoseInversion
    from soma.fitting.rts_smoothing import smooth_pose as implementation_smooth_pose
    from soma.pose_inversion import PoseInversion as LegacyPoseInversion
    from soma.pose_inversion_mhr import MHRPoseInversion as LegacyMHRPoseInversion
    from soma.rts_smoothing import smooth_pose as legacy_smooth_pose

    assert PoseInversion is ImplementationPoseInversion is LegacyPoseInversion
    assert MHRPoseInversion is ImplementationMHRPoseInversion is LegacyMHRPoseInversion
    assert smooth_pose is implementation_smooth_pose is legacy_smooth_pose


def test_legacy_fitting_modules_resolve_serialized_class_references() -> None:
    from soma.fitting.pose_inversion import PoseInversion, PoseInversionResult
    from soma.fitting.rts_smoothing import RTSSmoothingConfig

    cases = (
        (PoseInversion, b"soma.fitting.pose_inversion", b"soma.pose_inversion"),
        (PoseInversionResult, b"soma.fitting.pose_inversion", b"soma.pose_inversion"),
        (RTSSmoothingConfig, b"soma.fitting.rts_smoothing", b"soma.rts_smoothing"),
    )
    for class_type, canonical, legacy in cases:
        canonical_payload = pickle.dumps(class_type, protocol=0)
        assert canonical in canonical_payload

        legacy_payload = canonical_payload.replace(canonical, legacy)
        assert pickle.loads(legacy_payload) is class_type


def test_legacy_fitting_modules_delegate_private_names() -> None:
    import soma.fitting.pose_inversion as impl_pose
    import soma.fitting.pose_inversion_mhr as impl_mhr
    import soma.fitting.rts_smoothing as impl_rts
    import soma.pose_inversion as legacy_pose
    import soma.pose_inversion_mhr as legacy_mhr
    import soma.rts_smoothing as legacy_rts

    # Private helpers importable from the pre-0.3 module paths keep resolving.
    assert legacy_pose._align_vectors_auto is impl_pose._align_vectors_auto
    assert legacy_pose._AUTO_REFIT_PRIOR_STRENGTH is impl_pose._AUTO_REFIT_PRIOR_STRENGTH
    assert legacy_mhr._pose_local_from_result is impl_mhr._pose_local_from_result
    assert legacy_rts.RotationConvention is impl_rts.RotationConvention
    with pytest.raises(AttributeError):
        legacy_pose.does_not_exist  # noqa: B018
