# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference Maya helpers for SOMA procedural-control rig definitions."""

from .definition import (
    MatrixEntry,
    ProceduralDefinition,
    Segment,
    compile_rotation_rows,
    compile_translation_rows,
    load_definition,
)
from .evaluator import ProceduralEvaluation, SomaProceduralTransformEvaluator
from .maya_reference import (
    connect_procedural_node_to_rig,
    find_repo_definition,
    find_repo_template_asset,
    load_template_usd_rig,
    pose_template_rig_to_bind_pose,
    restore_template_rig_pose,
    setup_template_rig_scene,
)

__all__ = [
    "MatrixEntry",
    "ProceduralDefinition",
    "ProceduralEvaluation",
    "Segment",
    "SomaProceduralTransformEvaluator",
    "compile_rotation_rows",
    "connect_procedural_node_to_rig",
    "find_repo_definition",
    "find_repo_template_asset",
    "load_template_usd_rig",
    "pose_template_rig_to_bind_pose",
    "restore_template_rig_pose",
    "compile_translation_rows",
    "load_definition",
    "setup_template_rig_scene",
]
