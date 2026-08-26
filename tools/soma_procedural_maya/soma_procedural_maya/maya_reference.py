# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maya scene helpers for the SOMA procedural-control reference plug-in."""

import json
from pathlib import Path
from typing import Any

from .definition import (
    ProceduralDefinition,
    compile_rotation_rows,
    compile_translation_rows,
    load_definition,
)

DEFINITION_FILENAME = "SOMA_procedural_transforms.json"
TEMPLATE_RIG_FILENAME = "SOMA_template_rig.usda"
NODE_TYPE_NAME = "somaProceduralTransforms"
DEFAULT_TEMPLATE_RIG_GROUP = "SOMA_templateRig_PROC_GRP"


def find_repo_definition(start: str | Path | None = None) -> Path | None:
    """Find the checked-in SOMA procedural definition by walking parent dirs."""

    current = Path(__file__).resolve() if start is None else Path(start).resolve()
    for parent in (current, *current.parents):
        candidate = parent / "assets" / DEFINITION_FILENAME
        if candidate.exists():
            return candidate
    return None


def find_repo_template_asset(start: str | Path | None = None) -> Path | None:
    """Find the checked-in SOMA template USD rig by walking parent dirs."""

    current = Path(__file__).resolve() if start is None else Path(start).resolve()
    for parent in (current, *current.parents):
        candidate = parent / "assets" / TEMPLATE_RIG_FILENAME
        if candidate.exists():
            return candidate
    return None


def _safe_node_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_:" else "_" for ch in name)


def _leaf_name(node: str) -> str:
    return node.split("|")[-1].split(":")[-1]


def _descendants(cmds: Any, root: str) -> list[str]:
    return cmds.listRelatives(root, allDescendents=True, fullPath=True) or []


def _has_matrix_attrs(cmds: Any, node: str) -> bool:
    return cmds.attributeQuery("matrix", node=node, exists=True) and cmds.attributeQuery(
        "worldMatrix", node=node, exists=True
    )


def _resolve_nodes_by_leaf(cmds: Any, root: str, names: set[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for node in _descendants(cmds, root):
        name = _leaf_name(node)
        if name not in names or not _has_matrix_attrs(cmds, node):
            continue
        old = resolved.get(name)
        if old is None or (cmds.nodeType(node) == "joint" and cmds.nodeType(old) != "joint"):
            resolved[name] = node
    return resolved


def _load_usd_skeleton_poses(
    template_asset_path: str | Path,
) -> tuple[tuple[str, ...], tuple[list[float], ...], tuple[list[float], ...]]:
    from pxr import Usd, UsdSkel

    stage = Usd.Stage.Open(str(template_asset_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD template rig: {template_asset_path}")
    skeleton_prim = next((prim for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)), None)
    if skeleton_prim is None:
        raise RuntimeError(f"No UsdSkelSkeleton found in {template_asset_path}")

    skeleton = UsdSkel.Skeleton(skeleton_prim)
    joint_paths = skeleton.GetJointsAttr().Get()
    bind_transforms = skeleton.GetBindTransformsAttr().Get()
    rest_transforms = skeleton.GetRestTransformsAttr().Get()
    if not joint_paths:
        raise RuntimeError(f"USD skeleton has no joints in {template_asset_path}")
    if bind_transforms is None or len(bind_transforms) != len(joint_paths):
        raise RuntimeError(
            f"USD skeleton must have one bindTransform per joint in {template_asset_path}"
        )
    if rest_transforms is None or len(rest_transforms) != len(joint_paths):
        raise RuntimeError(
            f"USD skeleton must have one restTransform per joint in {template_asset_path}"
        )

    joint_names = tuple(str(path).rsplit("/", 1)[-1] for path in joint_paths)
    if len(set(joint_names)) != len(joint_names):
        raise RuntimeError(f"USD skeleton has duplicate joint leaf names in {template_asset_path}")
    bind_world = tuple(
        [float(transform[row][column]) for row in range(4) for column in range(4)]
        for transform in bind_transforms
    )
    rest_local = tuple(
        [float(transform[row][column]) for row in range(4) for column in range(4)]
        for transform in rest_transforms
    )
    return joint_names, bind_world, rest_local


def pose_template_rig_to_bind_pose(
    template_asset_path: str | Path,
    rig_root: str = DEFAULT_TEMPLATE_RIG_GROUP,
    joint_names: set[str] | None = None,
) -> dict[str, list[float]]:
    """Pose an imported template skeleton to its USD bind transforms.

    Returns the previous world matrices, which can be passed to
    :func:`restore_template_rig_pose`. The USD asset must author both
    ``restTransforms`` and ``bindTransforms`` so the two poses cannot be
    accidentally conflated.
    """

    from maya import cmds

    usd_joint_names, bind_world, _rest_local = _load_usd_skeleton_poses(template_asset_path)
    if joint_names is None:
        selected = tuple(zip(usd_joint_names, bind_world, strict=True))
    else:
        unknown = sorted(joint_names - set(usd_joint_names))
        if unknown:
            raise ValueError(f"Requested joints are not present in the USD skeleton: {unknown}")
        selected = tuple(
            (name, matrix)
            for name, matrix in zip(usd_joint_names, bind_world, strict=True)
            if name in joint_names
        )
    selected_names = tuple(name for name, _matrix in selected)
    resolved = _resolve_nodes_by_leaf(cmds, rig_root, set(selected_names))
    missing = [name for name in selected_names if name not in resolved]
    if missing:
        raise ValueError(f"Could not resolve imported SOMA joints for bind pose: {missing}")

    previous_world = {
        name: [float(value) for value in cmds.getAttr(f"{resolved[name]}.worldMatrix[0]")]
        for name in selected_names
    }
    try:
        for name, matrix in selected:
            cmds.xform(resolved[name], worldSpace=True, matrix=matrix)
    except Exception:
        restore_template_rig_pose(previous_world, rig_root=rig_root)
        raise
    return previous_world


def restore_template_rig_pose(
    world_matrices: dict[str, list[float]],
    rig_root: str = DEFAULT_TEMPLATE_RIG_GROUP,
) -> None:
    """Restore world matrices returned by :func:`pose_template_rig_to_bind_pose`."""

    from maya import cmds

    resolved = _resolve_nodes_by_leaf(cmds, rig_root, set(world_matrices))
    missing = [name for name in world_matrices if name not in resolved]
    if missing:
        raise ValueError(f"Could not resolve imported SOMA joints for pose restore: {missing}")
    for name, matrix in world_matrices.items():
        cmds.xform(resolved[name], worldSpace=True, matrix=matrix)


def _matrix_to_list(matrix: Any) -> list[float]:
    return [float(matrix.getElement(row, column)) for row in range(4) for column in range(4)]


def _parent_world_inverse_attr(cmds: Any, node: str) -> str | None:
    parents = cmds.listRelatives(node, parent=True, fullPath=True) or []
    if not parents:
        return None
    return f"{parents[0]}.worldInverseMatrix[0]"


def _reset_offset_parent_matrix(cmds: Any, node: str) -> None:
    import maya.api.OpenMaya as om

    attr = f"{node}.offsetParentMatrix"
    if not cmds.attributeQuery("offsetParentMatrix", node=node, exists=True):
        return
    for source in cmds.listConnections(attr, source=True, destination=False, plugs=True) or []:
        cmds.disconnectAttr(source, attr)
    cmds.setAttr(attr, *_matrix_to_list(om.MMatrix()), type="matrix")


def _world_rotation_matrix(cmds: Any, node: str) -> list[float]:
    values = list(cmds.getAttr(f"{node}.worldMatrix[0]"))
    for index in (3, 7, 11, 12, 13, 14):
        values[index] = 0.0
    values[15] = 1.0
    return [float(value) for value in values]


def _inverse_matrix(values: list[float]) -> list[float]:
    import maya.api.OpenMaya as om

    return _matrix_to_list(om.MMatrix(values).inverse())


def _create_hold_matrix(cmds: Any, name: str, matrix: list[float]) -> str:
    node = cmds.createNode("holdMatrix", name=_safe_node_name(name))
    cmds.setAttr(f"{node}.inMatrix", *matrix, type="matrix")
    return node


def _set_string_attr(cmds: Any, node: str, attr: str, value: str) -> None:
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, dataType="string")
    cmds.setAttr(f"{node}.{attr}", value, type="string")


def _set_bool_attr(cmds: Any, node: str, attr: str, value: bool) -> None:
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, attributeType="bool")
    cmds.setAttr(f"{node}.{attr}", bool(value))


def _set_float_attr(cmds: Any, node: str, attr: str, value: float) -> None:
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, attributeType="double")
    cmds.setAttr(f"{node}.{attr}", float(value))


def _set_int_attr(cmds: Any, node: str, attr: str, value: int) -> None:
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, attributeType="long")
    cmds.setAttr(f"{node}.{attr}", int(value))


def _set_builtin_string_attr(cmds: Any, node: str, attr: str, value: str) -> bool:
    if not cmds.attributeQuery(attr, node=node, exists=True):
        return False
    cmds.setAttr(f"{node}.{attr}", value, type="string")
    return True


def _matrix_payload(definition: ProceduralDefinition) -> str:
    payload = {
        "rotation": compile_rotation_rows(definition),
        "translation": compile_translation_rows(definition),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def create_reference_scene_nodes(
    definition_path: str | Path | None = None,
    node_name: str = "SOMA_proceduralTransforms",
    create_locators: bool = True,
) -> str:
    """Create a Maya procedural-transform node from the SOMA definition.

    The node evaluates twist-joint output matrices from public SOMA input
    matrices. It also stores sidecar metadata for inspection. Optional locators
    are connected to the node's output matrix array for quick visual smoke tests.
    """

    from maya import cmds

    if definition_path is None:
        definition_path = find_repo_definition()
        if definition_path is None:
            raise ValueError(
                "Could not find assets/SOMA_procedural_transforms.json. "
                "Pass -definitionPath explicitly."
            )

    definition = load_definition(definition_path)

    network = cmds.createNode(NODE_TYPE_NAME, name=_safe_node_name(node_name))
    _set_builtin_string_attr(cmds, network, "definitionPath", str(Path(definition_path).resolve()))
    _set_string_attr(cmds, network, "somaDefinitionPath", str(Path(definition_path).resolve()))
    _set_string_attr(cmds, network, "somaSchemaVersion", definition.schema_version)
    _set_string_attr(cmds, network, "somaDefinitionName", definition.definition_name)
    _set_string_attr(
        cmds, network, "somaPublicJointNames", json.dumps(definition.public_joint_names)
    )
    _set_string_attr(cmds, network, "somaTwistJointNames", json.dumps(definition.twist_joint_names))
    _set_string_attr(cmds, network, "somaMatrixRows", _matrix_payload(definition))

    segment_payload = [
        {
            "start_joint": segment.start_joint,
            "end_joint": segment.end_joint,
            "parent_joint": segment.parent_joint,
            "twist_joints": segment.twist_joints,
            "reverse": segment.reverse,
            "source_axis": segment.source_axis,
            "source_axis_id": segment.source_axis_id,
            "source_sign": segment.source_sign,
        }
        for segment in definition.segments
    ]
    _set_string_attr(cmds, network, "somaSegments", json.dumps(segment_payload, sort_keys=True))

    if create_locators:
        output_index = 0
        for segment in definition.segments:
            for twist_joint in segment.twist_joints:
                transform = cmds.spaceLocator(name=_safe_node_name(f"{twist_joint}_somaProcRef"))[0]
                _set_string_attr(cmds, transform, "somaDefinitionNode", network)
                _set_string_attr(cmds, transform, "somaTwistJoint", twist_joint)
                _set_int_attr(cmds, transform, "somaOutputIndex", output_index)
                _set_string_attr(cmds, transform, "somaStartJoint", segment.start_joint)
                _set_string_attr(cmds, transform, "somaEndJoint", segment.end_joint)
                _set_string_attr(cmds, transform, "somaSourceAxis", segment.source_axis)
                _set_int_attr(cmds, transform, "somaSourceAxisId", segment.source_axis_id)
                _set_float_attr(cmds, transform, "somaSourceSign", segment.source_sign)
                _set_bool_attr(cmds, transform, "somaReverseCompensation", segment.reverse)
                if cmds.attributeQuery("offsetParentMatrix", node=transform, exists=True):
                    cmds.connectAttr(
                        f"{network}.outputTransform[{output_index}]",
                        f"{transform}.offsetParentMatrix",
                        force=True,
                    )
                output_index += 1

    return network


def load_template_usd_rig(
    template_asset_path: str | Path | None = None,
    rig_root: str = DEFAULT_TEMPLATE_RIG_GROUP,
) -> dict[str, Any]:
    """Load the SOMA template USD rig into Maya when it is not already present."""

    from maya import cmds

    if template_asset_path is None:
        template_asset_path = find_repo_template_asset()
        if template_asset_path is None:
            raise ValueError(
                "Could not find assets/SOMA_template_rig.usda. Pass -templateAssetPath explicitly."
            )
    template_asset_path = Path(template_asset_path).resolve()
    if not template_asset_path.exists():
        raise FileNotFoundError(f"Missing SOMA template USD rig: {template_asset_path}")

    if not cmds.pluginInfo("mayaUsdPlugin", query=True, loaded=True):
        cmds.loadPlugin("mayaUsdPlugin", quiet=True)

    if not cmds.objExists(rig_root):
        rig_root = cmds.group(empty=True, name=_safe_node_name(rig_root))
    _set_string_attr(cmds, rig_root, "somaProceduralTemplateAsset", str(template_asset_path))

    existing_root = [
        node
        for node in _descendants(cmds, rig_root)
        if _leaf_name(node) == "Root" and cmds.nodeType(node) == "joint"
    ]
    imported_count = 0
    if not existing_root:
        before = set(cmds.ls(long=True) or [])
        cmds.mayaUSDImport(
            file=str(template_asset_path),
            primPath="/",
            parent=rig_root,
            readAnimData=True,
            unit=True,
            upAxis=True,
        )
        after = set(cmds.ls(long=True) or [])
        imported_count = len(after - before)

    return {
        "rig_root": rig_root,
        "template_asset_path": str(template_asset_path),
        "imported_count": imported_count,
    }


def connect_procedural_node_to_rig(
    definition_path: str | Path | None = None,
    rig_root: str = DEFAULT_TEMPLATE_RIG_GROUP,
    node_name: str = "SOMA_proceduralTransforms_live",
    template_asset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Wire a SOMA procedural node to an imported template rig.

    The node output intentionally mixes local procedural twist rotation with
    world-space interpolated translation. Rotation is composed with the imported
    static world-orientation baseline so Maya mirrors SOMA's joint-orient
    application, while translation is converted through the twist parent inverse
    matrix before driving local ``translate``.
    """

    from maya import cmds

    if definition_path is None:
        definition_path = find_repo_definition()
        if definition_path is None:
            raise ValueError(
                "Could not find assets/SOMA_procedural_transforms.json. "
                "Pass -definitionPath explicitly."
            )
    definition_path = Path(definition_path).resolve()
    definition = load_definition(definition_path)
    if template_asset_path is None:
        attr = f"{rig_root}.somaProceduralTemplateAsset"
        if not cmds.objExists(attr):
            raise ValueError(
                "Could not determine the template USD used by the imported rig. "
                "Pass template_asset_path explicitly."
            )
        template_asset_path = cmds.getAttr(attr)
    template_asset_path = Path(template_asset_path).resolve()
    wanted = set(definition.public_joint_names) | set(definition.twist_joint_names)
    resolved = _resolve_nodes_by_leaf(cmds, rig_root, wanted)

    missing_public = [name for name in definition.public_joint_names if name not in resolved]
    missing_twist = [name for name in definition.twist_joint_names if name not in resolved]
    if missing_public or missing_twist:
        raise ValueError(
            "Could not resolve imported SOMA joints. "
            f"Missing public joints: {missing_public}; missing twist joints: {missing_twist}"
        )

    existing = cmds.ls(node_name, type=NODE_TYPE_NAME) or []
    if existing:
        network = existing[0]
        _set_builtin_string_attr(cmds, network, "definitionPath", str(definition_path))
    else:
        network = create_reference_scene_nodes(
            definition_path=definition_path,
            node_name=node_name,
            create_locators=False,
        )

    previous_world = pose_template_rig_to_bind_pose(
        template_asset_path,
        rig_root=rig_root,
        joint_names=set(definition.public_joint_names),
    )
    try:
        bind_world = {
            joint_name: [
                float(value) for value in cmds.getAttr(f"{resolved[joint_name]}.worldMatrix[0]")
            ]
            for joint_name in definition.public_joint_names
        }
    finally:
        restore_template_rig_pose(previous_world, rig_root=rig_root)

    for index, joint_name in enumerate(definition.public_joint_names):
        joint = resolved[joint_name]
        cmds.connectAttr(f"{joint}.matrix", f"{network}.inputLocalMatrix[{index}]", force=True)
        cmds.connectAttr(
            f"{joint}.worldMatrix[0]",
            f"{network}.inputWorldMatrix[{index}]",
            force=True,
        )
        bind_matrix = _create_hold_matrix(
            cmds,
            f"SOMA_{joint_name}_procBindWorld_hm",
            bind_world[joint_name],
        )
        _set_string_attr(cmds, bind_matrix, "somaDefinitionNode", network)
        _set_string_attr(cmds, bind_matrix, "somaPublicJoint", joint_name)
        cmds.connectAttr(
            f"{bind_matrix}.outMatrix",
            f"{network}.inputBindWorldMatrix[{index}]",
            force=True,
        )

    for index, joint_name in enumerate(definition.twist_joint_names):
        joint = resolved[joint_name]
        _reset_offset_parent_matrix(cmds, joint)
        for pattern in (
            f"SOMA_{joint_name}_procRotateX_dm",
            f"SOMA_{joint_name}_procWorldToOffset_mm",
            f"SOMA_{joint_name}_procWorldToParent_mm",
            f"SOMA_{joint_name}_procWorldToParent_dm",
            f"SOMA_{joint_name}_procBaseline_cm",
            f"SOMA_{joint_name}_procJointOrient_hm",
            f"SOMA_{joint_name}_procParentOrientInv_hm",
            f"SOMA_{joint_name}_procLocalRot_mm",
            f"SOMA_{joint_name}_procLocalRot_dm",
        ):
            for old in cmds.ls(_safe_node_name(pattern)) or []:
                cmds.delete(old)

        rotation_decompose = cmds.createNode(
            "decomposeMatrix",
            name=_safe_node_name(f"SOMA_{joint_name}_procRotateX_dm"),
        )
        _set_string_attr(cmds, rotation_decompose, "somaDefinitionNode", network)
        _set_string_attr(cmds, rotation_decompose, "somaTwistJoint", joint_name)
        _set_int_attr(cmds, rotation_decompose, "somaOutputIndex", index)
        cmds.connectAttr(
            f"{network}.outputTransform[{index}]",
            f"{rotation_decompose}.inputMatrix",
            force=True,
        )
        cmds.connectAttr(f"{rotation_decompose}.outputRotateX", f"{joint}.rotateX", force=True)

    _set_string_attr(cmds, rig_root, "somaProceduralNode", network)
    _set_string_attr(cmds, rig_root, "somaProceduralDefinition", str(definition_path))

    return {
        "rig_root": rig_root,
        "procedural_node": network,
        "definition_path": str(definition_path),
        "public_inputs": len(definition.public_joint_names),
        "twist_outputs": len(definition.twist_joint_names),
    }


def setup_template_rig_scene(
    definition_path: str | Path | None = None,
    template_asset_path: str | Path | None = None,
    rig_root: str = DEFAULT_TEMPLATE_RIG_GROUP,
    node_name: str = "SOMA_proceduralTransforms_live",
) -> dict[str, Any]:
    """Load the template USD rig and wire the live SOMA procedural node."""

    import maya.cmds as cmds

    load_summary = load_template_usd_rig(
        template_asset_path=template_asset_path,
        rig_root=rig_root,
    )
    wire_summary = connect_procedural_node_to_rig(
        definition_path=definition_path,
        rig_root=load_summary["rig_root"],
        node_name=node_name,
        template_asset_path=load_summary["template_asset_path"],
    )
    cmds.select(wire_summary["procedural_node"], replace=True)
    return {**load_summary, **wire_summary}
