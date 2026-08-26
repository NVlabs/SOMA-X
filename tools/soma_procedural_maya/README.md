# SOMA Procedural Maya Reference

This directory contains a reference Maya Python plug-in for consuming
`assets/SOMA_procedural_transforms.json`. It is intentionally sidecar-driven:
the plug-in validates the JSON, compiles the named sparse parameter rows into
dense NumPy matrices, and registers a Maya dependency node that evaluates the
procedural twist transforms from public SOMA joint inputs.

## Install

From the repository root:

```bash
.venv/bin/python tools/soma_procedural_maya/setup_maya_plugin.py --force
```

The setup script writes a Maya module file named `SOMAProceduralMaya.mod` to the
current user's Maya modules directory. It points Maya at this checkout, adds this
directory to `PYTHONPATH`, and adds `plug-ins/` to `MAYA_PLUG_IN_PATH`.

Use `--dry-run` to inspect the target path without writing.

## Load In Maya

```python
import maya.cmds as cmds

cmds.loadPlugin("soma_procedural_maya_plugin.py")
node = cmds.somaCreateProceduralRigReference(
    definitionPath="/path/to/SOMA-X/assets/SOMA_procedural_transforms.json",
)
print(node)
```

The command returns the created `somaProceduralTransforms` node. By default it
also creates one locator per generated twist joint, connects each locator's
`offsetParentMatrix` from `outputTransform[index]`, and adds attributes naming
the source segment, axis, sign, reverse compensation flag, and output index.
Pass `noLocators=True` from Python or `-noLocators` from MEL to create only the
node.

## Load And Wire The Template Rig

For the checked-in template USD rig, use the setup mode so Maya imports the rig
first and then wires the imported public joints and twist joints:

```python
import maya.cmds as cmds

cmds.loadPlugin("soma_procedural_maya_plugin.py")
node = cmds.somaCreateProceduralRigReference(
    setupTemplateRig=True,
    templateAssetPath="/path/to/SOMA-X/assets/SOMA_template_rig.usda",
    definitionPath="/path/to/SOMA-X/assets/SOMA_procedural_transforms.json",
    nodeName="SOMA_proceduralTransforms_live",
)
print(node)
```

The setup path imports the USD under `SOMA_templateRig_PROC_GRP` when it is not
already present, connects all public joints to the node inputs, and connects all
generated twist joints through matrix helpers. Before wiring the bind inputs,
setup temporarily poses the public skeleton to USD `bindTransforms`, records
those world matrices, and restores the imported rest/T pose. The template USD
must author both `restTransforms` and `bindTransforms`.

The node output combines local
procedural twist rotation with world-space interpolated translation, so setup
wraps the output rotation with the imported twist joint and parent static world
orientations to match SOMA's joint-orient application, and drives local
`translate` from the output translation after parent-space conversion.

## Node Attributes

The node expects main SOMA joint matrices in the same order as
`public_rig_derivation.main_joint_names` in the JSON sidecar.

- `definitionPath`: path to `SOMA_procedural_transforms.json`. If empty, the
  node walks parent directories looking for `assets/SOMA_procedural_transforms.json`.
- `inputLocalMatrix[index]`: local transform matrix for main joint `index`.
  The rotation block drives the source twist channel extraction.
- `inputWorldMatrix[index]`: world transform matrix for main joint `index`.
  The translation block is multiplied by the sidecar translation parameter
  matrix. If left unconnected, identity matrices are used.
- `inputBindWorldMatrix[index]`: world-space USD skin bind transform for main
  joint `index`. Aligned twist is zero relative to this pose, independently of
  the imported rest/T pose.
- `outputTransform[index]`: generated transform matrix for procedural twist
  joint `index`, ordered by the sidecar segment `twist_joints` list.

Typical wiring:

```python
for index, joint in enumerate(public_joint_names):
    cmds.connectAttr(f"{joint}.matrix", f"{node}.inputLocalMatrix[{index}]", force=True)
    cmds.connectAttr(f"{joint}.worldMatrix[0]", f"{node}.inputWorldMatrix[{index}]", force=True)

for index, joint in enumerate(twist_joint_names):
    joint_orient = cmds.createNode("holdMatrix")
    parent_orient_inverse = cmds.createNode("holdMatrix")
    # Set joint_orient.inMatrix to the twist joint's imported world rotation.
    # Set parent_orient_inverse.inMatrix to the inverse imported world rotation
    # of the twist joint's parent.
    rotation_matrix = cmds.createNode("multMatrix")
    rotation_decompose = cmds.createNode("decomposeMatrix")
    cmds.connectAttr(
        f"{joint_orient}.outMatrix",
        f"{rotation_matrix}.matrixIn[0]",
        force=True,
    )
    cmds.connectAttr(
        f"{node}.outputTransform[{index}]",
        f"{rotation_matrix}.matrixIn[1]",
        force=True,
    )
    cmds.connectAttr(
        f"{parent_orient_inverse}.outMatrix",
        f"{rotation_matrix}.matrixIn[2]",
        force=True,
    )
    cmds.connectAttr(
        f"{rotation_matrix}.matrixSum",
        f"{rotation_decompose}.inputMatrix",
        force=True,
    )
    cmds.connectAttr(f"{rotation_decompose}.outputRotate", f"{joint}.rotate", force=True)

    translation_matrix = cmds.createNode("multMatrix")
    translation_decompose = cmds.createNode("decomposeMatrix")
    cmds.connectAttr(
        f"{node}.outputTransform[{index}]",
        f"{translation_matrix}.matrixIn[0]",
        force=True,
    )
    cmds.connectAttr(
        f"{parent}.worldInverseMatrix[0]",
        f"{translation_matrix}.matrixIn[1]",
        force=True,
    )
    cmds.connectAttr(
        f"{translation_matrix}.matrixSum",
        f"{translation_decompose}.inputMatrix",
        force=True,
    )
    cmds.connectAttr(f"{translation_decompose}.outputTranslate", f"{joint}.translate", force=True)
```

The implementation uses NumPy inside Maya and mirrors the SOMA runtime's
parameter-matrix path: local twist channels are extracted from main joint
rotations, multiplied by the dense rotation parameter matrix, emitted as
axis-angle rotation matrices, and paired with translations produced by the dense
translation parameter matrix.

Compared with the Maya `twistReader` node, this reference matches the
high-level segment topology: forearm and shin twist are driven by the distal
hand or foot, while upper arm and thigh use reverse start/end compensation. It
intentionally differs in representation: the source of truth is the JSON
sidecar, sparse rows are compiled into NumPy matrices, and generated transforms
include translations. The Maya node computes twist through bind-pose aligned
virtual drivers from world matrices and emits per-joint `rotateX` values; this
reference emits transform matrices from the sidecar parameter matrices.

## Command Flags

- `-definitionPath` / `definitionPath`: path to `SOMA_procedural_transforms.json`.
  If omitted, the plug-in walks parent directories looking for
  `assets/SOMA_procedural_transforms.json`.
- `-nodeName` / `nodeName`: name for the created network node.
- `-noLocators` / `noLocators`: skip per-twist locator creation.
- `-setupTemplateRig` / `setupTemplateRig`: import and wire
  `SOMA_template_rig.usda` before returning the node.
- `-templateAssetPath` / `templateAssetPath`: explicit path to
  `SOMA_template_rig.usda`; implies `setupTemplateRig`.
- `-rigRoot` / `rigRoot`: group to use when loading or resolving the template
  rig. Defaults to `SOMA_templateRig_PROC_GRP`.

## Reference Scope

This plug-in is a reference implementation for the portable sidecar. It
exercises name resolution, extraction-policy validation, sparse matrix loading, and live
dependency-graph evaluation without depending on the SOMA Python runtime or
Torch inside Maya.
