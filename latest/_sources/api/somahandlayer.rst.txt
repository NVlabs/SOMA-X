``SOMAHandLayer``
=================

``SOMAHandLayer`` is the hand-only counterpart to ``SOMALayer``. It provides a
25-joint parametric hand in wrist-local space for left and right hands at mid,
low, and extra-low LODs.

It combines:

- the native SOMA hand identity PCA, or a user-supplied MANO/MHR backend
- identity-dependent skeleton fitting
- Warp-accelerated or dense linear blend skinning
- an optional bind-relative articulation-pose PCA

Use ``prepare_identity(...)`` when identity changes and ``pose(...)`` for each
new pose. ``forward(...)`` is the one-call convenience wrapper.

The pose tensor has shape ``(B, 25, 3)`` in axis-angle form, or
``(B, 25, 3, 3)`` when ``pose2rot=False``. Joint zero is the wrist; the
remaining 24 joints articulate the fingers. Outputs include the supplied wrist
rotation and global translation, in the requested output unit.

Use constructor ``reference_pose=`` for a reusable default, or pass it to
``pose()`` / ``forward()`` for a one-call override.
See :doc:`../data_assets` for the shared body/hand lookup API and frames.

See :doc:`../hand_data_assets` for the checked-in identity and pose-PCA asset
contract and the MANO setup requirements.

Load full-body or hand-only USD motion with the same signature as the body layer:

.. code-block:: python

   poses, translation, fps = hand_layer.load_motion(path, absolute_pose=True)
   hand_layer.prepare_identity(identity_coeffs)
   out = hand_layer.pose(
       poses, pose2rot=False, absolute_pose=True, global_translation=translation,
   )

``load_motion(path, *, absolute_pose=False, reference_pose=None)`` returns all
25 rotation matrices, wrist world translation, and frame rate. The wrist's
animated ancestors are composed for full-body input. Hand-only input supplies
the wrist world transform directly. Do not apply another wrist transform.
Both formats need this hand's joint names and authored hierarchy; common leading
path prefixes may be absent, but gaps within an authored ancestor chain are errors.

Reference selection follows ``pose()``: explicit tensor or lookup dictionary,
then the constructor default, then the template T-pose. Absolute output bypasses
the default and rejects an explicit reference. Use the same options for loading
and posing. Global wrist rotation affects skinning, but not corrective prediction.

.. automodule:: soma.hand
   :no-members:

.. autoclass:: soma.hand.SOMAHandLayer
   :members: default_skin_mesh_name, num_shape_components, list_reference_poses, get_reference_pose, convert_reference, load_motion, prepare_identity, pose, forward
   :show-inheritance:

.. autoclass:: soma.hand.SOMAHandPoseOutput
   :members:
