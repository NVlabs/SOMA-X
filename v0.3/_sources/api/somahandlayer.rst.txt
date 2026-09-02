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
remaining 24 joints articulate the fingers. Outputs contain wrist-local
vertices, joints, and transforms in the requested output unit.

See :doc:`../hand_data_assets` for the checked-in identity and pose-PCA asset
contract and the MANO setup requirements.

.. automodule:: soma.hand
   :no-members:

.. autoclass:: soma.hand.SOMAHandLayer
   :members: default_skin_mesh_name, num_shape_components, prepare_identity, pose, forward
   :show-inheritance:

.. autoclass:: soma.hand.SOMAHandPoseOutput
   :members:
