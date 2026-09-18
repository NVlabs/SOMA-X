``SOMALayer``
=============

``SOMALayer`` is the full-body entry point for the library: a 78-joint parametric human model.

It combines:

- a selected identity backend: ``mhr`` (default), ``soma``, ``smpl`` / ``smplh`` / ``smplx``, ``anny``, or ``garment``
- identity-dependent skeleton fitting
- Warp-accelerated or dense linear blend skinning
- optional pose-dependent corrective vertex offsets

Two-phase API:

1. ``prepare_identity(...)`` when the identity changes
2. ``pose(...)`` for each new pose

``forward(...)`` is the one-call convenience wrapper.

Historical and custom reference poses
-------------------------------------

Set a default reference when creating the layer, or override it in a pose call:

.. code-block:: python

   layer = SOMALayer(
       reference_pose={"version": "v0.1.0"},
   )
   layer.prepare_identity(identity_coeffs)
   out = layer.pose(rotations, pose2rot=False)
   out = layer.pose(rotations, pose2rot=False, reference_pose=another_reference)

``forward()`` uses the same default and override rules. Omitted or ``None``
references inherit the constructor default. ``absolute_pose=True`` bypasses it;
an explicit call-time reference with ``absolute_pose=True`` is an error.

See :doc:`../data_assets` for lookup rules, reference shapes and coordinate frames.

See the module overview below for the full parameter reference (joint grouping,
per-backend identity dims, ``scale_params`` layout, units). The class reference
follows.

.. automodule:: soma.body.soma
   :no-members:

.. autoclass:: soma.body.SOMALayer
   :members: default_skin_mesh_name, num_shape_components, prepare_identity, pose, forward, list_reference_poses, get_reference_pose, convert_reference
   :show-inheritance:
