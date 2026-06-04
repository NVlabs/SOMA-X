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

See the module overview below for the full parameter reference (joint grouping,
per-backend identity dims, ``scale_params`` layout, units). The class reference
follows.

.. automodule:: soma.soma
   :no-members:

.. autoclass:: soma.soma.SOMALayer
   :members: default_skin_mesh_name, num_shape_components, prepare_identity, pose, forward
   :show-inheritance:
