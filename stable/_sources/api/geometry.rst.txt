``soma.geometry``
=================

The ``soma.geometry`` package contains the lower-level geometry, rigging, interpolation,
and Warp-accelerated kernels used internally by ``SOMALayer`` and ``PoseInversion``.

This section is useful when you want to build custom fitting or retargeting workflows
without going through the higher-level layer wrappers.

Transform and rig utilities
---------------------------

.. automodule:: soma.geometry.transforms
   :members: compute_covariance, kabsch, newton_schulz, rodrigues_rotation,
             align_vectors, SE3_from_Rt, SE3_inverse, matrix_to_rotvec,
             rotvec_to_matrix, rotation_6d_to_matrix

.. automodule:: soma.geometry.rig_utils
   :members: joint_world_to_local, joint_local_to_world, compute_skeleton_levels,
             joint_local_to_world_levelorder, precompute_joint_orient,
             apply_joint_orient_local, remove_joint_orient_local,
             get_joint_children_ids, get_joint_descendents,
             get_body_part_vertex_ids, PoseMirror_SOMA, PoseMirror_MHR

Skinning and skeleton fitting
-----------------------------

.. automodule:: soma.geometry.lbs
   :members: batch_rodrigues, lbs

.. automodule:: soma.geometry.batched_skinning
   :members: topk_skinning

.. automodule:: soma.geometry.skeleton_transfer
   :members: SkeletonTransfer

Interpolation and mesh utilities
--------------------------------

.. automodule:: soma.geometry.barycentric_interp
   :members: fabricate_tet, compute_barycentric_coords_3d,
             barycentric_interpolation, BarycentricInterpolator

.. automodule:: soma.geometry.interpolate
   :members: RadialBasisFunction

.. automodule:: soma.geometry.laplacian
   :members: cotangent_weights, build_cotangent_laplacian,
             build_uniform_laplacian, power_laplacian, LaplacianMesh

Warp backends
-------------

.. automodule:: soma.geometry.lbs_warp
   :members: linear_blend_skinning

.. automodule:: soma.geometry.align_vectors_warp
   :members: rodrigues_rotation_warp, align_vectors_warp, parallel_rodrigues_kabsch_warp

.. automodule:: soma.geometry.chamfer_warp
   :members: ChamferLoss
   :exclude-members: ChamferBatchedFunction

.. automodule:: soma.geometry.fused_refit_warp
   :members: fused_refit_level
