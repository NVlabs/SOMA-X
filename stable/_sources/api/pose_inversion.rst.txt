``PoseInversion``
=================

``PoseInversion`` is the inverse-fitting utility used by the conversion tools in this
repository.

It fits SOMA-compatible skeleton rotations from posed vertices and supports:

- analytical inverse-LBS refinement
- optional Lie algebra Gauss-Newton refinement
- optional autograd-based refinement through FK + LBS

This makes it the key API for conversions such as SMPL-to-SOMA and MHR-to-SOMA.

.. autoclass:: soma.pose_inversion.PoseInversion
   :members: joint_names, transfer_to_soma, prepare_identity, fit
   :show-inheritance:
