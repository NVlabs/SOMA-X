``soma.io``
===========

The ``soma.io`` module provides the file-format boundary for the library.

The main responsibilities are:

- USD mesh and skeleton import/export, and standalone joint animation import
- SOMA NPZ save/load helpers
- convenience export helpers built around ``SOMALayer``

Read a standalone UsdSkel joint animation without a skeleton or mesh:

.. code-block:: python

   from soma.io import load_usd_animation_sequence

   animation = load_usd_animation_sequence("assets/example_animation.usd")
   local_transforms = animation.local_transforms
   times_seconds = animation.time_codes / animation.time_codes_per_second

The reader preserves source joint paths, units, axes, and scale. Unit conversion
and interpretation of the motion for a particular rig belong to the caller.
``load_usd_animation`` retains its existing default-pose return contract.

For SOMA playback, ``load_soma_animation`` maps joints and converts translations
to meters. Set ``world_space=True`` to compose authored joint-path ancestors
before selecting joints. :meth:`soma.body.SOMALayer.load_motion` and
:meth:`soma.hand.SOMAHandLayer.load_motion` convert rotations and root translations
for their respective layers, with absolute or reference-relative output.

.. automodule:: soma.io
   :members: list_usd_meshes, load_usd_mesh, write_usd_mesh, fan_triangulate,
             load_usd_skeleton, load_usd_animation, load_usd_animation_sequence,
             load_soma_animation,
             AnimationUSDData, load_usd_skinning,
             load_rig_from_usd, add_npz_args, load_soma_npz,
             save_soma_usd, save_vertex_animation_usd
