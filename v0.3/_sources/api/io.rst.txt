``soma.io``
===========

The ``soma.io`` module provides the file-format boundary for the library.

The main responsibilities are:

- USD mesh and skeleton import/export
- SOMA NPZ save/load helpers
- convenience export helpers built around ``SOMALayer``

.. automodule:: soma.io
   :members: list_usd_meshes, load_usd_mesh, write_usd_mesh, fan_triangulate,
             load_usd_skeleton, load_usd_animation, load_usd_skinning,
             load_rig_from_usd, add_npz_args, load_soma_npz,
             save_soma_usd, save_vertex_animation_usd
