# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests verifying SOMAHandLayer works under PyTorch DataLoader + multiprocessing.

Mirrors tests/test_dataloader.py (which covers SOMALayer). Ensures warp
initialization survives fork() in DataLoader workers and scans worker stderr
for "Warp CUDA error 3" — the symptom Lin Duan hit when moving right→left
hand mirroring into __getitem__.

Also exercises the compute_joint_world_transforms fast path so that the
joints-only FK used by the pose-mirroring pipeline is proven fork-safe on
warp-mode CPU layers.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"

NUM_HAND_JOINTS = 25


def _assets_available():
    return (
        ASSETS_DIR.is_dir()
        and (ASSETS_DIR / "SOMA_neutral.npz").is_file()
        and (ASSETS_DIR / "SOMAHand.npz").is_file()
    )


class HandPoseDataset(Dataset):
    """Workers only load CPU tensors — SOMAHandLayer runs in the main process."""

    def __init__(self, id_coeffs_dim, size=4):
        self.poses = torch.zeros(size, NUM_HAND_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.transl = torch.zeros(size, 3)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx], self.identity_coeffs[idx], self.transl[idx]


class HandDataset(Dataset):
    """Dataset that initializes SOMAHandLayer once at construction (inherited via fork)."""

    def __init__(self, data_root, hand_type="right", size=4):
        from soma import SOMAHandLayer

        self.data_root = data_root
        self._layer = SOMAHandLayer(
            data_root=data_root,
            hand_type=hand_type,
            device="cpu",
            identity_model_type="soma",
            mode="warp",
        )
        K = self._layer.identity_model.num_identity_coeffs
        self.poses = torch.zeros(size, NUM_HAND_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, K)
        self.transl = torch.zeros(size, 3)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx].unsqueeze(0)
        coeffs = self.identity_coeffs[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(
                poses=pose,
                identity_coeffs=coeffs,
                global_translation=transl,
            )
        return {
            "vertices": out["vertices"].squeeze(0),
            "joints": out["joints"].squeeze(0),
            "pose": pose.squeeze(0),
            "coeffs": coeffs.squeeze(0),
            "transl": transl.squeeze(0),
        }


class _LazyHandDataset(Dataset):
    """Dataset that initializes SOMAHandLayer lazily inside __getitem__ (fresh in each worker)."""

    def __init__(self, data_root, id_coeffs_dim, hand_type="right", size=4):
        self.data_root = data_root
        self.hand_type = hand_type
        self.poses = torch.zeros(size, NUM_HAND_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.transl = torch.zeros(size, 3)
        self._layer = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        if self._layer is None:
            from soma import SOMAHandLayer

            self._layer = SOMAHandLayer(
                data_root=self.data_root,
                hand_type=self.hand_type,
                device="cpu",
                identity_model_type="soma",
                mode="warp",
            )
        pose = self.poses[idx].unsqueeze(0)
        coeffs = self.identity_coeffs[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(
                poses=pose,
                identity_coeffs=coeffs,
                global_translation=transl,
            )
        return out["vertices"].squeeze(0), out["joints"].squeeze(0)


class _WorkerInitHandDataset(Dataset):
    """Dataset where SOMAHandLayer is injected by worker_init_fn."""

    def __init__(self, id_coeffs_dim, size=4):
        self.poses = torch.zeros(size, NUM_HAND_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.transl = torch.zeros(size, 3)
        self._layer = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx].unsqueeze(0)
        coeffs = self.identity_coeffs[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(
                poses=pose,
                identity_coeffs=coeffs,
                global_translation=transl,
            )
        return {
            "vertices": out["vertices"].squeeze(0),
            "joints": out["joints"].squeeze(0),
            "pose": pose.squeeze(0),
            "coeffs": coeffs.squeeze(0),
            "transl": transl.squeeze(0),
        }


def _hand_worker_init(worker_id):
    """worker_init_fn: initialize SOMAHandLayer once per worker process."""
    info = torch.utils.data.get_worker_info()
    from soma import SOMAHandLayer

    info.dataset._layer = SOMAHandLayer(
        data_root=str(ASSETS_DIR),
        hand_type="right",
        device="cpu",
        identity_model_type="soma",
        mode="warp",
    )


class FkOnlyHandDataset(Dataset):
    """Dataset exercising pose(fk_only=True) in forked workers.

    This is Lin's actual use case: the hand layer built on CPU, warp mode
    active, but the fast FK path skips all LBS/warp kernel launches, so
    workers should neither crash nor emit "Warp CUDA error 3".
    """

    def __init__(self, data_root, hand_type="right", size=4):
        from soma import SOMAHandLayer

        self._layer = SOMAHandLayer(
            data_root=data_root,
            hand_type=hand_type,
            device="cpu",
            identity_model_type="soma",
            mode="warp",
        )
        K = self._layer.identity_model.num_identity_coeffs
        self.poses = torch.zeros(size, NUM_HAND_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, K)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx].unsqueeze(0)
        coeffs = self.identity_coeffs[idx].unsqueeze(0)
        self._layer.prepare_identity(coeffs, global_scale=1.0)
        with torch.no_grad():
            out = self._layer.pose(poses=pose, fk_only=True)
        return out["transforms"].squeeze(0)


@pytest.mark.slow
@pytest.mark.multiprocess
@pytest.mark.asset_heavy
class TestSomaHandLayerDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _assets_available():
            raise unittest.SkipTest(
                "Assets not found. Run `git lfs pull` to fetch SOMAHand.npz + SOMA_neutral.npz."
            )
        cls.data_root = str(ASSETS_DIR)
        layer = cls._make_layer_static(cls.data_root, "cpu")
        cls.id_coeffs_dim = layer.identity_model.num_identity_coeffs

    @staticmethod
    def _make_layer_static(data_root, device, hand_type="right"):
        from soma import SOMAHandLayer

        return SOMAHandLayer(
            data_root=data_root,
            hand_type=hand_type,
            device=device,
            identity_model_type="soma",
            mode="warp",
        )

    def _make_layer(self, device="cpu", hand_type="right"):
        return self._make_layer_static(self.data_root, device, hand_type)

    def _assert_output_shapes(self, vertices, joints, batch_size):
        self.assertEqual(vertices.dim(), 3)
        self.assertEqual(vertices.shape[0], batch_size)
        self.assertEqual(vertices.shape[2], 3)
        self.assertEqual(joints.shape, (batch_size, NUM_HAND_JOINTS, 3))

    def test_no_workers_baseline(self):
        """Sanity: single-process DataLoader, warp ops work."""
        layer = self._make_layer("cpu")
        dataset = HandPoseDataset(self.id_coeffs_dim, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        for poses, coeffs, transl in loader:
            with torch.no_grad():
                out = layer(poses=poses, identity_coeffs=coeffs, global_translation=transl)
            self.assertIn("vertices", out)
            self.assertIn("joints", out)
            self._assert_output_shapes(out["vertices"], out["joints"], 2)

    def test_multi_worker_warp_in_main_process(self):
        """Safe pattern: workers only load tensors; warp called only in main process."""
        layer = self._make_layer("cpu")
        dataset = HandPoseDataset(self.id_coeffs_dim, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)

        for poses, coeffs, transl in loader:
            with torch.no_grad():
                out = layer(poses=poses, identity_coeffs=coeffs, global_translation=transl)
            self._assert_output_shapes(out["vertices"], out["joints"], 2)

    def test_multi_worker_init_at_construction(self):
        """SOMAHandLayer constructed in main, inherited by forked workers."""
        import multiprocessing

        if multiprocessing.get_start_method() != "fork":
            self.skipTest(
                "test requires fork-based DataLoader workers; "
                f"current start method is {multiprocessing.get_start_method()!r}"
            )
        dataset = HandDataset(self.data_root, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        ref_layer = self._make_layer("cpu")

        with tempfile.TemporaryFile() as _tmp:
            _saved = os.dup(2)
            os.dup2(_tmp.fileno(), 2)
            try:
                for data in loader:
                    batch_size = data["vertices"].shape[0]
                    with torch.no_grad():
                        out = ref_layer(
                            poses=data["pose"],
                            identity_coeffs=data["coeffs"],
                            global_translation=data["transl"],
                        )
                    diff_joints = (data["joints"] - out["joints"]).abs().max()
                    self.assertLess(diff_joints, 1e-3)
                    self._assert_output_shapes(data["vertices"], data["joints"], batch_size)
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            _tmp.seek(0)
            _stderr = _tmp.read().decode("utf-8", errors="replace")

        self.assertNotIn(
            "Warp CUDA error 3",
            _stderr,
            "CUDA error 3 appeared in worker stderr — fork hook may not be working",
        )

    def test_multi_worker_lazy_init_in_worker(self):
        """SOMAHandLayer constructed fresh inside each forked worker."""
        dataset = _LazyHandDataset(self.data_root, self.id_coeffs_dim, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)

        for vertices, joints in loader:
            batch_size = vertices.shape[0]
            self._assert_output_shapes(vertices, joints, batch_size)

    def test_multi_worker_worker_init_fn(self):
        """Recommended pattern: SOMAHandLayer initialized once per worker via worker_init_fn."""
        dataset = _WorkerInitHandDataset(self.id_coeffs_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            worker_init_fn=_hand_worker_init,
        )
        ref_layer = self._make_layer("cpu")

        with tempfile.TemporaryFile() as _tmp:
            _saved = os.dup(2)
            os.dup2(_tmp.fileno(), 2)
            try:
                for data in loader:
                    batch_size = data["vertices"].shape[0]
                    with torch.no_grad():
                        out = ref_layer(
                            poses=data["pose"],
                            identity_coeffs=data["coeffs"],
                            global_translation=data["transl"],
                        )
                    diff_joints = (data["joints"] - out["joints"]).abs().max()
                    self.assertLess(diff_joints, 1e-3)
                    self._assert_output_shapes(data["vertices"], data["joints"], batch_size)
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            _tmp.seek(0)
            _stderr = _tmp.read().decode("utf-8", errors="replace")

        self.assertNotIn(
            "Warp CUDA error 3",
            _stderr,
            "CUDA error 3 appeared in worker stderr — fork hook may not be working",
        )

    def test_spawn_context(self):
        """spawn multiprocessing context: fresh processes, no fork state inheritance."""
        layer = self._make_layer("cpu")
        dataset = HandPoseDataset(self.id_coeffs_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )

        for poses, coeffs, transl in loader:
            with torch.no_grad():
                out = layer(poses=poses, identity_coeffs=coeffs, global_translation=transl)
            self._assert_output_shapes(out["vertices"], out["joints"], 2)

    def test_cuda_spawn_context(self):
        """CUDA-safe pattern via spawn context. Skipped if no GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        layer = self._make_layer("cuda")
        dataset = HandPoseDataset(self.id_coeffs_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )

        for poses, coeffs, transl in loader:
            poses = poses.cuda()
            coeffs = coeffs.cuda()
            transl = transl.cuda()
            with torch.no_grad():
                out = layer(poses=poses, identity_coeffs=coeffs, global_translation=transl)
            self._assert_output_shapes(out["vertices"], out["joints"], 2)

    # ----------------------------------------------------------------------
    # Joints-only fast path under fork — Lin's actual use case.
    # ----------------------------------------------------------------------

    def test_multi_worker_fk_only_fork(self):
        """pose(fk_only=True) inside forked workers: no warp crash, no CUDA error 3.

        This is the DataLoader pattern for the camera-space hand mirror
        pipeline — the pose-mirror decomposition in
        The camera-space mirror demo only needs joint
        transforms. Skipping LBS means warp kernels never launch, so the
        usual fork+warp+CUDA concerns don't apply.
        """
        import multiprocessing

        if multiprocessing.get_start_method() != "fork":
            self.skipTest(
                "test requires fork-based DataLoader workers; "
                f"current start method is {multiprocessing.get_start_method()!r}"
            )
        dataset = FkOnlyHandDataset(self.data_root, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)

        with tempfile.TemporaryFile() as _tmp:
            _saved = os.dup(2)
            os.dup2(_tmp.fileno(), 2)
            try:
                for T in loader:
                    batch_size = T.shape[0]
                    self.assertEqual(T.shape, (batch_size, NUM_HAND_JOINTS, 4, 4))
                    self.assertTrue(torch.isfinite(T).all())
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            _tmp.seek(0)
            _stderr = _tmp.read().decode("utf-8", errors="replace")

        self.assertNotIn(
            "Warp CUDA error 3",
            _stderr,
            "CUDA error 3 appeared in worker stderr — fk_only path "
            "unexpectedly launched a warp kernel",
        )


if __name__ == "__main__":
    unittest.main()
