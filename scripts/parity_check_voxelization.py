"""Bit-exact parity check: compiled ``voxel_op`` vs numba ``voxelization_numba``.

Dev-time gate. Run with the compiled ``.pyd`` still present. Once parity is
confirmed, the C++ extension can be removed from the shipped tree.

Usage:
    python scripts/parity_check_voxelization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party" / "PointPillars"
sys.path.insert(0, str(THIRD_PARTY))

# Import both implementations directly. Bypasses voxel_module.py so the parity
# check is independent of which one the wrapper currently points at.
from pointpillars.ops.voxel_op import hard_voxelize as hard_voxelize_cpp  # noqa: E402
from pointpillars.ops.voxelization_numba import hard_voxelize as hard_voxelize_numba  # noqa: E402

# Match PointPillars test-time config (pointpillars/model/pointpillars.py L221-226).
VOXEL_SIZE = [0.16, 0.16, 4.0]
COORS_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
MAX_POINTS = 32
MAX_VOXELS = 40000  # test-mode cap


def load_input() -> torch.Tensor:
    bin_path = REPO_ROOT / "0000000001.bin"
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # Apply the same range filter verify_pointpillars.py uses, so we hit the
    # same code path the model actually feeds.
    rng = COORS_RANGE
    mask = (
        (pts[:, 0] > rng[0]) & (pts[:, 0] < rng[3])
        & (pts[:, 1] > rng[1]) & (pts[:, 1] < rng[4])
        & (pts[:, 2] > rng[2]) & (pts[:, 2] < rng[5])
    )
    return torch.from_numpy(pts[mask].copy())


def run(impl, points: torch.Tensor):
    voxels = points.new_zeros(size=(MAX_VOXELS, MAX_POINTS, points.size(1)))
    coors = points.new_zeros(size=(MAX_VOXELS, 3), dtype=torch.int)
    npts = points.new_zeros(size=(MAX_VOXELS,), dtype=torch.int)
    voxel_num = impl(
        points, voxels, coors, npts,
        VOXEL_SIZE, COORS_RANGE,
        MAX_POINTS, MAX_VOXELS, 3, True,
    )
    return int(voxel_num), voxels[:voxel_num], coors[:voxel_num], npts[:voxel_num]


def main() -> int:
    pts = load_input()
    print(f"input points: {len(pts)}")

    n_cpp, vox_cpp, coors_cpp, npts_cpp = run(hard_voxelize_cpp, pts)
    n_nb, vox_nb, coors_nb, npts_nb = run(hard_voxelize_numba, pts)

    print(f"voxel_num: cpp={n_cpp} numba={n_nb}")
    assert n_cpp == n_nb, f"voxel_num differs: {n_cpp} vs {n_nb}"

    assert torch.equal(coors_cpp, coors_nb), "coors differ"
    assert torch.equal(npts_cpp, npts_nb), "num_points_per_voxel differ"
    # voxels are float32 copies of input rows — bit-exact equality expected.
    assert torch.equal(vox_cpp, vox_nb), "voxels differ"

    print("PARITY OK — voxel_num, coors, num_points_per_voxel, voxels all bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
