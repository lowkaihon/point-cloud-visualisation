"""Post-install smoke test: confirms the numba voxelizer produces sane output.

Lighter than the dev-time parity check — does not require the compiled
``.pyd``. Designed to surface a silently-broken port (e.g., empty output,
wrong dtype, clipping at max_voxels) on a fresh evaluator install.

Usage:
    python scripts/smoke_test_voxelization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party" / "PointPillars"
sys.path.insert(0, str(THIRD_PARTY))

from pointpillars.ops.voxelization_numba import hard_voxelize  # noqa: E402

VOXEL_SIZE = [0.16, 0.16, 4.0]
COORS_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
MAX_POINTS = 32
MAX_VOXELS = 40000


def main() -> int:
    bin_path = REPO_ROOT / "data" / "0000000001.bin"
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    rng = COORS_RANGE
    mask = (
        (pts[:, 0] > rng[0]) & (pts[:, 0] < rng[3])
        & (pts[:, 1] > rng[1]) & (pts[:, 1] < rng[4])
        & (pts[:, 2] > rng[2]) & (pts[:, 2] < rng[5])
    )
    points = torch.from_numpy(pts[mask].copy())

    voxels = points.new_zeros(size=(MAX_VOXELS, MAX_POINTS, points.size(1)))
    coors = points.new_zeros(size=(MAX_VOXELS, 3), dtype=torch.int)
    npts = points.new_zeros(size=(MAX_VOXELS,), dtype=torch.int)
    voxel_num = hard_voxelize(
        points, voxels, coors, npts,
        VOXEL_SIZE, COORS_RANGE,
        MAX_POINTS, MAX_VOXELS, 3, True,
    )

    assert voxel_num > 0, "voxelizer returned 0 voxels — silently broken?"
    assert voxel_num < MAX_VOXELS, (
        f"voxel_num={voxel_num} hit MAX_VOXELS={MAX_VOXELS}; "
        "check ROI / grid size — may be clipping"
    )
    assert voxels.dtype == torch.float32
    assert coors.dtype == torch.int32
    assert npts.dtype == torch.int32

    coors_xyz = coors[:voxel_num].flip(-1)  # (z,y,x) -> (x,y,z)
    # Voxel (x, y, z) indices reconstructed as world centroids should sit
    # inside the ROI. Use mid-z of the voxel as a robust check.
    z_centers = (
        coors_xyz[:, 2].float() * VOXEL_SIZE[2] + COORS_RANGE[2] + VOXEL_SIZE[2] / 2
    )
    median_z = float(z_centers.median())
    assert COORS_RANGE[2] <= median_z <= COORS_RANGE[5], (
        f"median voxel z={median_z:.2f} outside ROI [{COORS_RANGE[2]}, {COORS_RANGE[5]}]"
    )

    npts_max = int(npts[:voxel_num].max())
    assert 1 <= npts_max <= MAX_POINTS, (
        f"npts_max={npts_max} outside [1, {MAX_POINTS}]"
    )

    print(
        f"OK voxel_num={voxel_num} median_z={median_z:+.2f} "
        f"npts_max={npts_max} (cap={MAX_POINTS})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
