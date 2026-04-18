"""Pure-Python (numba-jit) replacement for the C++ ``voxel_op`` extension.

Drop-in for ``pointpillars.ops.voxel_op.hard_voxelize``. Matches the CPU kernel
in ``voxelization/voxelization_cpu.cpp`` line-for-line so that downstream
PointPillars output is bit-identical.

Eliminates the only remaining C++ build step in the vendored PointPillars
tree, so install collapses to ``pip install`` cross-platform.
"""
from __future__ import annotations

import numba
import numpy as np
import torch


@numba.njit(cache=True)
def _hard_voxelize_kernel(
    points,                # (N, C) float32, input points
    voxels,                # (max_voxels, max_points, C) float32, written in place
    coors,                 # (max_voxels, 3) int32, written in place; stored (z, y, x)
    num_points_per_voxel,  # (max_voxels,) int32, written in place
    voxel_size,            # (3,) float32, [sx, sy, sz]
    coors_range,           # (6,) float32, [xmin, ymin, zmin, xmax, ymax, zmax]
    grid_size,             # (3,) int32, [nx, ny, nz]
    max_points,
    max_voxels,
):
    nx = grid_size[0]
    ny = grid_size[1]
    nz = grid_size[2]

    # Dense lookup grid: -1 == unassigned. Indexed [z, y, x] to match the C++
    # coor_to_voxelidx layout (voxelization_cpu.cpp line 130).
    coor_to_voxelidx = np.full((nz, ny, nx), -1, dtype=np.int32)

    N = points.shape[0]
    C = points.shape[1]
    voxel_num = 0

    for i in range(N):
        # Per-axis floor; reject if any axis is outside the grid.
        cx = int(np.floor((points[i, 0] - coors_range[0]) / voxel_size[0]))
        if cx < 0 or cx >= nx:
            continue
        cy = int(np.floor((points[i, 1] - coors_range[1]) / voxel_size[1]))
        if cy < 0 or cy >= ny:
            continue
        cz = int(np.floor((points[i, 2] - coors_range[2]) / voxel_size[2]))
        if cz < 0 or cz >= nz:
            continue

        voxelidx = coor_to_voxelidx[cz, cy, cx]
        if voxelidx == -1:
            if voxel_num >= max_voxels:
                continue
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[cz, cy, cx] = voxelidx
            # coors stored in (z, y, x) order; caller flips to (x, y, z).
            coors[voxelidx, 0] = cz
            coors[voxelidx, 1] = cy
            coors[voxelidx, 2] = cx

        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            for k in range(C):
                voxels[voxelidx, num, k] = points[i, k]
            num_points_per_voxel[voxelidx] = num + 1

    return voxel_num


def _to_np_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def hard_voxelize(
    points,
    voxels,
    coors,
    num_points_per_voxel,
    voxel_size,
    coors_range,
    max_points,
    max_voxels,
    NDim,
    deterministic,
):
    """Drop-in for the compiled ``voxel_op.hard_voxelize`` (CPU, NDim=3).

    ``deterministic`` is ignored: the numba kernel is single-threaded and
    deterministic by construction.
    """
    assert NDim == 3
    assert points.device.type == "cpu"
    assert points.dtype == torch.float32

    vs = _to_np_f32(voxel_size)
    cr = _to_np_f32(coors_range)

    # Grid sizing matches voxelization_cpu.cpp line 123: round(), not ceil.
    grid_size = np.array(
        [
            round((cr[3] - cr[0]) / vs[0]),
            round((cr[4] - cr[1]) / vs[1]),
            round((cr[5] - cr[2]) / vs[2]),
        ],
        dtype=np.int32,
    )

    voxel_num = _hard_voxelize_kernel(
        points.numpy(),
        voxels.numpy(),
        coors.numpy(),
        num_points_per_voxel.numpy(),
        vs,
        cr,
        grid_size,
        int(max_points),
        int(max_voxels),
    )
    return int(voxel_num)
