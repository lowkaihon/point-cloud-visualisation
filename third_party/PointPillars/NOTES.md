# Vendoring notes

## Voxelization: numba replaces C++ `voxel_op`

The compiled `voxel_op` PyTorch extension was replaced with a numba-jit
implementation in `pointpillars/ops/voxelization_numba.py`. This eliminates
the only C++ build step in the vendored tree; install collapses to
`pip install` cross-platform.

Bit-exact parity verified against the original compiled CPU kernel
(`voxelization/voxelization_cpu.cpp`):

| Date       | Input                       | `voxel_num` | All buffers bit-exact? |
|------------|-----------------------------|-------------|------------------------|
| 2026-04-18 | `0000000001.bin` (63114 in-range pts after KITTI filter) | 8984 | yes (`voxels`, `coors`, `num_points_per_voxel`) |

End-to-end inference (`scripts/verify_pointpillars.py`) produces the same 23
detections as the pre-swap baseline (top Pedestrian 0.865, top Car 0.453,
median detection z = -1.48m).

Reproducer: `python scripts/parity_check_voxelization.py` (requires the
compiled `.pyd` to still be present, so dev-time only — see
`scripts/smoke_test_voxelization.py` for the post-install check that ships).

## Other patches

- `pointpillars/ops/iou3d_module.py` — pure-torch axis-aligned NMS on the BEV
  envelope, replacing the CUDA-only rotated NMS. Training-time IoU functions
  stubbed (`NotImplementedError`).
- `pointpillars/utils/__init__.py` — `vis_o3d` import commented out (pulls
  cv2; not needed for headless inference).
- `requirements.txt` — deleted. Upstream's pinned deps no longer apply after
  the patches above (no CUDA stack, no `opencv-python`, torch bumped to 2.11).
  Install is driven by the root `pyproject.toml` / `uv.lock`. Original list
  preserved at zhulf0804/PointPillars upstream if needed.
