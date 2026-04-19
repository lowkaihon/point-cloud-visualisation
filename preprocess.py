"""Preprocessing pipeline for lidar point cloud.

Framework: Open3D (voxel_down_sample, remove_statistical_outlier, segment_plane).
Pipeline:
  1. Ego-vehicle removal — reject points inside the KITTI sensor-car body.
  2. ROI crop — ±40m surveillance perimeter, z ∈ [-2.5, 1.5]m.
  3. Voxel downsample (0.05m) — uniform density, ~3-5x speedup.
  4. Statistical outlier removal — trim atmospheric speckle.
  5. RANSAC ground-plane fit + signed-distance band filter (0.3m above plane).
Intensity is carried forward as a parallel numpy array (Open3D's PointCloud
only stores XYZ).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import open3d as o3d

from io_utils import load_bin

# Ego + surveillance ROI
EGO_X, EGO_Y = 2.5, 2.0
ROI_X, ROI_Y = 40.0, 40.0
ROI_Z = (-2.5, 1.5)

# Voxel + SOR
VOXEL_SIZE = 0.05
SOR_NB, SOR_STD = 20, 2.0

# RANSAC + band filter
RANSAC_DIST = 0.2
RANSAC_N = 3
RANSAC_ITERS = 1000
RANSAC_PROB = 1.0  # force all iterations (deterministic)
HEIGHT_ABOVE_PLANE = 0.3

DEFAULT_BIN = "data/0000000001.bin"


def load_and_color_raw(bin_path: str | Path) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Load .bin; build Open3D cloud with viridis intensity colouring."""
    xyz, intensity = load_bin(bin_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(cm.viridis(intensity)[:, :3])
    return pcd, xyz, intensity


def ego_filter(xyz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ego-vehicle filter: reject |x|<2.5 AND |y|<2.0 (sensor car body/mirrors)."""
    keep = ~((np.abs(xyz[:, 0]) < EGO_X) & (np.abs(xyz[:, 1]) < EGO_Y))
    return xyz[keep], intensity[keep]


def roi_crop(xyz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surveillance ROI: |x|,|y|<40, z∈[-2.5, 1.5]."""
    keep = (
        (np.abs(xyz[:, 0]) < ROI_X)
        & (np.abs(xyz[:, 1]) < ROI_Y)
        & (xyz[:, 2] > ROI_Z[0])
        & (xyz[:, 2] < ROI_Z[1])
    )
    return xyz[keep], intensity[keep]


def _intensity_via_nn(
    downsampled_xyz: np.ndarray, source_xyz: np.ndarray, source_intensity: np.ndarray
) -> np.ndarray:
    """Re-attach intensity to voxel-downsampled points via KDTree nearest neighbour.

    Open3D's voxel_down_sample drops attributes; queries the pre-voxel cloud
    for the closest point to each voxel centroid (offset ≤ voxel_size/2 ≈ 2.5cm,
    well within intensity's spatial coherence).
    """
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    tree = o3d.geometry.KDTreeFlann(src_pcd)
    out = np.empty(len(downsampled_xyz), dtype=np.float32)
    for i, p in enumerate(downsampled_xyz):
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        out[i] = source_intensity[idx[0]]
    return out


def voxel_and_sor(
    xyz: np.ndarray,
    intensity: np.ndarray,
    voxel: float = VOXEL_SIZE,
    nb: int = SOR_NB,
    std: float = SOR_STD,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Voxel downsample → SOR. Returns (pcd, xyz, intensity) in lockstep."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(xyz)
    down = src.voxel_down_sample(voxel_size=voxel)
    down_xyz = np.asarray(down.points)
    down_intensity = _intensity_via_nn(down_xyz, xyz, intensity)

    _, keep = down.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    cleaned = down.select_by_index(keep)
    keep_np = np.asarray(keep, dtype=np.int64)
    return cleaned, down_xyz[keep_np], down_intensity[keep_np]


def segment_ground(
    pcd: o3d.geometry.PointCloud,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """RANSAC plane fit; sign-normalize plane normal.

    Returns (plane, inlier_idx).
    """
    o3d.utility.random.seed(42)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERS,
        probability=RANSAC_PROB,
    )
    a, b, c, d = plane_model
    # Force the plane normal to point upward (+z); band_filter then reads
    # `signed_distance > HEIGHT_ABOVE_PLANE` as "above ground".
    if c < 0:
        a, b, c, d = -a, -b, -c, -d
    return (float(a), float(b), float(c), float(d)), np.asarray(inliers, dtype=np.int64)


def band_filter(
    xyz: np.ndarray,
    plane: tuple[float, float, float, float],
    min_height: float = HEIGHT_ABOVE_PLANE,
) -> np.ndarray:
    """Signed-distance band filter: reject points ≤ 0.3m above the fitted plane."""
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=np.float64)
    signed = (xyz @ normal + d) / np.linalg.norm(normal)
    return signed > min_height


def run(bin_path: str | Path = DEFAULT_BIN) -> dict:
    """End-to-end preprocess. Returns intermediate + final clouds.

    Dict keys:
      - raw_xyz, raw_intensity
      - stage_clouds: dict of stage-name → xyz array
      - objects_xyz, objects_intensity: final objects cloud (input to cluster.py)
      - ground_xyz: primary RANSAC ground inliers
      - plane: (a, b, c, d) sign-normalized
    """
    _, raw_xyz, raw_intensity = load_and_color_raw(bin_path)
    print(f"Raw: {len(raw_xyz)} points")

    ego_xyz, ego_intensity = ego_filter(raw_xyz, raw_intensity)
    print(f"After ego removal: {len(ego_xyz)} points")

    roi_xyz, roi_intensity = roi_crop(ego_xyz, ego_intensity)
    print(f"After ROI crop: {len(roi_xyz)} points")

    cleaned_pcd, cleaned_xyz, cleaned_intensity = voxel_and_sor(roi_xyz, roi_intensity)
    print(f"After voxel + SOR: {len(cleaned_xyz)} points")

    plane, inlier_idx = segment_ground(cleaned_pcd)
    a, b, c, d = plane
    print(f"Plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    ground_mask = np.zeros(len(cleaned_xyz), dtype=bool)
    ground_mask[inlier_idx] = True
    remaining_xyz = cleaned_xyz[~ground_mask]
    remaining_intensity = cleaned_intensity[~ground_mask]

    band_mask = band_filter(remaining_xyz, plane)
    objects_xyz = remaining_xyz[band_mask]
    objects_intensity = remaining_intensity[band_mask]
    print(
        f"Primary ground inliers: {int(ground_mask.sum())}, "
        f"residual stripped: {int((~band_mask).sum())}, "
        f"objects kept: {len(objects_xyz)}"
    )

    return {
        "raw_xyz": raw_xyz,
        "raw_intensity": raw_intensity,
        "stage_clouds": {
            "roi": roi_xyz,
            "voxel_sor": cleaned_xyz,
            "ground_removed": remaining_xyz[band_mask],
        },
        "objects_xyz": objects_xyz,
        "objects_intensity": objects_intensity,
        "ground_xyz": cleaned_xyz[ground_mask],
        "plane": plane,
    }


if __name__ == "__main__":
    run()
