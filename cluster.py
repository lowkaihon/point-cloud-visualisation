"""Clustering model: DBSCAN (eps=0.3m, min_points=10).

Framework: Open3D (cluster_dbscan, axis-aligned bounding box). Runs DBSCAN
over the preprocessed objects cloud, computes per-cluster AABBs, applies a
geometric filter (point-count / volume / extent-ratio) to strip noise and
wall-scale structures, and writes detections to
outputs/detections_clustering.json per the §5 schema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from io_utils import make_detection, write_detections_json

# §6.0 / §6.2 — locked
DBSCAN_EPS = 0.3
DBSCAN_MIN_POINTS = 10

# §6.3 starting thresholds — re-tune only on obvious failure
FILTER_MIN_PTS = 30
FILTER_MAX_PTS = 5000
FILTER_MAX_VOL = 50.0  # m^3
FILTER_MAX_RATIO = 15.0  # max_extent / min_extent

DEFAULT_INPUT = Path("data/0000000001.bin")
DEFAULT_OUT = Path("outputs/detections_clustering.json")


def dbscan(
    objects_xyz: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_points: int = DBSCAN_MIN_POINTS,
) -> np.ndarray:
    """Run DBSCAN on an objects cloud; return per-point labels (int32, -1=noise)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objects_xyz)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points), dtype=np.int32)
    n_clusters = int(labels.max() + 1) if labels.size else 0
    n_noise = int((labels == -1).sum())
    frac = n_noise / len(labels) if labels.size else 0.0
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({frac:.1%} noise)")
    return labels


def cluster_aabb(
    xyz: np.ndarray, labels: np.ndarray, cluster_id: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (center, extent, num_points) for a single cluster."""
    mask = labels == cluster_id
    pts = xyz[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    aabb = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center())
    extent = np.asarray(aabb.get_extent())
    return center, extent, int(pts.shape[0])


def geometric_filter(
    num_points: int,
    extent: np.ndarray,
    min_pts: int = FILTER_MIN_PTS,
    max_pts: int = FILTER_MAX_PTS,
    max_vol: float = FILTER_MAX_VOL,
    max_ratio: float = FILTER_MAX_RATIO,
) -> tuple[bool, str]:
    """§6.3 geometric filter. Returns (passed, reason)."""
    if num_points < min_pts:
        return False, f"pts<{min_pts}"
    if num_points > max_pts:
        return False, f"pts>{max_pts}"
    volume = float(np.prod(extent))
    if volume > max_vol:
        return False, f"vol>{max_vol}"
    e_min = float(np.min(extent))
    e_max = float(np.max(extent))
    if e_min <= 0.0:
        return False, "degenerate extent"
    if e_max / e_min > max_ratio:
        return False, f"ratio>{max_ratio}"
    return True, "ok"


def build_clustering_detections(
    objects_xyz: np.ndarray, labels: np.ndarray, verbose: bool = True
) -> list[dict]:
    """Per cluster: AABB -> filter -> §5 dict. Prints §6.3 diagnostic table."""
    if verbose:
        print(f"{'id':>4} {'pts':>6} {'dx':>5} {'dy':>5} {'dz':>5} "
              f"{'vol':>7} {'cz':>6} passed  reason")
        print("-" * 64)

    n_clusters = int(labels.max() + 1) if labels.size else 0
    detections: list[dict] = []
    new_id = 0
    for cid in range(n_clusters):
        center, extent, npts = cluster_aabb(objects_xyz, labels, cid)
        volume = float(np.prod(extent))
        passed, reason = geometric_filter(npts, extent)
        if verbose:
            print(
                f"{cid:>4} {npts:>6} {extent[0]:>5.2f} {extent[1]:>5.2f} {extent[2]:>5.2f} "
                f"{volume:>7.2f} {center[2]:>+6.2f} {'PASS' if passed else 'FAIL':<6}  {reason}"
            )
        if not passed:
            continue
        detections.append(
            make_detection(
                id=new_id,
                label=f"cluster_{new_id}",
                score=1.0,
                center=center,
                extent=extent,
                yaw=0.0,
                num_points=npts,
            )
        )
        new_id += 1

    if verbose:
        print(f"Clusters: {n_clusters} raw -> {len(detections)} kept")
    return detections


def run(
    objects_xyz: np.ndarray | None = None,
    out_path: Path = DEFAULT_OUT,
    input_file: str = DEFAULT_INPUT.name,
) -> list[dict]:
    """End-to-end clustering stage.

    When called standalone (objects_xyz=None), preprocess.run() is invoked
    inline to produce the objects cloud.
    """
    if objects_xyz is None:
        import preprocess
        objects_xyz = preprocess.run()["objects_xyz"]
    labels = dbscan(objects_xyz)
    dets = build_clustering_detections(objects_xyz, labels)
    write_detections_json(
        out_path, source="clustering", input_file=input_file, detections=dets
    )
    print(f"Wrote {len(dets)} detections -> {out_path}")
    return dets


if __name__ == "__main__":
    run()
