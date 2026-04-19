"""Interactive scene browser for the lidar deliverable.

Opens the raw point cloud with ground stripped (viridis intensity) overlaid
with clustering boxes (yellow) and DL boxes (class-coloured). Orbit, zoom,
and pan to frame interesting moments and capture screenshots with your OS
screenshot tool (Windows: `Win+Shift+S`).

Ground removal calls `preprocess.run()` inline to get the RANSAC plane, then
applies the same 0.3m band threshold as `preprocess.HEIGHT_ABOVE_PLANE` — so
the viewer matches what DL inferred on (no voxel / SOR / ROI crop), minus
the visually dominant ground return.

Mouse: left-drag rotate, right-drag pan, scroll zoom.
Exit: Q / Esc / close window.

Usage:
    python interactive_viewer.py
    python interactive_viewer.py --no-boxes
    python interactive_viewer.py --raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import open3d as o3d

import preprocess
from io_utils import load_bin, read_detections_json

BIN_PATH = Path("data/0000000001.bin")
CLUSTERING_JSON = Path("outputs/detections_clustering.json")
DL_JSON = Path("outputs/detections_dl.json")
WIN_W, WIN_H = 1920, 1080
HEIGHT_ABOVE_PLANE = 0.3  # matches preprocess.HEIGHT_ABOVE_PLANE

# Class colours for DL boxes (RGB 0-1) — match archive/visualize.py.
CLASS_COLOR = {
    "Car": (0.0, 1.0, 1.0),       # cyan
    "Pedestrian": (1.0, 0.0, 1.0),  # magenta
    "Cyclist": (1.0, 0.5, 0.0),    # orange
}
CLUSTER_OUTLINE = (1.0, 0.85, 0.2)  # warm yellow — pops against grey cloud

# 12 edges of a box: 4 bottom, 4 top, 4 vertical risers.
BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def _rotz(yaw: float) -> np.ndarray:
    """3x3 rotation matrix about Z by `yaw` radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _normalize(values: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] with a floor on the range to avoid divide-by-zero."""
    return (values - values.min()) / max(float(np.ptp(values)), 1e-6)


def _det_to_lineset(det: dict, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """12-edge box from a detection dict (center/extent/yaw, rotation about Z)."""
    cx, cy, cz = det["center"]
    dx, dy, dz = det["extent"]
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    corners = local @ _rotz(det["yaw"]).T + np.array([cx, cy, cz])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(BOX_EDGES))
    return ls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="suppress clustering + DL overlays (bare point cloud only)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="skip ground removal; show the unmodified raw cloud",
    )
    args = parser.parse_args()

    xyz, intensity = load_bin(BIN_PATH)
    source = "raw"
    if not args.raw:
        state = preprocess.run()
        a, b, c, d = state["plane"]
        normal = np.array([a, b, c], dtype=np.float64)
        signed = (xyz @ normal + d) / np.linalg.norm(normal)
        keep = signed > HEIGHT_ABOVE_PLANE
        xyz = xyz[keep]
        intensity = intensity[keep]
        source = "raw minus ground (plane computed inline)"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = cm.viridis(_normalize(intensity))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries: list = [pcd]

    n_cl = n_dl = 0
    if not args.no_boxes:
        if CLUSTERING_JSON.exists():
            clustering = read_detections_json(CLUSTERING_JSON)["detections"]
            for d in clustering:
                geometries.append(_det_to_lineset(d, CLUSTER_OUTLINE))
            n_cl = len(clustering)
        if DL_JSON.exists():
            dl = read_detections_json(DL_JSON)["detections"]
            for d in dl:
                color = CLASS_COLOR.get(d["label"], (1.0, 1.0, 1.0))
                geometries.append(_det_to_lineset(d, color))
            n_dl = len(dl)

    print("=" * 60)
    print(f"Loaded {len(xyz)} points ({source})")
    if args.no_boxes:
        print("Boxes suppressed (--no-boxes)")
    else:
        print(f"Clustering boxes: {n_cl}  |  DL boxes: {n_dl}")
    print("Orbit to frame the scene; screenshot with Win+Shift+S.")
    print("=" * 60)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="interactive_viewer", width=WIN_W, height=WIN_H)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])
    opt.point_size = 2.5
    opt.line_width = 2.5

    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
