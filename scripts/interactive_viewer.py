"""Interactive scene browser for the lidar deliverable.

Opens the raw point cloud with ground stripped via the cached RANSAC
plane (viridis intensity) overlaid with clustering boxes (yellow) and DL
boxes (class-coloured). Orbit, zoom, and pan to frame interesting
moments (detected cars / pedestrians, edge-of-ROI hallucinations, bike
rack clusters) and capture screenshots with your OS screenshot tool
(Windows: `Win+Shift+S`).

Ground removal reuses `plane` from `outputs/preprocessed.pkl` and the
same 0.3m band threshold as `preprocess.HEIGHT_ABOVE_PLANE` — so the
viewer matches what DL inferred on (no voxel / SOR / ROI crop), minus
the visually dominant ground return. Falls back to the full raw cloud
if the preprocessing cache doesn't exist yet.

Mouse: left-drag rotate, right-drag pan, scroll zoom.
Exit: Q / Esc / close window.

Usage:
    python scripts/interactive_viewer.py
    python scripts/interactive_viewer.py --no-boxes
    python scripts/interactive_viewer.py --raw
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from io_utils import load_bin, read_detections_json  # noqa: E402
from visualize import CLASS_COLOR, CLUSTER_OUTLINE, det_to_lineset  # noqa: E402

BIN_PATH = REPO_ROOT / "data" / "0000000001.bin"
PREPROC_PKL = REPO_ROOT / "outputs" / "preprocessed.pkl"
CLUSTERING_JSON = REPO_ROOT / "outputs" / "detections_clustering.json"
DL_JSON = REPO_ROOT / "outputs" / "detections_dl.json"
WIN_W, WIN_H = 1920, 1080
HEIGHT_ABOVE_PLANE = 0.3  # matches preprocess.HEIGHT_ABOVE_PLANE


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
    if not args.raw and PREPROC_PKL.exists():
        with open(PREPROC_PKL, "rb") as f:
            state = pickle.load(f)
        a, b, c, d = state["plane"]
        normal = np.array([a, b, c], dtype=np.float64)
        signed = (xyz @ normal + d) / np.linalg.norm(normal)
        keep = signed > HEIGHT_ABOVE_PLANE
        xyz = xyz[keep]
        intensity = intensity[keep]
        source = "raw minus ground (plane from preprocessed.pkl)"
    elif not args.raw:
        print(f"  [note] {PREPROC_PKL.name} missing — falling back to raw cloud")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    norm = (intensity - intensity.min()) / max(float(np.ptp(intensity)), 1e-6)
    colors = cm.viridis(norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries: list = [pcd]

    n_cl = n_dl = 0
    if not args.no_boxes:
        if CLUSTERING_JSON.exists():
            clustering = read_detections_json(CLUSTERING_JSON)["detections"]
            for d in clustering:
                geometries.append(det_to_lineset(d, CLUSTER_OUTLINE))
            n_cl = len(clustering)
        if DL_JSON.exists():
            dl = read_detections_json(DL_JSON)["detections"]
            for d in dl:
                color = CLASS_COLOR.get(d["label"], (1.0, 1.0, 1.0))
                geometries.append(det_to_lineset(d, color))
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
