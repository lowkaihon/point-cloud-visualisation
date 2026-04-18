"""Offscreen Open3D rendering (§8).

Produces the screenshot gallery required by §8.1 / §12. Each render uses a
pinhole camera JSON from views/ when available, otherwise Open3D's default
auto-fit. Falls back gracefully so the pipeline runs end-to-end before the
interactive view-capture session of §8.3.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from io_utils import read_detections_json

CLASS_COLOR = {"Car": (0.0, 1.0, 1.0), "Pedestrian": (1.0, 0.0, 1.0), "Cyclist": (1.0, 1.0, 0.0)}
CLUSTER_OUTLINE = (1.0, 0.85, 0.2)  # warm yellow — pops against grey cloud
BG_GREY = (0.55, 0.55, 0.55)
GROUND_COLOR = (0.8, 0.4, 0.4)

DEFAULT_BIN = Path("data/0000000001.bin")
DEFAULT_PICKLE = Path("outputs/preprocessed.pkl")
DEFAULT_CLUSTERING = Path("outputs/detections_clustering.json")
DEFAULT_DL = Path("outputs/detections_dl.json")
DEFAULT_VIEWS = Path("views")
DEFAULT_OUT = Path("outputs/screenshots")
WIN_W, WIN_H = 1920, 1080


def _pcd(xyz: np.ndarray, color: tuple[float, float, float] | np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        pcd.colors = o3d.utility.Vector3dVector(color)
    else:
        pcd.paint_uniform_color(list(color))
    return pcd


def det_to_lineset(det: dict, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """12-edge box from §5 detection (center/extent/yaw, rotation about Z)."""
    cx, cy, cz = det["center"]
    dx, dy, dz = det["extent"]
    yaw = det["yaw"]

    local = np.array([
        [-dx / 2, -dy / 2, -dz / 2], [dx / 2, -dy / 2, -dz / 2],
        [dx / 2, dy / 2, -dz / 2], [-dx / 2, dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, dz / 2], [dx / 2, -dy / 2, dz / 2],
        [dx / 2, dy / 2, dz / 2], [-dx / 2, dy / 2, dz / 2],
    ])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    corners = local @ R.T + np.array([cx, cy, cz])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(edges))
    return ls


def _render(
    geometries: list,
    out_png: Path,
    view_json: Path | None = None,
    width: int = WIN_W,
    height: int = WIN_H,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    try:
        for g in geometries:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.point_size = 1.5
        opt.line_width = 2.0

        if view_json is not None and view_json.exists():
            params = o3d.io.read_pinhole_camera_parameters(str(view_json))
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_png), do_render=True)
    finally:
        vis.destroy_window()
    print(f"  wrote {out_png.name}")


def _label_colors(labels: np.ndarray) -> np.ndarray:
    """tab20 colouring per label; noise (-1) rendered black."""
    max_label = int(labels.max()) if labels.size and labels.max() > 0 else 1
    colors = plt.get_cmap("tab20")(labels / max_label)[:, :3]
    colors[labels < 0] = 0.0
    return colors


def render_raw(xyz: np.ndarray, intensity: np.ndarray, out_png: Path, view_json: Path | None) -> None:
    colors = cm.viridis(intensity)[:, :3]
    _render([_pcd(xyz, colors)], out_png, view_json)


def render_stage_cloud(xyz: np.ndarray, out_png: Path, view_json: Path | None) -> None:
    intensity_like = np.linspace(0.0, 1.0, len(xyz))
    colors = cm.viridis(intensity_like)[:, :3]
    _render([_pcd(xyz, colors)], out_png, view_json)


def render_ground_split(
    ground_xyz: np.ndarray, objects_xyz: np.ndarray, out_png: Path, view_json: Path | None
) -> None:
    geoms = [_pcd(ground_xyz, GROUND_COLOR), _pcd(objects_xyz, (0.3, 0.8, 0.5))]
    _render(geoms, out_png, view_json)


def render_dbscan(
    objects_xyz: np.ndarray, labels: np.ndarray, out_png: Path, view_json: Path | None
) -> None:
    colors = _label_colors(labels)
    _render([_pcd(objects_xyz, colors)], out_png, view_json)


def render_clustering(
    bg_xyz: np.ndarray, detections: list[dict], out_png: Path, view_json: Path | None
) -> None:
    geoms: list = [_pcd(bg_xyz, BG_GREY)]
    for d in detections:
        geoms.append(det_to_lineset(d, CLUSTER_OUTLINE))
    _render(geoms, out_png, view_json)


def render_dl(
    bg_xyz: np.ndarray, detections: list[dict], out_png: Path, view_json: Path | None
) -> None:
    geoms: list = [_pcd(bg_xyz, BG_GREY)]
    for d in detections:
        color = CLASS_COLOR.get(d["label"], (1.0, 1.0, 1.0))
        geoms.append(det_to_lineset(d, color))
    _render(geoms, out_png, view_json)


def render_combined(
    bg_xyz: np.ndarray,
    clustering_dets: list[dict],
    dl_dets: list[dict],
    out_png: Path,
    view_json: Path | None,
) -> None:
    geoms: list = [_pcd(bg_xyz, BG_GREY)]
    for d in clustering_dets:
        geoms.append(det_to_lineset(d, CLUSTER_OUTLINE))
    for d in dl_dets:
        color = CLASS_COLOR.get(d["label"], (1.0, 1.0, 1.0))
        geoms.append(det_to_lineset(d, color))
    _render(geoms, out_png, view_json)


def _view(views_dir: Path, name: str) -> Path | None:
    p = views_dir / f"{name}.json"
    return p if p.exists() else None


def run(
    bin_path: str | Path = DEFAULT_BIN,
    pickle_path: Path = DEFAULT_PICKLE,
    clustering_json: Path = DEFAULT_CLUSTERING,
    dl_json: Path = DEFAULT_DL,
    views_dir: Path = DEFAULT_VIEWS,
    out_dir: Path = DEFAULT_OUT,
) -> None:
    """Render the §8 screenshot set from whatever JSONs/caches exist (§5.1 partial)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = Path(bin_path)

    from io_utils import load_bin
    raw_xyz, raw_intensity = load_bin(bin_path)

    state = None
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            state = pickle.load(f)

    iso = _view(views_dir, "iso")
    top = _view(views_dir, "top")
    side = _view(views_dir, "side")

    # 1. Raw intensity-coloured
    render_raw(raw_xyz, raw_intensity, out_dir / "01_raw_intensity.png", iso)

    # 2-5. Preprocessing stages (top-down)
    if state is not None:
        render_stage_cloud(state["stage_clouds"]["roi"], out_dir / "02_preproc_roi.png", top)
        render_stage_cloud(state["stage_clouds"]["voxel_sor"], out_dir / "03_preproc_voxel.png", top)
        render_ground_split(
            state["ground_xyz"], state["stage_clouds"]["ground_removed"],
            out_dir / "04_preproc_ground.png", top,
        )
        # DBSCAN colouring needs labels — recompute quickly
        from cluster import dbscan as _dbscan
        labels = _dbscan(state["objects_xyz"])
        render_dbscan(state["objects_xyz"], labels, out_dir / "05_preproc_dbscan.png", top)
    else:
        print("  [viz] preprocessed.pkl missing — skipping preprocessing sub-shots 02-05")

    # 6-8. Clustering detections, 3 angles
    if clustering_json.exists():
        clustering = read_detections_json(clustering_json)["detections"]
        render_clustering(raw_xyz, clustering, out_dir / "06_clustering_iso.png", iso)
        render_clustering(raw_xyz, clustering, out_dir / "07_clustering_top.png", top)
        render_clustering(raw_xyz, clustering, out_dir / "08_clustering_side.png", side)
    else:
        print(f"  [viz] {clustering_json} missing — skipping clustering shots 06-08")
        clustering = []

    # 9-11. DL detections, 3 angles
    if dl_json.exists():
        dl = read_detections_json(dl_json)["detections"]
        render_dl(raw_xyz, dl, out_dir / "09_dl_iso.png", iso)
        render_dl(raw_xyz, dl, out_dir / "10_dl_top.png", top)
        render_dl(raw_xyz, dl, out_dir / "11_dl_side.png", side)
    else:
        print(f"  [viz] {dl_json} missing — skipping DL shots 09-11")
        dl = []

    # 12. Combined (money shot)
    if clustering or dl:
        render_combined(raw_xyz, clustering, dl, out_dir / "12_combined_iso.png", iso)
    else:
        print("  [viz] no detections available — skipping combined shot 12")

    print(f"Screenshots -> {out_dir}")


if __name__ == "__main__":
    run()
