"""I/O helpers — KITTI Velodyne .bin loader + detection JSON schema.

Framework: numpy. JSON envelope used by cluster.py and detect_dl.py:
  {source, coord_frame, input_file, detections[]}
Each detection: id, label, score, center, extent, yaw, num_points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# First .bin alphabetically in data/, falling back to the provided frame.
# Replacing the file in data/ re-points the pipeline with no code edits.
INPUT_FRAME: Path = next(
    iter(sorted(Path("data").glob("*.bin"))),
    Path("data/0000000001.bin"),
)


def load_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a KITTI Velodyne .bin file.

    Returns (xyz, intensity) as separate float32 arrays — intensity is kept
    separate because Open3D's PointCloud only holds XYZ.
    """
    raw = np.fromfile(Path(path), dtype=np.float32).reshape(-1, 4)
    return raw[:, :3].copy(), raw[:, 3].copy()


def make_detection(
    *,
    id: int,
    label: str,
    score: float,
    center: tuple[float, float, float] | np.ndarray,
    extent: tuple[float, float, float] | np.ndarray,
    yaw: float,
    num_points: int,
) -> dict:
    """Build a single detection dict."""
    c = np.asarray(center, dtype=float).reshape(3)
    e = np.asarray(extent, dtype=float).reshape(3)
    return {
        "id": int(id),
        "label": str(label),
        "score": float(score),
        "center": [float(c[0]), float(c[1]), float(c[2])],
        "extent": [float(e[0]), float(e[1]), float(e[2])],
        "yaw": float(yaw),
        "num_points": int(num_points),
    }


def write_detections_json(
    path: str | Path,
    *,
    source: str,
    input_file: str,
    detections: list[dict],
) -> None:
    """Write the detection envelope to JSON."""
    if source not in ("clustering", "pointpillars"):
        raise ValueError(f"source must be 'clustering' or 'pointpillars', got {source!r}")
    payload = {
        "source": source,
        "coord_frame": "velodyne",
        "input_file": input_file,
        "detections": detections,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_detections_json(path: str | Path) -> dict:
    """Load a detection envelope JSON."""
    return json.loads(Path(path).read_text())
