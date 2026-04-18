"""Verify that PointPillars (KITTI pretrained) runs locally on CPU.

Uses the patched zhulf0804/PointPillars vendored in third_party/PointPillars:
  - voxelization compiled CPU-only (build_cpu.bat)
  - iou3d_module.py replaced with a pure-torch axis-aligned NMS (CUDA-free)

Run: uv run python scripts/verify_pointpillars.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party" / "PointPillars"
sys.path.insert(0, str(THIRD_PARTY))

from pointpillars.model import PointPillars  # noqa: E402

LABEL2CLASS = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}

# Matches test.py's point_range_filter; this is the KITTI front-facing range
# the model was trained on. Points outside are dropped before inference.
POINT_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]


def point_range_filter(pts: np.ndarray, rng: list[float]) -> np.ndarray:
    mask = (
        (pts[:, 0] > rng[0]) & (pts[:, 0] < rng[3])
        & (pts[:, 1] > rng[1]) & (pts[:, 1] < rng[4])
        & (pts[:, 2] > rng[2]) & (pts[:, 2] < rng[5])
    )
    return pts[mask]


def main() -> int:
    bin_path = REPO_ROOT / "0000000001.bin"
    ckpt_path = THIRD_PARTY / "pretrained" / "epoch_160.pth"
    out_path = REPO_ROOT / "outputs" / "detections_dl.json"
    out_path.parent.mkdir(exist_ok=True)

    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    print(f"loaded {len(pts)} points from {bin_path.name}")
    pts_in_range = point_range_filter(pts, POINT_RANGE)
    print(f"after range filter {POINT_RANGE}: {len(pts_in_range)} points")

    model = PointPillars(nclasses=3)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    pc_torch = torch.from_numpy(pts_in_range)
    with torch.no_grad():
        result = model(batched_pts=[pc_torch], mode="test")[0]

    lidar_bboxes = result["lidar_bboxes"]  # (N, 7) = [x, y, z, dx, dy, dz, yaw]
    labels = result["labels"]               # (N,) int
    scores = result["scores"]               # (N,) float

    print(f"\nraw detections: {len(scores)}")
    detections = []
    for i in range(len(scores)):
        box = lidar_bboxes[i]
        det = {
            "id": i,
            "label": LABEL2CLASS.get(int(labels[i]), f"cls_{int(labels[i])}"),
            "score": float(scores[i]),
            "center": [float(box[0]), float(box[1]), float(box[2])],
            "extent": [float(box[3]), float(box[4]), float(box[5])],
            "yaw": float(box[6]),
        }
        detections.append(det)

    detections.sort(key=lambda d: -d["score"])
    print("top detections:")
    for d in detections[:10]:
        print(
            f"  {d['label']:12s} score={d['score']:.3f} "
            f"center=({d['center'][0]:+6.2f}, {d['center'][1]:+6.2f}, {d['center'][2]:+6.2f}) "
            f"extent=({d['extent'][0]:.2f}, {d['extent'][1]:.2f}, {d['extent'][2]:.2f}) "
            f"yaw={d['yaw']:+.2f}"
        )

    if detections:
        zs = [d["center"][2] for d in detections]
        median_z = float(np.median(zs))
        print(f"\nmedian detection z = {median_z:+.2f} (expected ~-1.0 to +1.0 in Velodyne frame)")
        assert -3.0 <= median_z <= 2.5, f"suspicious median z={median_z}; frame mismatch?"

    out_payload = {
        "source": "pointpillars",
        "coord_frame": "velodyne",
        "input_file": bin_path.name,
        "detections": detections,
    }
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {len(detections)} detections to {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
