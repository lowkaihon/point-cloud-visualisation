"""DL detection — PointPillars (KITTI-pretrained; Car / Pedestrian / Cyclist).

Framework: PyTorch 2 CPU. Model vendored under third_party/PointPillars.
Two-pass 180° Z-rotation for 360° surveillance coverage (KITTI's training
ROI covers the front hemisphere only). Writes score-filtered detections
to outputs/detections_dl.json per the detection JSON schema.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from io_utils import INPUT_FRAME, load_bin, make_detection, write_detections_json

REPO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = REPO_ROOT / "third_party" / "PointPillars"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from pointpillars.model import PointPillars  # noqa: E402

# KITTI front-facing training ROI (applied per pass).
KITTI_RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
LABEL2CLASS = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
DEFAULT_WEIGHTS = THIRD_PARTY / "pretrained" / "epoch_160.pth"
DEFAULT_OUT = Path("outputs/detections_dl.json")
DEFAULT_SCORE_THRESH = 0.3


def _load_model(weights_path: Path, device: torch.device) -> PointPillars:
    model = PointPillars(nclasses=len(LABEL2CLASS)).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _kitti_range_mask(pts: np.ndarray) -> np.ndarray:
    xmin, ymin, zmin, xmax, ymax, zmax = KITTI_RANGE
    return (
        (pts[:, 0] > xmin) & (pts[:, 0] < xmax)
        & (pts[:, 1] > ymin) & (pts[:, 1] < ymax)
        & (pts[:, 2] > zmin) & (pts[:, 2] < zmax)
    )


def _infer_once(
    model: PointPillars, pts: np.ndarray, device: torch.device, id_offset: int = 0
) -> list[dict]:
    """Single forward pass. Returns list of detection dicts (no score filter)."""
    pts_in = pts[_kitti_range_mask(pts)]
    pc = torch.from_numpy(pts_in).to(device)
    with torch.no_grad():
        result = model(batched_pts=[pc], mode="test")[0]
    boxes = np.asarray(result["lidar_bboxes"])
    labels = np.asarray(result["labels"])
    scores = np.asarray(result["scores"])
    dets: list[dict] = []
    for i in range(len(scores)):
        # PointPillars box format: (x, y, z_bottom, dx, dy, dz, yaw).
        cx, cy, z_bottom, dx, dy, dz, yaw = boxes[i]
        # Schema wants the geometric centroid, so lift by dz/2.
        z_center = z_bottom + dz / 2.0
        label = LABEL2CLASS.get(int(labels[i]), f"cls_{int(labels[i])}")
        dets.append(
            make_detection(
                id=id_offset + i,
                label=label,
                score=scores[i],
                center=(cx, cy, z_center),
                extent=(dx, dy, dz),
                yaw=yaw,
                num_points=0,  # PointPillars does not emit per-detection point counts
            )
        )
    return dets


def _rotate_pts_180z(pts: np.ndarray) -> np.ndarray:
    """180° rotation about Z: (x, y) -> (-x, -y); z and intensity unchanged."""
    out = pts.copy()
    out[:, 0] = -pts[:, 0]
    out[:, 1] = -pts[:, 1]
    return out


def _rotate_detection_180z(det: dict) -> dict:
    """Undo the 180° Z pre-rotation applied to pass-B points (self-inverse)."""
    cx, cy, cz = det["center"]
    return {
        **det,
        "center": [-cx, -cy, cz],
        "yaw": float((det["yaw"] + np.pi) % (2 * np.pi)),
    }


def run(
    bin_path: str | Path = INPUT_FRAME,
    weights: str | Path = DEFAULT_WEIGHTS,
    out_path: Path = DEFAULT_OUT,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> list[dict]:
    """Run two-pass PointPillars inference, write the score-filtered JSON."""
    bin_path = Path(bin_path)
    device = torch.device("cpu")
    model = _load_model(Path(weights), device)

    xyz, intensity = load_bin(bin_path)
    pts = np.hstack([xyz, intensity[:, None]]).astype(np.float32)
    print(f"Loaded {len(pts)} points from {bin_path.name}")

    # Pass A (front hemisphere)
    dets_a = _infer_once(model, pts, device, id_offset=0)
    print(f"Pass A: {len(dets_a)} detections")

    # Pass B (rear hemisphere via 180° Z rotation)
    pts_rot = _rotate_pts_180z(pts)
    dets_b_raw = _infer_once(model, pts_rot, device, id_offset=len(dets_a))
    dets_b = [_rotate_detection_180z(d) for d in dets_b_raw]
    print(f"Pass B: {len(dets_b)} detections (after rotate-back)")

    raw = dets_a + dets_b
    filtered = [d for d in raw if d["score"] >= score_thresh]

    print(
        f"Total: {len(raw)} (A={len(dets_a)} + B={len(dets_b)}), "
        f"filtered (score>={score_thresh}): {len(filtered)}"
    )
    print(f"Class breakdown: {dict(Counter(d['label'] for d in filtered))}")
    if filtered:
        top = max(filtered, key=lambda d: d["score"])
        print(
            f"Top: {top['label']} score={top['score']:.2f} "
            f"at ({top['center'][0]:+.2f}, {top['center'][1]:+.2f}, {top['center'][2]:+.2f})"
        )

    write_detections_json(
        out_path, source="pointpillars", input_file=bin_path.name, detections=filtered
    )
    print(f"Wrote {len(filtered)} detections -> {out_path}")
    return filtered


if __name__ == "__main__":
    run()
