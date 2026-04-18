"""PointPillars DL detection (§7.3, §7.3.1, §7.4).

Thin wrapper around the vendored third_party/PointPillars. Two-pass rotation
for 360° coverage: pass A = raw; pass B = 180° rotation about Z, then
rotate detections back. Writes the union to both raw and score-filtered JSONs.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from io_utils import load_bin, make_detection, write_detections_json

REPO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = REPO_ROOT / "third_party" / "PointPillars"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from pointpillars.model import PointPillars  # noqa: E402

# §7.3 — KITTI front-facing training ROI (applied per pass).
KITTI_RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
LABEL2CLASS = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
DEFAULT_BIN = Path("data/0000000001.bin")
DEFAULT_WEIGHTS = THIRD_PARTY / "pretrained" / "epoch_160.pth"
DEFAULT_OUT_RAW = Path("outputs/detections_dl_raw.json")
DEFAULT_OUT_FILT = Path("outputs/detections_dl.json")
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
    """Single forward pass. Returns list of §5-schema dicts (no score filter)."""
    pts_in = pts[_kitti_range_mask(pts)]
    pc = torch.from_numpy(pts_in).to(device)
    with torch.no_grad():
        result = model(batched_pts=[pc], mode="test")[0]
    boxes = np.asarray(result["lidar_bboxes"])
    labels = np.asarray(result["labels"])
    scores = np.asarray(result["scores"])
    dets: list[dict] = []
    for i in range(len(scores)):
        cls = LABEL2CLASS.get(int(labels[i]), f"cls_{int(labels[i])}")
        dets.append(
            make_detection(
                id=id_offset + i,
                label=cls,
                score=float(scores[i]),
                center=boxes[i, :3],
                extent=boxes[i, 3:6],
                yaw=float(boxes[i, 6]),
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


def _rotate_det_back(det: dict) -> dict:
    """Inverse of the 180° Z rotation (the rotation is self-inverse)."""
    det["center"][0] = -det["center"][0]
    det["center"][1] = -det["center"][1]
    det["yaw"] = float((det["yaw"] + np.pi) % (2 * np.pi))
    return det


def _sanity_checks(dets_a: list[dict], dets_b: list[dict], score_thresh: float) -> None:
    """§7.4 per-pass warnings + median-z assert."""
    if not dets_a:
        print("WARN: pass A empty - front-hemisphere inference returned nothing")
    if not dets_b and dets_a:
        print(
            "WARN: pass A produced detections but pass B did not; "
            "verify rotate-back transform (x/y sign flips, yaw+pi) "
            "against the rendered output before trusting the result."
        )
    for name, dets in [("A", dets_a), ("B", dets_b)]:
        zs = [d["center"][2] for d in dets if d["score"] >= score_thresh]
        if not zs:
            continue
        mz = float(np.median(zs))
        assert -2.5 <= mz <= 0.0, (
            f"Pass {name} median detection z={mz:.2f} outside expected "
            "[-2.5, 0.0]m. Likely coordinate frame mismatch or buggy "
            "pass-B rotate-back (check x/y sign flips and yaw+pi)."
        )


def run(
    bin_path: str | Path = DEFAULT_BIN,
    weights: str | Path = DEFAULT_WEIGHTS,
    out_raw: Path = DEFAULT_OUT_RAW,
    out_filt: Path = DEFAULT_OUT_FILT,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> tuple[list[dict], list[dict]]:
    """Run two-pass PointPillars inference, write raw and score-filtered JSONs."""
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
    dets_b = [_rotate_det_back(d) for d in dets_b_raw]
    print(f"Pass B: {len(dets_b)} detections (after rotate-back)")

    _sanity_checks(dets_a, dets_b, score_thresh)

    raw = dets_a + dets_b
    filtered = [d for d in raw if d["score"] >= score_thresh]

    print(
        f"Raw: {len(raw)} (A={len(dets_a)} + B={len(dets_b)}), "
        f"filtered (score>={score_thresh}): {len(filtered)}"
    )
    print(f"Class breakdown (raw): {dict(Counter(d['label'] for d in raw))}")
    if filtered:
        top = max(filtered, key=lambda d: d["score"])
        print(
            f"Top: {top['label']} score={top['score']:.2f} "
            f"at ({top['center'][0]:+.2f}, {top['center'][1]:+.2f}, {top['center'][2]:+.2f})"
        )

    write_detections_json(
        out_raw, source="pointpillars", input_file=bin_path.name, detections=raw
    )
    write_detections_json(
        out_filt, source="pointpillars", input_file=bin_path.name, detections=filtered
    )
    print(f"Wrote {len(raw)} raw -> {out_raw}")
    print(f"Wrote {len(filtered)} filtered -> {out_filt}")
    return raw, filtered


if __name__ == "__main__":
    run()
