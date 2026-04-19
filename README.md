# Lidar Surveillance Demo

## Overview

A lidar surveillance demo on a single KITTI Velodyne frame
(`data/0000000001.bin`, 125 826 points, `(x, y, z, intensity)` float32).
Two complementary detection pipelines run over the frame:

- **Geometric clustering** — every entity in the scene gets a 3D bounding
  box (unclassified, labelled `cluster_N`).
- **Deep learning** — classifiable entities additionally get a class label
  and confidence score via a KITTI-pretrained PointPillars model.

Results combine into one scene overlay via the interactive viewer so each
technique's strengths and limits are surfaced honestly.

## How to run

```
pip install -r requirements.txt
python run.py
python interactive_viewer.py
```

Orbit, zoom, and pan in the viewer; capture screenshots with your OS tool
(Windows `Win+Shift+S`). First-run warm-up: the numba voxelization kernel
JIT-compiles on first inference (a few seconds).

## Pipeline

```
load (data/0000000001.bin)
   |
   |-> ego removal -> ROI crop (±40m) -> voxel (0.05m) -> SOR
   |     |
   |     +-> RANSAC ground plane + band filter -> objects cloud
   |           |
   |           +-> DBSCAN (eps=0.3, min_pts=10) -> AABB -> geometric filter
   |                 -> outputs/detections_clustering.json          [360°]
   |
   +-> KITTI front ROI -> PointPillars (pass A)
       +-> 180° Z rotation -> PointPillars (pass B) -> rotate back
             -> outputs/detections_dl.json                          [360°]
```

**Frameworks:** Open3D 0.19 (I/O, RANSAC, DBSCAN, AABB, visualisation),
PyTorch 2 CPU (PointPillars inference), numba (JIT voxelization), numpy,
matplotlib. See `requirements.txt`.

**Models:** DBSCAN for unclassified clustering; vendored
[`zhulf0804/PointPillars`](https://github.com/zhulf0804/PointPillars)
pretrained on KITTI (Car / Pedestrian / Cyclist).

## Results on `0000000001.bin`

- Clustering: 68 boxes post geometric-filter
- PointPillars: ~10 detections at `score ≥ 0.3` (top: Pedestrian 0.87 at
  `(+9.13, -5.65, -0.55)`)

Screenshots (hand-framed via `interactive_viewer.py`) live in
`screenshots/`.

## Limitations

- **PointPillars training ROI** — the model was trained on KITTI's front
  camera frustum. 360° coverage is achieved via a two-pass 180° Z-rotation
  trick; a thin lateral wedge at `|y| > 40m` remains uncovered.
- **Class vocabulary** — PointPillars knows Car / Pedestrian / Cyclist
  only. Other entities (vegetation, benches, poles, bike racks) are caught
  by clustering as `cluster_N` rather than by the DL detector.
- **Single frame** — no temporal tracking, no velocity estimation.

## Layout

```
.
├── preprocess.py           # ego/ROI/voxel/SOR/RANSAC + band filter
├── cluster.py              # DBSCAN + AABB + geometric filter -> JSON
├── detect_dl.py            # two-pass PointPillars inference -> JSON
├── run.py                  # sequential orchestrator
├── io_utils.py             # .bin loader + detection JSON schema
├── interactive_viewer.py   # orbit/zoom viewer for screenshots
├── third_party/PointPillars/   # vendored, CPU-patched
│   └── pretrained/epoch_160.pth
├── data/0000000001.bin
├── outputs/
│   ├── detections_clustering.json
│   └── detections_dl.json
├── screenshots/
├── requirements.txt
└── README.md
```
