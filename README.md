# Lidar Surveillance Demo

## Overview

This demo shows two complementary approaches on a single lidar frame: every
entity in the scene gets a 3D bounding box via geometric clustering
(unclassified, labelled `cluster_N`), and classifiable entities additionally
get a class label and confidence score via a pretrained PointPillars model.
A combined visualization shows both, honestly surfacing what each technique
can and cannot claim.

**Dataset:** a single KITTI-format Velodyne `.bin` file
(`data/0000000001.bin`, 125 826 points, `(x, y, z, intensity)` float32). The provided file is KITTI-format Velodyne .bin (X, Y, Z, intensity); XYZ coordinates are extracted for clustering, all four channels feed the DL model..

## How to run

- **Install:** `pip install -r requirements.txt` (Python 3.12+, developed on
  Windows 11; pure-Python deps, no compiler required).
- **Smoke test (recommended first):** `python scripts/smoke_test_voxelization.py`
  — should print `OK voxel_num=8984 ...`.
- **Full pipeline:** `python run.py` — runs preprocess → cluster → detect_dl
  → visualize end-to-end, writing JSONs + screenshots to `outputs/`.
- **Individual stages:** `python preprocess.py`, `python cluster.py`,
  `python detect_dl.py`, `python visualize.py`.
- **Partial runs:** `python run.py --skip-dl` (no PyTorch) or
  `python run.py --skip-viz`.

Expected first-run warm-up: the numba voxelization kernel JIT-compiles on
first inference (a few seconds); subsequent calls are hot.

## Pipeline

```
load (data/0000000001.bin)
   |
   |-> ego removal -> ROI crop (surveillance ±40m) -> voxel (0.05m) -> SOR
   |     |
   |     +-> RANSAC ground plane + band filter -> objects cloud
   |           |
   |           +-> DBSCAN (eps=0.3, min_pts=10) -> AABB -> geometric filter
   |                 -> detections_clustering.json                  [360°]
   |
   +-> KITTI front ROI (X>=0, <=70m, ±40m) -> PointPillars (pass A)
       +-> 180° Z rotation -> PointPillars (pass B) -> rotate back
             -> detections_dl.json + detections_dl_raw.json   [360° via 7.3.1]
                 |
                 +-> visualize.py -> outputs/screenshots/*.png
```

**Frameworks:** Open3D 0.19 (I/O, RANSAC, DBSCAN, AABB, offscreen rendering),
PyTorch 2 CPU (PointPillars inference), numba (JIT voxelization), numpy,
matplotlib.

**Models:** DBSCAN for unclassified clustering; vendored
[`zhulf0804/PointPillars`](https://github.com/zhulf0804/PointPillars)
pretrained on KITTI (Car / Pedestrian / Cyclist).

## Results

Screenshot gallery (rendered by `visualize.py` into `outputs/screenshots/`;
view angles are configured via `views/*.json` captured interactively — see
§8.3 of `SPEC.md`):

| File | What it shows |
|------|---------------|
| `01_raw_intensity.png` | Raw cloud, viridis-coloured by intensity |
| `02_preproc_roi.png` | After ego removal + surveillance ROI crop |
| `03_preproc_voxel.png` | After voxel downsample + SOR |
| `04_preproc_ground.png` | RANSAC ground (red) vs objects (green) |
| `05_preproc_dbscan.png` | DBSCAN clusters, tab20-coloured |
| `06_clustering_iso.png` | Clustering result, isometric (AABB outlines) |
| `07_clustering_top.png` | Clustering result, top-down |
| `08_clustering_side.png` | Clustering result, side view |
| `09_dl_iso.png` | PointPillars detections, isometric (class-coloured) |
| `10_dl_top.png` | PointPillars detections, top-down |
| `11_dl_side.png` | PointPillars detections, side view |
| `12_combined_iso.png` | Both pipelines overlaid on one scene |

Verified numbers on `0000000001.bin`: 68 clusters survive the §6.3 geometric
filter; PointPillars' two-pass inference yields ≈35 raw detections (pass A
≈23 front-hemisphere, pass B ≈12 rear-hemisphere), 10 of which pass the
`score >= 0.3` threshold. Top detection: Pedestrian 0.87 at (+9.13, -5.65, -0.55).

## Limitations

- **PointPillars training ROI:** the model was trained on the KITTI front
  camera frustum (X ≥ 0, ±40m lateral, up to 70m forward). 360° surveillance
  coverage is achieved via a two-pass rotation trick (see §7.3.1 of
  `SPEC.md`), exploiting lidar's rotational symmetry. A thin lateral wedge
  at `|y| > 40m` remains uncovered in both passes. A production deployment
  would retrain on 360° surveillance data, removing both the trick and the
  residual wedge.
- **Class vocabulary:** PointPillars knows Car / Pedestrian / Cyclist only.
  Out-of-vocabulary entities (parked bicycles on racks, vegetation,
  buildings, benches, poles, bins) are captured by the clustering pipeline
  as `cluster_N`, not by the DL detector. The combined view therefore shows
  *every* entity; only the classifiable ones get a named class. Note that
  KITTI's "Cyclist" means rider + bicycle together — an empty bike rack is
  genuinely OOV rather than a missed Cyclist.
- **Edge-of-ROI DL reliability:** 5 `Car` detections in
  `detections_dl.json` sit at `|x| > 35m` with scores 0.30-0.66. Without
  KITTI labels for this frame they cannot be confirmed true/false
  positives; they may be genuine parked vehicles or model noise in sparse
  lateral returns near the §7.3.1 uncovered wedge. The score threshold is
  held at 0.3 rather than tuned to hide them.
- **Single frame:** no temporal tracking, no velocity estimation.
- **Clustering has no classification:** the §6.3 geometric filter removes
  obvious non-entities (walls, ground stripes) but does not attempt to name
  what remains.
- **Static sensor assumption** — the pipeline is not SLAM-aware.

## Appendix A — Why lidar for surveillance

- **Native 3D geometry.** Metric distances fall out of the sensor without a
  stereo rig or depth inference network.
- **Illumination invariance.** Night, glare, smoke, and moderate rain do not
  degrade lidar returns the way they degrade cameras.
- **Privacy by construction.** No faces, license plates, or clothing colour
  — the representation is geometric, so identifying individuals takes
  additional fusion work that has to be done on purpose.
- **Constant range accuracy.** Unlike stereo cameras, metric accuracy does
  not fall off with distance (up to the sensor's range limit).
- **Occlusion reasoning without ML.** Ray-casting through the point cloud
  answers "is this object behind that one?" geometrically.
- **Honest caveat:** fine-grained classification is weaker than camera,
  because point clouds lack texture/colour cues. Mature systems fuse both
  modalities rather than pick one.

## Appendix B — Preprocessing parameter rationale

| Step | Param | Rationale |
|------|-------|-----------|
| Ego removal | `|x| < 2.5m & |y| < 2.0m` | KITTI sensor-car body / mirrors appear as floating clusters near origin |
| ROI crop | `|x|, |y| < 40m, -2.5 < z < 1.5` | Surveillance perimeter; discards sparse far returns |
| Voxel | `0.05m` | Uniform density; ~3-5× speedup. Voxel *before* SOR — SOR's neighbour stats are skewed on raw Velodyne (density varies ~100× between rings) |
| SOR | `nb_neighbors=20, std_ratio=2.0` | Atmospheric speckle |
| RANSAC | `dist=0.2, n=3, iter=1000, prob=1.0` | `prob=1.0` disables nondeterministic early termination |
| Band filter | `0.3m above plane normal` | Catches residual curbs / sidewalk / scan-ring artefacts the symmetric RANSAC threshold missed |
| DBSCAN | `eps=0.3m, min_points=10` | At 0.05m voxel size, `eps` is ~6 voxel-widths — enough gap to separate adjacent vehicles/pedestrians while still joining sparse distant returns |
| Geometric filter | `pts∈[30,5000], vol≤50m³, max/min extent≤15` | Strips undersampled noise and wall-scale structures without over-filtering entities |

## Appendix C — Reproducibility

- `o3d.utility.random.seed(42)` before `segment_plane` with
  `probability=1.0` — all 1000 RANSAC iterations run, so the plane is
  deterministic across runs.
- DBSCAN is deterministic given fixed input ordering; voxel downsample
  preserves ordering.
- PointPillars inference is deterministic given fixed weights and no
  dropout at eval time (`model.eval()`, `torch.no_grad()`).
- The numba voxelization kernel is single-threaded and deterministic by
  construction (no parallel reduction, no non-deterministic atomic).
- Target Open3D 0.19.x (per `pyproject.toml`); re-verify plane coefficients
  if upgrading. On this frame the expected plane (Open3D 0.19, seed 42) is
  `(a, b, c, d) ≈ (-0.02, -0.02, 1.00, 1.70)`.

## Layout

```
.
├── preprocess.py           # §6.0 phases A-C + plane sign fix
├── cluster.py              # §6.0 phase D + §6.3 AABB + filter + JSON
├── detect_dl.py            # §7: two-pass PointPillars inference
├── visualize.py            # §8: offscreen Open3D rendering
├── io_utils.py             # .bin loader + §5 detection JSON schema
├── run.py                  # §5.1 orchestrator (partial-success tolerant)
├── scripts/
│   ├── smoke_test_voxelization.py
│   ├── verify_pointpillars.py
│   └── parity_check_voxelization.py
├── third_party/PointPillars/       # vendored, CPU-patched (see NOTES.md)
│   └── pretrained/epoch_160.pth    # 19 MB KITTI weights, bundled
├── data/0000000001.bin
├── views/                  # pinhole camera JSONs for screenshots (user-captured)
├── outputs/
│   ├── detections_clustering.json
│   ├── detections_dl.json          # score >= 0.3
│   ├── detections_dl_raw.json      # all pass-A + pass-B
│   ├── preprocessed.pkl
│   └── screenshots/
├── requirements.txt
├── pyproject.toml          # dev dep manifest (uv-managed)
└── SPEC.md                 # authoritative implementation spec
```
