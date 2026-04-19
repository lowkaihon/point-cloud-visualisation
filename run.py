"""End-to-end pipeline: preprocess -> cluster -> detect_dl.

Frameworks: Open3D (RANSAC, DBSCAN, AABB), PyTorch (PointPillars inference),
numba (voxelization), numpy, matplotlib. See requirements.txt.
"""
from __future__ import annotations

import preprocess
import cluster
import detect_dl


def main() -> None:
    print("\n=== preprocess ===")
    state = preprocess.run()
    print("\n=== cluster ===")
    cluster.run(objects_xyz=state["objects_xyz"])
    print("\n=== detect_dl ===")
    detect_dl.run()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
