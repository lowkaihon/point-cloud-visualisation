"""End-to-end orchestrator (§5.1).

Partial-success tolerant: each stage runs in its own try/except so a DL
failure still produces clustering-only output, a viz failure leaves JSONs
on disk, etc. Exit code is 1 if any stage failed, 0 otherwise.

Stages (§5.1 table): preprocess -> cluster -> detect_dl -> visualize.
Only the preprocess -> cluster link is a hard prerequisite.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import preprocess
import cluster
import detect_dl
import visualize


def _run_stage(name: str, fn, **kwargs):
    """Run a stage; log failures; return (ok, result)."""
    print(f"\n=== [{name}] ===")
    try:
        return True, fn(**kwargs)
    except Exception:
        traceback.print_exc()
        print(f"[{name}] FAILED - continuing")
        return False, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lidar surveillance demo pipeline")
    parser.add_argument("--skip-dl", action="store_true", help="skip PointPillars DL stage")
    parser.add_argument("--skip-viz", action="store_true", help="skip visualization stage")
    args = parser.parse_args(argv)

    had_failure = False

    # preprocess — hard prerequisite for cluster
    ok, state = _run_stage("preprocess", preprocess.run)
    if not ok:
        had_failure = True

    # cluster — only runs if preprocess succeeded
    if ok and state is not None:
        ok_cl, _ = _run_stage(
            "cluster", cluster.run, objects_xyz=state["objects_xyz"]
        )
        if not ok_cl:
            had_failure = True
    else:
        print("\n=== [cluster] SKIPPED - preprocess failed ===")
        had_failure = True

    # detect_dl — independent of preprocess
    if not args.skip_dl:
        ok_dl, _ = _run_stage("detect_dl", detect_dl.run)
        if not ok_dl:
            had_failure = True
    else:
        print("\n=== [detect_dl] SKIPPED - --skip-dl ===")

    # visualize — renders whatever JSONs exist
    if not args.skip_viz:
        ok_viz, _ = _run_stage("visualize", visualize.run)
        if not ok_viz:
            had_failure = True
    else:
        print("\n=== [visualize] SKIPPED - --skip-viz ===")

    print("\n=== DONE ===")
    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
