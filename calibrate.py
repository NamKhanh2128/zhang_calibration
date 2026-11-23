"""
Calibrate a camera from photos of a planar checkerboard (Zhang's method).

    python calibrate.py                                  # data/checkerboard -> results/
    python calibrate.py --images <folder> --inner-corners 8 11 --square-mm 17

Outputs (in --output):
    camera_intrinsics.json   K, distortion, uncertainties, per-image errors (read by BundleAdjustment-3DGS)
    figures/*.png            reprojection error, corner coverage + distortion field, detection example

Author: Nguyen Duc Nam Khanh
"""

import argparse
import os
import sys

import numpy as np

from zhang import Board, calibrate, detect_all, save_json
from zhang.images import camera_model, list_images
from zhang.plots import plot_coverage_and_distortion, plot_detection_example, plot_reprojection_error

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Zhang camera calibration from checkerboard photos")
    p.add_argument("--images", default=os.path.join("data", "checkerboard"), help="folder of board photos")
    p.add_argument("--inner-corners", type=int, nargs=2, default=[8, 11], metavar=("COLS", "ROWS"),
                   help="inner corners of the board (columns rows)")
    p.add_argument("--square-mm", type=float, default=17.0, help="side of one square in millimetres")
    p.add_argument("--outlier-factor", type=float, default=3.0,
                   help="drop images whose RMS error exceeds this multiple of the median")
    p.add_argument("--fit-k3", action="store_true",
                   help="also estimate k3 (only if the board reached the image corners)")
    p.add_argument("--output", default="results", help="output folder")
    return p.parse_args()


def main():
    args = parse_args()
    board = Board(tuple(args.inner_corners), args.square_mm / 1000.0)
    paths = list_images(args.images)
    print(f"[1/3] Detecting a {board.inner_corners[0]}x{board.inner_corners[1]} board "
          f"({args.square_mm:g} mm squares) in {len(paths)} images")
    detections, failed, size = detect_all(paths, board)

    print(f"[2/3] Calibrating on {len(detections)} images of {size[0]}x{size[1]} px")
    result = calibrate(detections, board, size, args.outlier_factor, args.fit_k3, failed)
    K, d, s = result.K, result.dist, result.std
    Kc = result.K_closed_form
    print(f"      closed form : fx={Kc[0, 0]:.1f} fy={Kc[1, 1]:.1f} cx={Kc[0, 2]:.1f} cy={Kc[1, 2]:.1f} "
          f"skew={Kc[0, 1]:.2f}")
    print(f"      refined     : fx={K[0, 0]:.2f}±{s['fx']:.2f}  fy={K[1, 1]:.2f}±{s['fy']:.2f}  "
          f"cx={K[0, 2]:.2f}±{s['cx']:.2f}  cy={K[1, 2]:.2f}±{s['cy']:.2f}")
    print(f"      distortion  : k1={d[0]:.5f} k2={d[1]:.5f} p1={d[2]:.6f} p2={d[3]:.6f} k3={d[4]:.5f}")
    rms = result.per_image_rms(board)
    print(f"      RMS reprojection error {result.rms(board):.3f} px "
          f"(per image: median {np.median(rms):.3f}, max {rms.max():.3f})")
    if result.rejected:
        print(f"      rejected as outliers: {', '.join(result.rejected)}")
    if result.not_detected:
        print(f"      board not found in: {', '.join(result.not_detected)}")

    print(f"[3/3] Writing {args.output}/")
    figures = os.path.join(args.output, "figures")
    os.makedirs(figures, exist_ok=True)
    save_json(result, board, os.path.join(args.output, "camera_intrinsics.json"), camera_model(paths[0]))
    plot_reprojection_error(result, board, os.path.join(figures, "reprojection_error.png"))
    plot_coverage_and_distortion(result, os.path.join(figures, "coverage_distortion.png"))
    plot_detection_example(result, board, os.path.join(figures, "detection_example.png"))
    print("Done.")


if __name__ == "__main__":
    main()
