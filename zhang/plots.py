"""
Figures of a calibration:
  reprojection_error.png   RMS error per image and the cloud of corner residuals
  coverage_distortion.png  where the board corners fell in the frame, and the lens distortion field
  detection_example.png    one photo with the detected corners and the board axes

Author: Nguyen Duc Nam Khanh
"""

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import Calibration  # noqa: E402
from .corners import Board  # noqa: E402
from .images import load_rgb  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID, "axes.grid": True,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10,
})


def plot_reprojection_error(result: Calibration, board: Board, path: str) -> None:
    residuals = result.residuals(board)
    rms = result.per_image_rms(board)
    total = result.rms(board)
    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

    x = np.arange(len(rms))
    ax_bar.bar(x, rms, color=BLUE, width=0.7)
    ax_bar.axhline(total, color=INK, linewidth=1)
    ax_bar.annotate(f"overall RMS {total:.3f} px", (len(rms) - 0.5, total), xytext=(0, 4),
                    textcoords="offset points", ha="right", color=INK)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([os.path.splitext(d.name)[0].replace("IMG_", "") for d in result.detections],
                           rotation=90, fontsize=7)
    ax_bar.set_xlabel("image")
    ax_bar.set_ylabel("RMS reprojection error (px)")
    ax_bar.set_title(f"Per-image error ({len(rms)} images, {board.num_corners} corners each)")

    r = np.vstack(residuals)
    lim = max(np.percentile(np.abs(r), 99.5), 0.5)
    ax_sc.scatter(r[:, 0], r[:, 1], s=2, color=BLUE, alpha=0.35, linewidths=0)
    ax_sc.add_patch(plt.Circle((0, 0), total, fill=False, color=ORANGE, linewidth=1.2))
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("x residual (px)")
    ax_sc.set_ylabel("y residual (px)")
    ax_sc.set_title(f"All {len(r)} corner residuals (circle = RMS)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage_and_distortion(result: Calibration, path: str) -> None:
    w, h = result.image_size
    fig, (ax_cov, ax_dist) = plt.subplots(1, 2, figsize=(10, 6.6))

    for det in result.detections:
        ax_cov.scatter(det.corners[:, 0], det.corners[:, 1], s=1.5, linewidths=0, color=BLUE, alpha=0.5)
    ax_cov.set_title("Board corners of all images in the frame\n(good coverage of the edges constrains distortion)")

    # Distortion field: how far each pixel of an ideal pinhole image moves on the sensor.
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 86))
    ideal = np.stack([gx.ravel(), gy.ravel()], axis=1)
    rays = np.hstack([cv2.undistortPoints(ideal.reshape(-1, 1, 2), result.K, None).reshape(-1, 2),
                      np.ones((len(ideal), 1))])
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), result.K, result.dist)
    shift = np.linalg.norm(distorted.reshape(-1, 2) - ideal, axis=1).reshape(gx.shape)
    contour = ax_dist.contourf(gx, gy, shift, levels=12, cmap="Blues")
    ax_dist.contour(gx, gy, shift, levels=contour.levels, colors=INK_2, linewidths=0.3)
    fig.colorbar(contour, ax=ax_dist, fraction=0.05, label="displacement (px)")
    k1, k2, p1, p2, k3 = result.dist
    ax_dist.set_title(f"Lens distortion magnitude\nk1={k1:.4f} k2={k2:.4f} k3={k3:.4f} p1={p1:.5f} p2={p2:.5f}")

    for ax in (ax_cov, ax_dist):
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_detection_example(result: Calibration, board: Board, path: str) -> None:
    """The image with the median error: detected corners and the board coordinate axes."""
    rms = result.per_image_rms(board)
    k = int(np.argsort(rms)[len(rms) // 2])
    det = result.detections[k]
    length = 3 * board.square_size
    axes_3d = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]])   # -Z points at the camera
    axes_2d, _ = cv2.projectPoints(axes_3d, result.rvecs[k], result.tvecs[k], result.K, result.dist)
    o, *ends = axes_2d.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(load_rgb(det.path))
    ax.plot(*det.corners.T, color=BLUE, linewidth=0.6)                               # row-by-row order
    ax.scatter(*det.corners.T, s=10, facecolors="none", edgecolors=ORANGE, linewidths=0.8)
    for end, color, name in zip(ends, ["#d62728", "#2ca02c", "#1f3fbf"], ["X", "Y", "-Z"]):
        ax.annotate("", end, o, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2))
        ax.annotate(name, end, color=color, fontsize=10, fontweight="bold", xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"{det.name}: detected corners (orange, joined in board order) and board axes\nRMS {rms[k]:.3f} px")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
