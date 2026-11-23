"""Synthetic tests: a known camera photographs a known board from several poses."""

import cv2
import numpy as np
import pytest

from zhang.calibration import Detection, calibrate
from zhang.closed_form import calibrate_closed_form, homography_dlt
from zhang.corners import Board

K_TRUE = np.array([[3200.0, 0.0, 2140.0], [0.0, 3190.0, 2860.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([0.12, -0.35, 0.0004, -0.0003, 0.0])
SIZE = (4284, 5712)
BOARD = Board((8, 11), 0.017)


def _views(dist, noise_px=0.0, seed=0, num_views=12):
    rng = np.random.default_rng(seed)
    obj = BOARD.object_points().astype(np.float64)
    center = obj.mean(axis=0)
    views = []
    for _ in range(num_views):
        rvec = rng.normal(0.0, 0.35, 3)
        R = cv2.Rodrigues(rvec)[0]
        tvec = -R @ center + np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04), rng.uniform(0.3, 0.45)])
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, dist)
        views.append((uv.reshape(-1, 2) + rng.normal(0.0, noise_px, (len(obj), 2)), R, tvec))
    return views


def test_homography_dlt_recovers_exact_homography():
    H = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, 12.0], [1e-4, 2e-4, 1.0]])
    src = np.random.default_rng(1).uniform(-1, 1, (20, 2))
    dst_h = np.hstack([src, np.ones((20, 1))]) @ H.T
    dst = dst_h[:, :2] / dst_h[:, 2:]
    np.testing.assert_allclose(homography_dlt(src, dst), H, rtol=1e-8, atol=1e-10)


def test_closed_form_is_exact_without_noise_and_distortion():
    views = _views(np.zeros(5))
    K, poses = calibrate_closed_form(BOARD.object_points()[:, :2].astype(float), [v[0] for v in views])
    np.testing.assert_allclose(K, K_TRUE, rtol=1e-6, atol=1e-6)
    for (R, t), (_, R_true, t_true) in zip(poses, views):
        np.testing.assert_allclose(R, R_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)


def test_closed_form_needs_three_views():
    views = _views(np.zeros(5), num_views=2)
    with pytest.raises(ValueError):
        calibrate_closed_form(BOARD.object_points()[:, :2].astype(float), [v[0] for v in views])


def test_full_calibration_recovers_intrinsics_and_distortion():
    views = _views(DIST_TRUE, noise_px=0.2, num_views=15)
    detections = [Detection(f"{k}.jpg", uv) for k, (uv, _, _) in enumerate(views)]
    result = calibrate(detections, BOARD, SIZE)
    assert abs(result.K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.005
    assert abs(result.K[0, 2] - K_TRUE[0, 2]) < 10.0
    assert abs(result.K[1, 2] - K_TRUE[1, 2]) < 10.0
    assert abs(result.dist[0] - DIST_TRUE[0]) < 0.03
    assert result.rms(BOARD) == pytest.approx(0.2 * np.sqrt(2), rel=0.15)   # 2-D noise of 0.2 px per axis
    assert not result.rejected


def test_outlier_image_is_rejected():
    views = _views(DIST_TRUE, noise_px=0.2, num_views=12)
    detections = [Detection(f"{k}.jpg", uv) for k, (uv, _, _) in enumerate(views)]
    detections[3].corners = detections[3].corners + np.random.default_rng(5).normal(0, 4.0, (BOARD.num_corners, 2))
    result = calibrate(detections, BOARD, SIZE)
    assert result.rejected == ["3.jpg"]
