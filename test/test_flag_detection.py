"""Tests for `utils/flag_detection.py` — numpy + math only, no ROS."""
import math

import numpy as np
import pytest

from evolutionary_controller_ros.utils import flag_detection as fd


# --------------------------------------------------------------------------
# detect_flag
# --------------------------------------------------------------------------

def test_detect_flag_returns_false_when_no_matching_pixels():
    labels = np.zeros((240, 320), dtype=np.uint8)
    detected, centroid, area = fd.detect_flag(labels, target_label=20)
    assert detected is False
    assert centroid is None
    assert area == 0


def test_detect_flag_returns_false_below_min_area():
    labels = np.zeros((240, 320), dtype=np.uint8)
    # 10 isolated pixels — well below default min_area_px=30
    labels[100, 150:160] = 20
    detected, centroid, area = fd.detect_flag(labels, target_label=20,
                                              min_area_px=30)
    assert detected is False
    assert centroid is None
    assert area == 10


def test_detect_flag_finds_centred_blob():
    labels = np.zeros((240, 320), dtype=np.uint8)
    # 10x10 square right at image centre
    labels[115:125, 155:165] = 20
    detected, centroid, area = fd.detect_flag(labels, target_label=20,
                                              min_area_px=30)
    assert detected is True
    assert area == 100
    assert centroid == pytest.approx(159.5, abs=0.5)   # ~middle column


def test_detect_flag_centroid_for_offset_blob():
    labels = np.zeros((240, 320), dtype=np.uint8)
    # Blob at columns 240-249 (well right of centre 160)
    labels[100:120, 240:250] = 25
    detected, centroid, area = fd.detect_flag(labels, target_label=25,
                                              min_area_px=30)
    assert detected is True
    assert centroid == pytest.approx(244.5, abs=0.5)


def test_detect_flag_ignores_other_labels():
    labels = np.zeros((240, 320), dtype=np.uint8)
    labels[100:200, 100:200] = 20         # lots of label 20
    labels[100:110, 100:110] = 25         # small patch of label 25
    detected, centroid, area = fd.detect_flag(labels, target_label=25,
                                              min_area_px=30)
    assert detected is True
    assert area == 100                    # only the 10x10 patch counts


# --------------------------------------------------------------------------
# bearing_from_centroid — sign convention is the load-bearing piece
# --------------------------------------------------------------------------

def test_bearing_zero_at_image_centre():
    bearing = fd.bearing_from_centroid(centroid_x=160.0,
                                       image_width_px=320,
                                       horizontal_fov_rad=1.57)
    assert bearing == pytest.approx(0.0, abs=1e-6)


def test_bearing_negative_when_centroid_right_of_centre():
    # Pixel to the RIGHT of centre → target is to the right of the
    # robot → NEGATIVE yaw under ROS convention.
    bearing = fd.bearing_from_centroid(centroid_x=240.0,
                                       image_width_px=320,
                                       horizontal_fov_rad=1.57)
    assert bearing < 0


def test_bearing_positive_when_centroid_left_of_centre():
    bearing = fd.bearing_from_centroid(centroid_x=80.0,
                                       image_width_px=320,
                                       horizontal_fov_rad=1.57)
    assert bearing > 0


def test_bearing_magnitude_at_image_edge_is_half_fov():
    bearing_left  = fd.bearing_from_centroid(0.0,   320, 1.57)
    bearing_right = fd.bearing_from_centroid(320.0, 320, 1.57)
    assert bearing_left  == pytest.approx(+1.57 / 2.0, abs=1e-6)
    assert bearing_right == pytest.approx(-1.57 / 2.0, abs=1e-6)


# --------------------------------------------------------------------------
# bearing_to_scan_index
# --------------------------------------------------------------------------

def test_scan_index_centre_bearing_maps_to_middle_ray():
    idx = fd.bearing_to_scan_index(bearing_rad=0.0,
                                   scan_angle_min=-math.pi,
                                   scan_angle_increment=2 * math.pi / 360,
                                   n_ranges=360)
    # bearing 0, angle_min = -π, increment = 2π/360 → index = 180
    assert idx == 180


def test_scan_index_returns_none_outside_coverage():
    # Scan covers only [-pi/4, +pi/4]; a bearing of pi/2 is outside.
    n = 90
    idx = fd.bearing_to_scan_index(bearing_rad=math.pi / 2,
                                   scan_angle_min=-math.pi / 4,
                                   scan_angle_increment=(math.pi / 2) / n,
                                   n_ranges=n)
    assert idx is None


def test_scan_index_raises_on_invalid_increment():
    with pytest.raises(ValueError):
        fd.bearing_to_scan_index(0.0, -math.pi, 0.0, 360)


def test_scan_index_wraps_negative_bearing_for_360_lidar():
    """prm_2026 LIDAR is published as angle_min=0, angle_max≈2π.

    A bearing of -10° (right of robot) must wrap to ≈ 350° on this kind
    of scan, not return None. Regression test for the bug that left
    /flag/distance_m as NaN every time the flag was to the right of
    the robot.
    """
    n = 360
    inc = 2 * math.pi / n
    # bearing = -10° → should map to ≈ index 350
    idx = fd.bearing_to_scan_index(
        bearing_rad=math.radians(-10.0),
        scan_angle_min=0.0,
        scan_angle_increment=inc,
        n_ranges=n,
    )
    assert idx == 350


def test_scan_index_centre_bearing_zero_on_360_lidar():
    n = 360
    inc = 2 * math.pi / n
    idx = fd.bearing_to_scan_index(0.0, 0.0, inc, n)
    assert idx == 0


def test_scan_index_partial_arc_negative_bearing_still_none():
    """A non-360 scan keeps the old behaviour: outside coverage = None."""
    n = 90
    idx = fd.bearing_to_scan_index(
        bearing_rad=math.radians(-30.0),
        scan_angle_min=0.0,
        scan_angle_increment=math.radians(1.0),
        n_ranges=n,
    )
    # arc is only 90° wide so -30° is outside → None
    assert idx is None


def test_fuse_distance_wraps_window_on_360_lidar():
    """Window straddles the array boundary on a 360° scan."""
    n = 360
    inc = 2 * math.pi / n
    scan = [10.0] * n
    # put a "flag" at indices 358, 359, 0, 1, 2 — straddles the seam
    for i in (358, 359, 0, 1, 2):
        scan[i] = 1.5
    d = fd.fuse_distance_from_scan(
        bearing_rad=0.0,        # exactly the seam
        scan_ranges=scan,
        scan_angle_min=0.0,
        scan_angle_increment=inc,
        window_half_width=3,
    )
    assert d == pytest.approx(1.5, abs=1e-6)


# --------------------------------------------------------------------------
# fuse_distance_from_scan — uses the median of a small window
# --------------------------------------------------------------------------

def test_fuse_distance_returns_median_of_window():
    # Synthetic 360-ray scan, all 5.0 m except a "flag" at index 180 of
    # 2.3 m surrounded by close neighbours.
    n = 360
    scan = [5.0] * n
    for i in range(177, 184):     # 7 rays centred at 180
        scan[i] = 2.3
    d = fd.fuse_distance_from_scan(
        bearing_rad=0.0,
        scan_ranges=scan,
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / n,
        window_half_width=3,
    )
    assert d == pytest.approx(2.3, abs=1e-6)


def test_fuse_distance_skips_nan_and_inf():
    n = 360
    scan = [float("inf")] * n
    scan[178] = float("nan")
    scan[180] = 1.5
    scan[182] = float("inf")
    d = fd.fuse_distance_from_scan(
        bearing_rad=0.0,
        scan_ranges=scan,
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / n,
        window_half_width=3,
    )
    assert d == pytest.approx(1.5, abs=1e-6)


def test_fuse_distance_returns_none_when_window_empty():
    scan = [float("inf")] * 360
    d = fd.fuse_distance_from_scan(
        bearing_rad=0.0,
        scan_ranges=scan,
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / 360,
    )
    assert d is None


def test_fuse_distance_respects_max_distance_m():
    # All readings are at 50 m → beyond the 10 m default cap → discarded.
    scan = [50.0] * 360
    d = fd.fuse_distance_from_scan(
        bearing_rad=0.0,
        scan_ranges=scan,
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / 360,
        max_distance_m=10.0,
    )
    assert d is None


def test_fuse_distance_bearing_outside_coverage_returns_none():
    scan = [3.0] * 90    # covers only [-pi/4, +pi/4]
    d = fd.fuse_distance_from_scan(
        bearing_rad=math.pi / 2,
        scan_ranges=scan,
        scan_angle_min=-math.pi / 4,
        scan_angle_increment=(math.pi / 2) / 90,
    )
    assert d is None
