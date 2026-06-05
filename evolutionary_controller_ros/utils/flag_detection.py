"""Pure-Python flag detection from a Gazebo semantic-segmentation image.

ROS-free, numpy-only. Lets us unit-test the geometry without bringing up
a simulator. The ROS-side glue (subscriber, message decoding, LIDAR
fusion) lives in `controllers/gp_controller.py`.

Concept
    The prm_2026 robot publishes `/robot_cam/labels_map`: each pixel
    carries the integer label ID of the Gazebo model it belongs to. The
    enemy flag has a known label (defined in `arena_cilindros.sdf`: red
    flag = 20, blue flag = 25). Detection reduces to: "find pixels
    where value == enemy_label, return their centroid + pixel count."

Bearing convention
    The image axis is X right / Y down. The robot frame (and ROS
    convention) is X forward, Y left, yaw positive counterclockwise.
    A pixel to the RIGHT of the image centre therefore corresponds to
    a target on the RIGHT of the robot, which is a NEGATIVE yaw
    angle. `bearing_from_centroid` applies that sign flip explicitly.

Distance is NOT computed here — the fusion with `/scan` happens in the
ROS layer, where the LIDAR data is already in hand. This module only
gives the bearing and the area; the controller takes care of the rest.
"""
from typing import Optional


def detect_flag(
    labels: "numpy.ndarray",
    target_label: int,
    min_area_px: int = 30,
) -> tuple[bool, Optional[float], int]:
    """Find pixels matching `target_label`; return detection state + centroid.

    Parameters
    ----------
    labels : 2-D numpy array of integer pixel labels.
    target_label : int
        Pixel value that identifies the enemy flag.
    min_area_px : int, default 30
        Minimum number of matching pixels to declare a detection. Below
        this threshold the function returns (False, None, area) — useful
        to ignore single-pixel noise on edge cases.

    Returns
    -------
    (detected, centroid_x, area_px)
        detected : bool   — True iff `area_px >= min_area_px`.
        centroid_x : Optional[float]  — fractional x coordinate of the
            centroid in pixel space, None when `detected` is False.
        area_px : int     — raw count of matching pixels (always returned
            even when below threshold; useful for debugging / tuning).
    """
    # Lazy-import numpy so the module is importable in environments
    # without numpy (e.g. minimal test runs); calling detect_flag
    # without numpy will raise the usual ImportError.
    import numpy as np

    mask = labels == int(target_label)
    area = int(mask.sum())
    if area < min_area_px:
        return False, None, area

    # Centroid: mean column index of matching pixels. We compute it as
    # column sums / total, which is a tad faster than np.where + mean
    # for large masks.
    col_indices = np.arange(labels.shape[1])
    col_weights = mask.sum(axis=0)            # how many matching pixels per column
    centroid_x = float((col_indices * col_weights).sum()) / area
    return True, centroid_x, area


def bearing_from_centroid(
    centroid_x: float,
    image_width_px: int,
    horizontal_fov_rad: float,
) -> float:
    """Convert a horizontal image centroid to a bearing in radians.

    A centroid at the image centre returns 0. Pixels to the right of
    centre return NEGATIVE values (target is to the right of the robot,
    i.e. negative yaw under ROS convention). Pixels to the left return
    POSITIVE values.

    Assumes a rectilinear (pinhole) camera and treats horizontal FOV as
    spanning the full image width. Small-angle approximation is NOT used
    — this is a `tan` mapping via simple linear scaling, which is exact
    enough at the ~90° FOV used in prm_2026.
    """
    centred = centroid_x - (image_width_px / 2.0)
    return -(centred / image_width_px) * horizontal_fov_rad


def bearing_to_scan_index(
    bearing_rad: float,
    scan_angle_min: float,
    scan_angle_increment: float,
    n_ranges: int,
) -> Optional[int]:
    """Map a robot-frame bearing onto the LIDAR's `ranges` array index.

    Returns None when the bearing falls outside the scan's angular
    coverage (the LIDAR may not be 360°, depending on the robot). For
    full-circle LIDARs (`angle_max - angle_min >= 2π`), this function
    wraps the bearing modulo 2π so that bearings like -10° map onto
    the upper end of the ranges array (≈ 350° on a 0–360° scan).
    Without that wrap, the prm_2026 LIDAR — which is published as
    `angle_min=0, angle_max≈2π` — would silently return None for any
    negative (right-of-robot) bearing, so the camera↔LIDAR fusion
    would always read NaN for a flag on the right.

    This is the geometric half of the camera↔LIDAR fusion: the camera
    gives the bearing, this function picks the matching LIDAR ray, and
    the controller reads `ranges[index]` to get the actual distance.
    """
    import math
    if scan_angle_increment <= 0.0:
        raise ValueError("scan_angle_increment must be positive")

    total_arc = scan_angle_increment * n_ranges
    is_full_circle = total_arc >= 2.0 * math.pi - 1e-3

    if is_full_circle:
        # Wrap bearing into [scan_angle_min, scan_angle_min + 2π).
        normalised = (bearing_rad - scan_angle_min) % (2.0 * math.pi)
        idx = int(round(normalised / scan_angle_increment)) % n_ranges
        return idx

    idx = int(round((bearing_rad - scan_angle_min) / scan_angle_increment))
    if idx < 0 or idx >= n_ranges:
        return None
    return idx


def fuse_distance_from_scan(
    bearing_rad: float,
    scan_ranges: list,
    scan_angle_min: float,
    scan_angle_increment: float,
    *,
    window_half_width: int = 3,
    max_distance_m: float = 10.0,
) -> Optional[float]:
    """Median LIDAR range in a small angular window around the bearing.

    Reading a single ray is noisy; reading too wide a window risks
    catching the wrong object (a wall behind the flag). A 7-ray median
    (`window_half_width=3`) is a good compromise: smooths sensor
    noise, narrow enough to stay on a flag-sized target at typical
    distances.

    Returns None when no valid range is available (all NaN/inf, the
    bearing falls outside the scan, or every reading is beyond
    `max_distance_m`).
    """
    import math

    n = len(scan_ranges)
    centre = bearing_to_scan_index(bearing_rad, scan_angle_min,
                                   scan_angle_increment, n)
    if centre is None:
        return None
    # Build a window that wraps around the array if the centre is near
    # the edge of a 360° scan — otherwise a flag at bearing ≈ 0 on the
    # prm_2026 LIDAR (which starts at angle_min=0) would only see half
    # of the window, biasing the median.
    indices = [(centre + d) % n
               for d in range(-window_half_width, window_half_width + 1)]
    window = [scan_ranges[i] for i in indices
              if math.isfinite(scan_ranges[i])
              and 0.0 < scan_ranges[i] <= max_distance_m]
    if not window:
        return None
    window.sort()
    return window[len(window) // 2]   # median
