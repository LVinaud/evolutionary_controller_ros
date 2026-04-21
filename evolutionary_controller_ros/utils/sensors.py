"""Pure sensor → feature-dict pipeline for the GP tree's terminal set.

No ROS or rclpy dependency: the caller decomposes every ROS message into
primitive Python/numpy types before calling `compute_features`. This keeps
the module unit-testable without a simulator and matches the same
"pure-Python evolution core" policy as `evolution/`.

Inputs to `compute_features` (all keyword-only for clarity):
    scan_ranges          : sequence[float] of length N (LaserScan.ranges)
    scan_angle_min       : float (rad) — LaserScan.angle_min
    scan_angle_increment : float (rad) — LaserScan.angle_increment
    robot_pose           : (x, y, yaw) in world frame
    linear_vel, angular_vel : floats (odom twist)
    labels_map           : numpy.ndarray HxW uint8 (pixel value = label id)
                           or None when no frame is available yet.
    enemy_flag_pose      : (x, y) in world frame (pose of the flag we hunt)
    own_base_pose        : (x, y)
    enemy_base_pose      : (x, y)
    deploy_zone_pose     : (x, y)
    holding_flag         : bool (maintained by episode.py; see project memory)
    enemy_flag_label     : int (20 if enemy is red, 25 if blue)
    enemy_base_label     : int (10 if enemy is red, 15 if blue)
    params               : dict with thresholds — see Params below.

Output: dict mapping each Bool/Float terminal name to a value. Float
terminals are normalized to `[-1, 1]`; Bool terminals are bool.

Params (all optional, fall back to defaults):
    obstaculo_threshold_m    : Float (default 0.5) — `obstaculo_X` threshold.
    cone_half_width_rad      : Float (default 15° = 0.262) — width of the
                               frontal/side/back cones over which we take the
                               minimum range.
    d_max_m                  : Float (default 10.0) — distance normalization.
    v_max_mps                : Float (default 1.0) — linear-vel normalization.
    w_max_rps                : Float (default 1.0) — angular-vel normalization.
    at_base_threshold_m      : Float (default 1.0) — `na_base_propria` /
                               `na_zona_deploy` distance threshold.
    flag_center_band_frac    : Float (default 1/3) — width fraction of the
                               central column for `bandeira_centralizada`.

All terminals from the genome's terminal set are present in the output
dict — no missing keys — so `genome.evaluate` can index any of them.
"""
import math

try:
    import numpy as np
except ImportError:  # keeps the module importable without numpy
    np = None


DEFAULT_PARAMS = {
    "obstaculo_threshold_m": 0.5,
    "cone_half_width_rad":   math.radians(15.0),
    "d_max_m":               10.0,
    "v_max_mps":             1.0,
    "w_max_rps":             1.0,
    "at_base_threshold_m":   1.0,
    "flag_center_band_frac": 1.0 / 3.0,
}


# ==========================================================================
# Top-level entry point
# ==========================================================================

def compute_features(
    *,
    scan_ranges,
    scan_angle_min: float,
    scan_angle_increment: float,
    robot_pose: tuple,
    linear_vel: float,
    angular_vel: float,
    labels_map,
    enemy_flag_pose: tuple,
    own_base_pose: tuple,
    deploy_zone_pose: tuple,
    holding_flag: bool,
    enemy_flag_label: int,
    enemy_base_label: int,
    enemy_base_pose: tuple = None,
    params: dict = None,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}

    lidar = _lidar_features(scan_ranges, scan_angle_min,
                            scan_angle_increment, p)
    cam = _camera_features(labels_map, enemy_flag_label, enemy_base_label, p)
    spatial = _spatial_features(
        robot_pose, enemy_flag_pose, own_base_pose,
        deploy_zone_pose, enemy_base_pose, p,
    )
    kinetic = _kinetic_features(linear_vel, angular_vel, p)

    return {**lidar, **cam, **spatial, **kinetic,
            "segurando_bandeira": bool(holding_flag)}


# ==========================================================================
# Lidar cones (min of each quadrant)
# ==========================================================================

def _lidar_features(scan_ranges, angle_min, angle_increment, p):
    half = p["cone_half_width_rad"]
    d_max = p["d_max_m"]
    thr = p["obstaculo_threshold_m"]

    def cone_min_m(center_rad):
        """Min range in metres within `center_rad ± half`."""
        best = math.inf
        n = len(scan_ranges)
        for i, r in enumerate(scan_ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            a = angle_min + i * angle_increment
            # wrap a and center into [-π, π] before diffing
            d_ang = _angle_diff(a, center_rad)
            if abs(d_ang) <= half and r < best:
                best = r
        return best  # may be math.inf if no valid ray in the cone

    d_f = cone_min_m(0.0)
    d_b = cone_min_m(math.pi)
    d_l = cone_min_m(math.pi / 2)
    d_r = cone_min_m(-math.pi / 2)

    return {
        "dist_frente": _norm_dist(d_f, d_max),
        "dist_atras":  _norm_dist(d_b, d_max),
        "dist_esq":    _norm_dist(d_l, d_max),
        "dist_dir":    _norm_dist(d_r, d_max),
        "obstaculo_frente": math.isfinite(d_f) and d_f < thr,
        "obstaculo_esq":    math.isfinite(d_l) and d_l < thr,
        "obstaculo_dir":    math.isfinite(d_r) and d_r < thr,
    }


def _norm_dist(d_m: float, d_max: float) -> float:
    if not math.isfinite(d_m):
        return 1.0
    return max(0.0, min(1.0, d_m / d_max))


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest-difference a−b wrapped to [-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


# ==========================================================================
# Camera — segmentation labels map
# ==========================================================================

def _camera_features(labels_map, enemy_flag_label, enemy_base_label, p):
    """Derive flag-visibility / position / enemy-base-visibility.

    `labels_map` is HxW uint8; pixel value == label id of the model occupying
    that pixel, 0 = background. `None` means no frame yet (all flags False).
    """
    if labels_map is None or np is None:
        return {
            "bandeira_inimiga_visivel": False,
            "bandeira_esquerda": False,
            "bandeira_centralizada": False,
            "bandeira_direita": False,
            "base_inimiga_visivel": False,
            "angulo_bandeira_inimiga": 0.0,
        }

    h, w = labels_map.shape[:2]
    flag_mask = (labels_map == enemy_flag_label)
    n_flag = int(flag_mask.sum())

    flag_seen = n_flag > 0
    flag_left = flag_center = flag_right = False
    angle_norm = 0.0

    if flag_seen:
        ys, xs = np.nonzero(flag_mask)
        cx = float(xs.mean())
        band = p["flag_center_band_frac"]
        left_edge = w * (0.5 - band / 2)
        right_edge = w * (0.5 + band / 2)
        flag_left = cx < left_edge
        flag_right = cx > right_edge
        flag_center = not (flag_left or flag_right)
        # horizontal offset -> normalized angle in [-1, 1]
        # cx=0 → -1 (full left), cx=w → +1 (full right)
        angle_norm = max(-1.0, min(1.0, (cx - w / 2) / (w / 2)))

    base_seen = bool((labels_map == enemy_base_label).any())

    return {
        "bandeira_inimiga_visivel": bool(flag_seen),
        "bandeira_esquerda":   bool(flag_left),
        "bandeira_centralizada": bool(flag_center),
        "bandeira_direita":    bool(flag_right),
        "base_inimiga_visivel": base_seen,
        "angulo_bandeira_inimiga": float(angle_norm),
    }


# ==========================================================================
# World-frame spatial relations (distances + relative angles)
# ==========================================================================

def _spatial_features(robot_pose, enemy_flag, own_base,
                      deploy, enemy_base, p):
    rx, ry, ryaw = robot_pose
    d_max = p["d_max_m"]
    thr_at = p["at_base_threshold_m"]

    def rel(pt):
        dx = pt[0] - rx
        dy = pt[1] - ry
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - ryaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return dist, bearing

    d_base, ang_base = rel(own_base)
    d_zone, ang_zone = rel(deploy)

    return {
        "dist_base_propria": _norm_dist(d_base, d_max),
        "angulo_base_propria": max(-1.0, min(1.0, ang_base / math.pi)),
        "dist_zona_deploy": _norm_dist(d_zone, d_max),
        "angulo_zona_deploy": max(-1.0, min(1.0, ang_zone / math.pi)),
        "na_base_propria": d_base < thr_at,
        "na_zona_deploy":  d_zone < thr_at,
    }


# ==========================================================================
# Kinetic features (from odom twist)
# ==========================================================================

def _kinetic_features(linear_vel, angular_vel, p):
    return {
        "velocidade_linear":  max(-1.0, min(1.0, linear_vel / p["v_max_mps"])),
        "velocidade_angular": max(-1.0, min(1.0, angular_vel / p["w_max_rps"])),
    }
