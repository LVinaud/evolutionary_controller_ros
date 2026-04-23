"""Pure sensor → feature-dict pipeline for the GP tree's terminal set.

No ROS or rclpy dependency: the caller decomposes every ROS message into
primitive Python/numpy types before calling `compute_features`. This keeps
the module unit-testable without a simulator and matches the same
"pure-Python evolution core" policy as `evolution/`.

The feature set matches the go-to-goal refactor of `evolution/genome.py`:
the controller's job is to drive the robot to an arbitrary target pose
that is injected externally (`target_pose`). It does not know whether
that target is a flag, a base, or a random waypoint — the hardcoded
CTF state machine in `gp_controller` swaps the target through phases,
but from the GP's perspective it is always just "the goal".

Inputs to `compute_features` (all keyword-only for clarity):
    scan_ranges          : sequence[float] of length N (LaserScan.ranges)
    scan_angle_min       : float (rad) — LaserScan.angle_min
    scan_angle_increment : float (rad) — LaserScan.angle_increment
    robot_pose           : (x, y, yaw) in world frame
    linear_vel, angular_vel : floats (odom twist)
    target_pose          : (x, y) in world frame — the current goal
    params               : dict with thresholds — see Params below.

Output: dict mapping each Bool/Float terminal name to a value. Float
terminals are normalized to `[-1, 1]`; Bool terminals are bool.

Params (all optional, fall back to defaults):
    obstaculo_threshold_m    : Float (default 0.5) — `obstaculo_X` threshold.
    cone_half_width_rad      : Float (default 15° = 0.262) — lidar cone width.
    d_max_m                  : Float (default 10.0) — distance normalization.
    v_max_mps                : Float (default 1.0) — linear-vel normalization.
    w_max_rps                : Float (default 1.0) — angular-vel normalization.
    alvo_prox_radius_m       : Float (default 0.5) — `alvo_proximo` threshold.
    alvo_front_cone_rad      : Float (default π/6 = 30°) — half-angle of the
                               "target in front" cone. Bearings inside
                               ±alvo_front_cone_rad mark `alvo_frente`; larger
                               |bearing| splits into `alvo_esq` / `alvo_dir`.
    alvo_back_cone_rad       : Float (default 2π/3 = 120°) — half-angle of the
                               rear cone. |bearing| > this → `alvo_atras`.

All terminals from the genome's terminal set are present in the output
dict — no missing keys — so `genome.evaluate` can index any of them.
"""
import math


DEFAULT_PARAMS = {
    "obstaculo_threshold_m": 0.5,
    "cone_half_width_rad":   math.radians(15.0),
    "d_max_m":               10.0,
    "v_max_mps":             1.0,
    "w_max_rps":             1.0,
    "alvo_prox_radius_m":    0.5,
    "alvo_front_cone_rad":   math.pi / 6,
    "alvo_back_cone_rad":    2 * math.pi / 3,
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
    target_pose: tuple,
    params: dict = None,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}

    lidar   = _lidar_features(scan_ranges, scan_angle_min,
                              scan_angle_increment, p)
    target  = _target_features(robot_pose, target_pose, p)
    kinetic = _kinetic_features(linear_vel, angular_vel, p)

    return {**lidar, **target, **kinetic}


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
        for i, r in enumerate(scan_ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            a = angle_min + i * angle_increment
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
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ==========================================================================
# Target features (distance + relative bearing + sector booleans)
# ==========================================================================

def _target_features(robot_pose, target_pose, p):
    rx, ry, ryaw = robot_pose
    tx, ty = target_pose

    dx = tx - rx
    dy = ty - ry
    dist = math.hypot(dx, dy)
    # Bearing from robot's heading to the target, in [-π, π].
    bearing = math.atan2(dy, dx) - ryaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))

    abs_b = abs(bearing)
    front_cone = p["alvo_front_cone_rad"]
    back_cone = p["alvo_back_cone_rad"]

    alvo_frente = abs_b <= front_cone
    alvo_atras  = abs_b >= back_cone
    # Side sectors are everything not front/back; sign of bearing decides
    # which side. With ROS convention (+yaw = CCW = left), bearing > 0
    # means the target is to the robot's left.
    in_side = not alvo_frente and not alvo_atras
    alvo_esq = in_side and bearing > 0
    alvo_dir = in_side and bearing < 0

    return {
        "dist_alvo":     _norm_dist(dist, p["d_max_m"]),
        "angulo_alvo":   max(-1.0, min(1.0, bearing / math.pi)),
        "alvo_frente":   bool(alvo_frente),
        "alvo_esq":      bool(alvo_esq),
        "alvo_dir":      bool(alvo_dir),
        "alvo_atras":    bool(alvo_atras),
        "alvo_proximo":  bool(dist < p["alvo_prox_radius_m"]),
    }


# ==========================================================================
# Kinetic features (from odom twist)
# ==========================================================================

def _kinetic_features(linear_vel, angular_vel, p):
    return {
        "velocidade_linear":  max(-1.0, min(1.0, linear_vel / p["v_max_mps"])),
        "velocidade_angular": max(-1.0, min(1.0, angular_vel / p["w_max_rps"])),
    }
