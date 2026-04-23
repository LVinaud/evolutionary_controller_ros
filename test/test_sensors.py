"""Unit tests for utils.sensors — pure sensor → feature-dict pipeline."""
import math

import pytest

from evolutionary_controller_ros.utils import sensors as s
from evolutionary_controller_ros.evolution import genome as g


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------

def _uniform_scan(n=360, value=5.0):
    return [value] * n


def _base_kwargs(**over):
    n = 360
    kw = dict(
        scan_ranges=_uniform_scan(n=n, value=5.0),
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / n,
        robot_pose=(0.0, 0.0, 0.0),
        linear_vel=0.0,
        angular_vel=0.0,
        target_pose=(8.0, 0.0),
    )
    kw.update(over)
    return kw


# --------------------------------------------------------------------------
# Coverage of the whole terminal set
# --------------------------------------------------------------------------

def test_all_terminals_present():
    feats = s.compute_features(**_base_kwargs())
    for name in g.BOOL_TERMINALS:
        assert name in feats, f"missing bool terminal: {name}"
        assert isinstance(feats[name], bool)
    for name in g.FLOAT_TERMINALS:
        assert name in feats, f"missing float terminal: {name}"
        assert isinstance(feats[name], float)
        assert -1.0 <= feats[name] <= 1.0


def test_features_suffice_to_evaluate_a_tree():
    tree = {"op": "IF",
            "cond": {"op": "AND",
                     "a": {"term": "alvo_proximo"},
                     "b": {"op": "LT",
                           "a": {"term": "dist_frente"},
                           "b": {"erc": 0.5}}},
            "then": {"leaf": "GIRA_DIR", "dur_ms": 200},
            "else": {"leaf": "FRENTE", "dur_ms": 300}}
    feats = s.compute_features(**_base_kwargs())
    out = g.evaluate(tree, feats)
    assert out in [("GIRA_DIR", 200), ("FRENTE", 300)]


# --------------------------------------------------------------------------
# Lidar cones
# --------------------------------------------------------------------------

def test_obstaculo_frente_triggers_when_front_ray_below_threshold():
    n = 360
    scan = _uniform_scan(n, 5.0)
    # Index corresponding to angle=0 with angle_min=-π and inc=2π/n is n/2.
    scan[n // 2] = 0.2
    feats = s.compute_features(**_base_kwargs(scan_ranges=scan))
    assert feats["obstaculo_frente"] is True
    assert feats["dist_frente"] < 0.1  # 0.2 m / 10 m = 0.02


def test_obstaculo_frente_false_when_all_rays_far():
    feats = s.compute_features(**_base_kwargs())
    assert feats["obstaculo_frente"] is False
    assert feats["dist_frente"] == pytest.approx(0.5)  # 5 m / 10 m


def test_obstaculo_side_detection_independent():
    n = 360
    scan = _uniform_scan(n, 5.0)
    # angle = +π/2 → index = (π/2 - (-π)) / (2π/n) = 3n/4
    scan[3 * n // 4] = 0.3
    feats = s.compute_features(**_base_kwargs(scan_ranges=scan))
    assert feats["obstaculo_esq"] is True
    assert feats["obstaculo_frente"] is False
    assert feats["obstaculo_dir"] is False


def test_lidar_ignores_non_finite_and_zero_rays():
    n = 360
    scan = _uniform_scan(n, 5.0)
    scan[n // 2] = 0.0           # invalid
    scan[n // 2 + 1] = math.inf  # invalid
    feats = s.compute_features(**_base_kwargs(scan_ranges=scan))
    # The two invalid rays are skipped; remaining rays in the cone are 5.0.
    assert feats["obstaculo_frente"] is False


# --------------------------------------------------------------------------
# Target features — distance + bearing + sectors
# --------------------------------------------------------------------------

def test_dist_alvo_normalization():
    # Robot at origin, target at (6, 0), yaw=0 → distance = 6 → 0.6
    feats = s.compute_features(**_base_kwargs(target_pose=(6.0, 0.0)))
    assert feats["dist_alvo"] == pytest.approx(0.6)


def test_dist_alvo_clips_at_one():
    feats = s.compute_features(**_base_kwargs(target_pose=(100.0, 0.0)))
    assert feats["dist_alvo"] == 1.0


def test_angulo_alvo_depends_on_yaw():
    # Target directly in front: bearing ≈ 0.
    feats = s.compute_features(
        **_base_kwargs(target_pose=(5.0, 0.0), robot_pose=(0.0, 0.0, 0.0)))
    assert abs(feats["angulo_alvo"]) < 1e-6
    assert feats["alvo_frente"] is True

    # Rotate robot by +π/2: target at +x is now to the right (bearing=-π/2).
    feats = s.compute_features(
        **_base_kwargs(target_pose=(5.0, 0.0),
                       robot_pose=(0.0, 0.0, math.pi / 2)))
    assert feats["angulo_alvo"] < 0.0
    assert feats["alvo_dir"] is True
    assert feats["alvo_frente"] is False


def test_alvo_sectors_are_mutually_exclusive():
    # Any yaw → exactly one of frente/esq/dir/atras must be True.
    for yaw in [0.0, math.pi / 2, math.pi, -math.pi / 2]:
        feats = s.compute_features(
            **_base_kwargs(target_pose=(5.0, 0.0),
                           robot_pose=(0.0, 0.0, yaw)))
        flags = [feats["alvo_frente"], feats["alvo_esq"],
                 feats["alvo_dir"], feats["alvo_atras"]]
        assert sum(flags) == 1, f"yaw={yaw} → {flags}"


def test_alvo_atras_when_target_behind():
    # Robot facing +x, target at -x → bearing ≈ π → alvo_atras.
    feats = s.compute_features(
        **_base_kwargs(target_pose=(-5.0, 0.0), robot_pose=(0.0, 0.0, 0.0)))
    assert feats["alvo_atras"] is True
    assert feats["alvo_frente"] is False


def test_alvo_proximo_threshold():
    near = s.compute_features(
        **_base_kwargs(target_pose=(0.3, 0.0)),
        params={"alvo_prox_radius_m": 0.5})
    assert near["alvo_proximo"] is True

    far = s.compute_features(
        **_base_kwargs(target_pose=(2.0, 0.0)),
        params={"alvo_prox_radius_m": 0.5})
    assert far["alvo_proximo"] is False


# --------------------------------------------------------------------------
# Kinetic
# --------------------------------------------------------------------------

def test_velocidade_normalized_and_clipped():
    feats = s.compute_features(
        **_base_kwargs(linear_vel=0.5, angular_vel=-2.0))
    assert feats["velocidade_linear"] == pytest.approx(0.5)
    assert feats["velocidade_angular"] == -1.0  # clipped
