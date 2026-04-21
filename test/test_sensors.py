"""Unit tests for utils.sensors — pure sensor → feature-dict pipeline."""
import math

import numpy as np
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
        labels_map=None,
        enemy_flag_pose=(8.0, 0.0),
        own_base_pose=(6.0, 0.0),
        enemy_base_pose=(-6.0, 0.0),
        deploy_zone_pose=(-8.0, -0.5),
        holding_flag=False,
        enemy_flag_label=20,
        enemy_base_label=10,
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
                     "a": {"term": "bandeira_inimiga_visivel"},
                     "b": {"op": "LT",
                           "a": {"term": "dist_frente"},
                           "b": {"erc": 0.5}}},
            "then": {"leaf": "PEGAR", "dur_ms": 200},
            "else": {"leaf": "FRENTE", "dur_ms": 300}}
    feats = s.compute_features(**_base_kwargs())
    out = g.evaluate(tree, feats)
    assert out in [("PEGAR", 200), ("FRENTE", 300)]


# --------------------------------------------------------------------------
# Lidar cones — frontal obstacle
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
# Camera — labels_map
# --------------------------------------------------------------------------

def test_camera_no_frame_means_nothing_visible():
    feats = s.compute_features(**_base_kwargs(labels_map=None))
    assert feats["bandeira_inimiga_visivel"] is False
    assert feats["base_inimiga_visivel"] is False
    assert feats["angulo_bandeira_inimiga"] == 0.0


def test_camera_detects_flag_blob_on_right():
    lm = np.zeros((240, 320), dtype=np.uint8)
    # Paint a blob at x ≈ 280 (rightmost third)
    lm[100:150, 270:290] = 20
    feats = s.compute_features(
        **_base_kwargs(labels_map=lm, enemy_flag_label=20))
    assert feats["bandeira_inimiga_visivel"] is True
    assert feats["bandeira_direita"] is True
    assert feats["bandeira_esquerda"] is False
    assert feats["bandeira_centralizada"] is False
    assert feats["angulo_bandeira_inimiga"] > 0.0


def test_camera_detects_flag_blob_centered():
    lm = np.zeros((240, 320), dtype=np.uint8)
    lm[100:150, 150:170] = 20
    feats = s.compute_features(
        **_base_kwargs(labels_map=lm, enemy_flag_label=20))
    assert feats["bandeira_inimiga_visivel"] is True
    assert feats["bandeira_centralizada"] is True
    assert abs(feats["angulo_bandeira_inimiga"]) < 0.1


def test_camera_detects_enemy_base_by_label():
    lm = np.zeros((240, 320), dtype=np.uint8)
    lm[50:100, 50:100] = 10
    feats = s.compute_features(
        **_base_kwargs(labels_map=lm, enemy_base_label=10))
    assert feats["base_inimiga_visivel"] is True
    assert feats["bandeira_inimiga_visivel"] is False


# --------------------------------------------------------------------------
# Spatial features — distances and relative angles
# --------------------------------------------------------------------------

def test_spatial_distance_normalization():
    # Robot at origin, own base at (6, 0) → distance = 6m → normalized = 0.6
    feats = s.compute_features(**_base_kwargs(own_base_pose=(6.0, 0.0)))
    assert feats["dist_base_propria"] == pytest.approx(0.6)


def test_spatial_distance_clips_at_one():
    feats = s.compute_features(
        **_base_kwargs(own_base_pose=(100.0, 0.0)))
    assert feats["dist_base_propria"] == 1.0


def test_spatial_relative_angle_accounts_for_yaw():
    # Robot facing +x; deploy at (-8, -0.5) → bearing ~ π (behind, slight right)
    kw = _base_kwargs(robot_pose=(0.0, 0.0, 0.0),
                      deploy_zone_pose=(-1.0, 0.0))
    feats = s.compute_features(**kw)
    # Target directly behind → bearing π, normalized to 1.0 (or -1.0).
    assert abs(abs(feats["angulo_zona_deploy"]) - 1.0) < 1e-6

    # Rotate robot to face target (+π yaw) → target now in front, angle ≈ 0.
    kw2 = _base_kwargs(robot_pose=(0.0, 0.0, math.pi),
                       deploy_zone_pose=(-1.0, 0.0))
    feats2 = s.compute_features(**kw2)
    assert abs(feats2["angulo_zona_deploy"]) < 1e-6


def test_na_base_triggers_within_threshold():
    feats = s.compute_features(
        **_base_kwargs(own_base_pose=(0.5, 0.0)),
        params={"at_base_threshold_m": 1.0},
    )
    assert feats["na_base_propria"] is True
    assert feats["na_zona_deploy"] is False


# --------------------------------------------------------------------------
# Kinetic
# --------------------------------------------------------------------------

def test_velocidade_normalized_and_clipped():
    feats = s.compute_features(
        **_base_kwargs(linear_vel=0.5, angular_vel=-2.0),
    )
    assert feats["velocidade_linear"] == pytest.approx(0.5)
    assert feats["velocidade_angular"] == -1.0  # clipped


# --------------------------------------------------------------------------
# Holding flag passthrough
# --------------------------------------------------------------------------

def test_holding_flag_reflects_input():
    for val in (True, False):
        feats = s.compute_features(**_base_kwargs(holding_flag=val))
        assert feats["segurando_bandeira"] is val
