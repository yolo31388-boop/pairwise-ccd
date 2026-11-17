import pytest

from physics.ccd import (
    swept_sphere,
    time_of_impact,
    substeps,
    contact_normal,
    resolve_hit
)


def test_sweep_bounds():
    assert swept_sphere((0.0, 0.0), (0.0, 10.0), 1.0) == (-1.0, -1.0, 1.0, 11.0)

def test_toi_hit():
    assert time_of_impact((0.0, 0.0), (0.0, 10.0), (5.0, 0.0), (5.0, 1.0)) == pytest.approx(0.4)

def test_toi_miss():
    assert time_of_impact((0.0, 0.0), (0.0, 10.0), (5.0, 5.0), (5.0, 6.0)) is None

def test_substeps():
    assert substeps(10.0, 1.0, 2.0) == 5

def test_substeps_min():
    assert substeps(0.1, 1.0, 2.0) == 1

def test_normal_down():
    assert contact_normal((0.0, 0.0), (0.0, 1.0)) == (0.0, -1.0)

def test_normal_up():
    assert contact_normal((0.0, 1.0), (0.0, 0.0)) == (0.0, 1.0)

def test_resolve_toi():
    assert resolve_hit((0.0, 0.0), 0.5) == (0.0, 5.0)

def test_resolve_zero():
    assert resolve_hit((0.0, 0.0), 0.0) == (0.0, 0.0)

def test_resolve_one():
    assert resolve_hit((0.0, 0.0), 1.0) == (0.0, 10.0)

def test_toi_zero():
    assert time_of_impact((0.0, 0.0), (0.0, 10.0), (0.0, 0.0), (0.0, 0.0)) == 0.0

def test_substeps_zero():
    assert substeps(0.0, 1.0, 2.0) == 1

def test_normal_zero():
    assert contact_normal((0.0, 0.0), (0.0, 0.0)) == (0.0, 0.0)

def test_resolve_negative():
    assert resolve_hit((0.0, 0.0), -0.1) == (0.0, 0.0)

def test_resolve_clamp():
    assert resolve_hit((0.0, 0.0), 1.5) == (0.0, 10.0)
