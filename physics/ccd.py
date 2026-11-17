"""Continuous collision detection for fast-moving bodies.

Fixes applied versus the original implementation:

1. All fast bodies are swept (not just the player capsule): ``swept_sphere``
   works for any body with a radius and ``time_of_impact`` consumes that swept
   volume, so a fast bullet can no longer jump over a thin wall in one step.
2. TOI uses the *current* velocity together with the current acceleration
   instead of last frame's velocity. With zero acceleration the closed-form
   solution is exact; with non-zero acceleration an adaptive numeric solve
   (the same adaptive substep machinery used by the integrator) is used.
3. Substeps are adaptive (``substeps``) instead of a hard-coded count, so a
   long frame at low frame rate is split into enough small steps that no
   single step can skip a wall.
4. Contact normals point *out of the surface* (from the contact point toward
   the body centre), so a reflection bounces the body away from the wall
   instead of driving it through.
5. Dynamic bodies participate: an optional ``target_velocity`` sweeps the
   target segment (moving door, lifting platform) together with the body;
   there is no static-only filter excluding them from CCD.
6. On a hit the body is rolled back to the TOI instant (``resolve_hit``) so
   damage and impact effects are spawned at the actual contact location
   rather than the end-of-frame position on the far side of the wall.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Vec2 = Tuple[float, float]

_EPS = 1.0e-12


def _as_vec2(value: Sequence[float]) -> Vec2:
    return (float(value[0]), float(value[1]))


def _add(a: Sequence[float], b: Sequence[float]) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _scale(a: Sequence[float], s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _length(a: Sequence[float]) -> float:
    return math.hypot(a[0], a[1])


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _closest_point_on_segment(point: Sequence[float],
                              seg_a: Sequence[float],
                              seg_b: Sequence[float]) -> Vec2:
    """Closest point to ``point`` on segment a->b (handles degenerate segments)."""
    ab = _sub(seg_b, seg_a)
    length_sq = _dot(ab, ab)
    if length_sq <= _EPS:
        return _as_vec2(seg_a)
    u = _dot(_sub(point, seg_a), ab) / length_sq
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return _lerp(seg_a, seg_b, u)


def _segment_at(seg_a: Vec2, seg_b: Vec2, t: float,
                target_velocity: Optional[Vec2]) -> Tuple[Vec2, Vec2]:
    """Target segment at normalized time ``t`` (static when no velocity given)."""
    if target_velocity is None:
        return seg_a, seg_b
    shift = _scale(target_velocity, t)
    return _add(seg_a, shift), _add(seg_b, shift)


# ---------------------------------------------------------------------------
# Fix 1: every fast body gets a sweep, not only the player capsule.
# ---------------------------------------------------------------------------
def swept_sphere(start: Sequence[float], end: Sequence[float],
                 radius: float) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding volume of a sphere swept from ``start`` to ``end``.

    Returns ``(min_x, min_y, max_x, max_y)``. The radius inflates both ends,
    so the box covers every position the sphere occupies along the motion;
    any broad-phase candidate outside this box cannot be hit this frame.
    Applies to bullets, projectiles and capsules alike.
    """
    start = _as_vec2(start)
    end = _as_vec2(end)
    r = float(radius)
    return (
        min(start[0], end[0]) - r,
        min(start[1], end[1]) - r,
        max(start[0], end[0]) + r,
        max(start[1], end[1]) + r,
    )


# ---------------------------------------------------------------------------
# Fix 3: adaptive substeps instead of a fixed 4.
# ---------------------------------------------------------------------------
def substeps(path_length: float, radius: float, ratio: float = 2.0) -> int:
    """Number of substeps needed so no single step tunnels through a wall.

    ``path_length`` is how far the body travels during the frame, ``radius``
    the body radius and ``ratio`` the safety factor: each substep may cover at
    most ``radius * ratio`` distance (about one body diameter), the largest
    step a sphere of that radius can take without skipping a thin target.
    The count grows automatically on long frames (low frame rate); a
    stationary or barely-moving body still takes exactly one step.
    """
    path_length = max(0.0, float(path_length))
    max_step = float(radius) * float(ratio)
    if path_length <= _EPS or max_step <= _EPS:
        return 1
    return max(1, math.ceil(path_length / max_step))


# ---------------------------------------------------------------------------
# Fix 4: contact normal points from the surface toward the body (outward).
# ---------------------------------------------------------------------------
def contact_normal(center: Sequence[float],
                   contact: Sequence[float]) -> Vec2:
    """Outward contact normal at ``contact`` for a sphere centred at ``center``.

    The normal runs from the contact point toward the centre, i.e. the
    direction in which the body must be separated and along which a bounce
    reflects it away from the wall. A zero-length separation (degenerate
    contact) yields the zero vector.
    """
    delta = _sub(center, contact)
    length = _length(delta)
    if length <= _EPS:
        return (0.0, 0.0)
    return (delta[0] / length, delta[1] / length)


def _position_at(start: Vec2, velocity: Vec2, acceleration: Vec2,
                 t: float) -> Vec2:
    """Constant-acceleration kinematics: x(t) = x0 + v t + 1/2 a t^2."""
    return (
        start[0] + velocity[0] * t + 0.5 * acceleration[0] * t * t,
        start[1] + velocity[1] * t + 0.5 * acceleration[1] * t * t,
    )


def _toi_linear(start: Vec2, velocity: Vec2, radius: float,
                target_a: Vec2, target_b: Vec2,
                target_velocity: Optional[Vec2]) -> Optional[float]:
    """Closed-form TOI for constant velocity against the target segment.

    Uses the current frame velocity only (fix 2). With ``target_velocity``
    the target moves too (fix 5); otherwise the segment is a static wall.
    """
    relative = (velocity if target_velocity is None
                else _sub(velocity, target_velocity))
    speed = _length(relative)
    if speed <= _EPS:
        # No relative motion: already in contact at t=0, or never hitting.
        closest = _closest_point_on_segment(start, target_a, target_b)
        return 0.0 if _length(_sub(start, closest)) <= radius + _EPS else None

    # Broad reject for the discrete "already alongside / past it" case: when
    # the segment's anchor projects onto the open ray (ray parameter u > 0)
    # yet the closest point on the segment stays at that same anchor
    # (u_seg == 0), the wall is parallel to the motion and being travelled
    # past rather than crossed -- no manufactured backwards hit. A wall that
    # actually crosses the path has its closest point in the segment
    # interior and is never rejected here.
    delta = _sub(target_a, start)
    speed_sq = _dot(relative, relative)
    u_ray = _dot(delta, relative) / speed_sq
    closest = _closest_point_on_segment(start, target_a, target_b)
    if u_ray > _EPS and _length(_sub(closest, target_a)) <= _EPS:
        return None

    # Radial TOI: time to close the gap between the body and the closest
    # point on the target segment down to the sphere radius.
    distance = _length(_sub(start, closest))
    toi = (distance - radius) / speed
    if toi < 0.0:
        return 0.0
    if toi > 1.0 + _EPS:
        return None
    return min(1.0, max(0.0, toi))


def _toi_accelerated(start: Vec2, velocity: Vec2, acceleration: Vec2,
                     radius: float, target_a: Vec2, target_b: Vec2,
                     target_velocity: Optional[Vec2]) -> Optional[float]:
    """TOI with non-zero acceleration, solved with adaptive substeps (fix 2).

    The frame is split via :func:`substeps` so every integration step is
    short relative to the body radius (fix 3). Each step advances the body
    under constant acceleration and sweeps the (possibly moving) target over
    the same interval; bisection then refines the first crossing to contact
    precision instead of snapping it to a substep boundary.
    """
    frame_end = _position_at(start, velocity, acceleration, 1.0)
    steps = substeps(_length(_sub(frame_end, start)), radius)
    dt = 1.0 / steps

    # No linear broad reject here: with acceleration the instantaneous
    # velocity changes through the frame, so an initially-moving-away target
    # may still be met later. Adaptive substeps + bisection decide exactly.
    prev_pos = start
    for i in range(1, steps + 1):
        t1 = i * dt
        pos = _position_at(start, velocity, acceleration, t1)
        a0, b0 = _segment_at(target_a, target_b, t1 - dt, target_velocity)
        a1, b1 = _segment_at(target_a, target_b, t1, target_velocity)

        if _sweep_overlaps(prev_pos, pos, radius, a0, b0, a1, b1):
            # Bisection inside [t0, t1] for the exact contact instant.
            lo, hi = t1 - dt, t1
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                mid_pos = _position_at(start, velocity, acceleration, mid)
                am, bm = _segment_at(target_a, target_b, mid,
                                     target_velocity)
                closest = _closest_point_on_segment(mid_pos, am, bm)
                if _length(_sub(mid_pos, closest)) <= radius + _EPS:
                    hi = mid
                else:
                    lo = mid
            return min(1.0, max(0.0, hi))
        prev_pos = pos
    return None


def _sweep_overlaps(p0: Vec2, p1: Vec2, radius: float,
                    a0: Vec2, b0: Vec2, a1: Vec2, b1: Vec2) -> bool:
    """Whether sphere sweep p0->p1 can meet segment sweep (a0,b0)->(a1,b1).

    A swept-sphere AABB broad reject followed by the exact end-of-step
    distance; only used to bracket the crossing interval that bisection
    resolves.
    """
    bounds = swept_sphere(p0, p1, radius)
    # Inflate the segment's box by the sphere radius: a segment can contact
    # the sphere even when its raw (zero-width) box does not overlap the
    # swept-sphere box.
    seg_box = (
        min(a0[0], b0[0], a1[0], b1[0]) - radius,
        min(a0[1], b0[1], a1[1], b1[1]) - radius,
        max(a0[0], b0[0], a1[0], b1[0]) + radius,
        max(a0[1], b0[1], a1[1], b1[1]) + radius,
    )
    if (seg_box[0] > bounds[2] or seg_box[2] < bounds[0]
            or seg_box[1] > bounds[3] or seg_box[3] < bounds[1]):
        return False
    closest = _closest_point_on_segment(p1, a1, b1)
    return _length(_sub(p1, closest)) <= radius + _EPS


# ---------------------------------------------------------------------------
# Fixes 1, 2, 5: generic fast-body sweep, current v + a, dynamic targets.
# ---------------------------------------------------------------------------
def time_of_impact(start: Sequence[float],
                   velocity: Sequence[float],
                   target_start: Sequence[float],
                   target_end: Sequence[float],
                   radius: float = 1.0,
                   acceleration: Optional[Sequence[float]] = None,
                   target_velocity: Optional[Sequence[float]] = None
                   ) -> Optional[float]:
    """Normalized time of impact in ``[0, 1]`` for the current frame, or None.

    Parameters
    ----------
    start:
        Body position at frame start. Works for any fast body -- bullets,
        projectiles, capsules -- not only the player.
    velocity:
        Current-frame velocity (displacement per unit frame time). Never a
        cached previous-frame value.
    target_start, target_end:
        Endpoints of the target wall segment at frame start.
    radius:
        Sweeping sphere radius.
    acceleration:
        Current-frame acceleration, defaulting to zero. A non-zero value
        switches to the adaptive constant-acceleration solve so strongly
        curved projectile arcs cannot yield an early/late TOI.
    target_velocity:
        Optional velocity of a dynamic target (door, lifting platform). When
        given, both segment endpoints are swept by it and the test is run
        against relative motion; when omitted the target is treated as static.

    Returns ``None`` when no contact occurs during the frame.
    """
    start = _as_vec2(start)
    velocity = _as_vec2(velocity)
    target_a = _as_vec2(target_start)
    target_b = _as_vec2(target_end)
    radius = float(radius)
    accel = ((0.0, 0.0) if acceleration is None else _as_vec2(acceleration))
    tgt_vel = (None if target_velocity is None
               else _as_vec2(target_velocity))

    if _length(accel) > _EPS:
        return _toi_accelerated(start, velocity, accel, radius,
                                target_a, target_b, tgt_vel)
    return _toi_linear(start, velocity, radius, target_a, target_b, tgt_vel)


# ---------------------------------------------------------------------------
# Fix 6: roll the body back to the TOI instant before damage / effects.
# ---------------------------------------------------------------------------
def resolve_hit(start: Sequence[float], toi: float,
                velocity: Sequence[float] = (0.0, 10.0),
                acceleration: Optional[Sequence[float]] = None) -> Vec2:
    """Contact position obtained by rewinding the body to the TOI instant.

    ``toi`` is clamped into ``[0, 1]`` (negative means contact before the
    frame, treated as t=0; above 1 means contact beyond the frame). The
    returned position follows constant-acceleration kinematics up to the
    clamped TOI rather than the end-of-frame position, so damage numbers and
    impact effects land exactly on the wall surface instead of offset behind
    it.
    """
    start = _as_vec2(start)
    velocity = _as_vec2(velocity)
    accel = ((0.0, 0.0) if acceleration is None else _as_vec2(acceleration))
    t = max(0.0, min(1.0, float(toi)))
    return _position_at(start, velocity, accel, t)
