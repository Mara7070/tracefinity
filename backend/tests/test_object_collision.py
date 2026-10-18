"""Exact occupancy semantics for the Shapely collision adapter."""

from copy import deepcopy

import pytest

from app.services.object_collision import placements_conflict
from app.services.object_geometry import CanonicalOutline, PlacedOutline


def rectangle(width=2, height=2):
    return CanonicalOutline.from_points(((0, 0), (width, 0), (width, height), (0, height)))


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (4, 0, False),  # separated
        (1, 1, True),  # positive-area overlap
        (0, 0, True),  # coincident
        (2, 0, True),  # shared edge
        (2, 2, True),  # shared corner
        (2 + 1e-12, 0, False),  # no implicit clearance or tolerance
    ],
)
def test_square_occupancies(x, y, expected):
    outline = rectangle()
    a, b = PlacedOutline(outline), PlacedOutline(outline, x, y)
    assert placements_conflict(a, b) is expected
    assert placements_conflict(b, a) is expected


def test_containment():
    outer = PlacedOutline(rectangle(10, 10))
    inner = PlacedOutline(rectangle(), 3, 3)
    assert placements_conflict(outer, inner) is True
    assert placements_conflict(inner, outer) is True


def test_rotated_and_translated_geometry():
    # After rotation and translation the long rectangle occupies [8, 10] x [5, 11].
    outline = rectangle(6, 2)
    rotated = PlacedOutline(outline, 10, 5, 1)
    assert placements_conflict(rotated, PlacedOutline(rectangle(1, 1), 8.5, 9)) is True
    assert placements_conflict(rotated, PlacedOutline(rectangle(1, 1), 12, 5)) is False
    assert placements_conflict(rotated, PlacedOutline(outline)) is False


def test_concave_outline_notch_and_overlap():
    elbow = CanonicalOutline.from_points(
        ((0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6))
    )
    concave = PlacedOutline(elbow)
    in_notch = PlacedOutline(rectangle(), 3, 3)
    overlapping_arm = PlacedOutline(rectangle(), 1, 3)
    assert placements_conflict(concave, in_notch) is False
    assert placements_conflict(in_notch, concave) is False
    assert placements_conflict(concave, overlapping_arm) is True
    assert placements_conflict(overlapping_arm, concave) is True


def test_self_intersection_is_rejected_in_either_argument():
    # Unlike a symmetric bow tie, this crossing ring has nonzero signed area
    # and therefore survives CanonicalOutline's basic validation.
    crossing = CanonicalOutline.from_points(((0, 0), (4, 4), (0, 4), (3, 0)))
    invalid = PlacedOutline(crossing)
    valid = PlacedOutline(rectangle(), 20, 20)
    for a, b in ((invalid, valid), (valid, invalid), (invalid, invalid)):
        with pytest.raises(ValueError, match="nonempty and valid"):
            placements_conflict(a, b)


def test_inputs_remain_unchanged():
    outline = rectangle(6, 2)
    a, b = PlacedOutline(outline, 10, 5, 1), PlacedOutline(outline, 9, 6, 3)
    before = deepcopy((a, b, a.points, b.points, outline.points))
    placements_conflict(a, b)
    assert (a, b, a.points, b.points, outline.points) == before
    assert a.outline is outline
    assert b.outline is outline
