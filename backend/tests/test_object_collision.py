"""Exact occupancy semantics for the Shapely collision adapter."""

from copy import deepcopy

import pytest

from app.services import object_collision
from app.services.object_collision import (
    conflicting_pairs,
    placement_conflicts_any,
    placements_conflict,
)
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


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize(
    "offsets,expected",
    [([], False), ([4, 8], False), ([4, 1], True), ([0], True), ([1, 0], True)],
)
def test_candidate_conflicts(container, offsets, expected):
    candidate = PlacedOutline(rectangle())
    occupied = container(PlacedOutline(rectangle(), x, 0) for x in offsets)
    assert placement_conflicts_any(candidate, occupied) is expected


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize(
    "offsets,expected",
    [
        ([], ()),
        ([0], ()),
        ([0, 4, 8], ()),
        ([0, 1, 8], ((0, 1),)),
        ([0, 0, 0], ((0, 1), (0, 2), (1, 2))),
        ([0, 4, 2, 1], ((0, 2), (0, 3), (1, 2), (2, 3))),
    ],
)
def test_conflicting_pair_indices(container, offsets, expected):
    placements = container(PlacedOutline(rectangle(), x, 0) for x in offsets)
    assert conflicting_pairs(placements) == expected


def test_helpers_delegate_in_index_order(monkeypatch):
    placements = [PlacedOutline(rectangle(), x, 0) for x in (0, 10, 20, 30)]
    calls = []

    def conflict(a, b):
        calls.append((a, b))
        return b is placements[2]

    monkeypatch.setattr(object_collision, "placements_conflict", conflict)
    assert placement_conflicts_any(placements[0], placements[1:]) is True
    assert calls == [(placements[0], placements[1]), (placements[0], placements[2])]
    calls.clear()
    assert conflicting_pairs(placements) == ((0, 2), (1, 2))
    assert calls == [
        (placements[i], placements[j])
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    ]


@pytest.mark.parametrize("container", [tuple, list])
def test_query_helpers_propagate_invalid_geometry(container):
    invalid = PlacedOutline(
        CanonicalOutline.from_points(((0, 0), (4, 4), (0, 4), (3, 0)))
    )
    valid = PlacedOutline(rectangle(), 20, 20)
    for a, b in ((invalid, valid), (valid, invalid), (invalid, invalid)):
        with pytest.raises(ValueError, match="nonempty and valid"):
            placement_conflicts_any(a, container([b]))
        with pytest.raises(ValueError, match="nonempty and valid"):
            conflicting_pairs(container([a, b]))
    assert placement_conflicts_any(invalid, container()) is False
    assert conflicting_pairs(container([invalid])) == ()
    # A conflict ends the candidate scan before the invalid placement is reached.
    assert placement_conflicts_any(valid, container([valid, invalid])) is True
    with pytest.raises(ValueError, match="nonempty and valid"):
        conflicting_pairs(container([valid, valid, invalid]))


@pytest.mark.parametrize("container", [tuple, list])
def test_query_inputs_remain_unchanged(container):
    outline = rectangle(6, 2)
    candidate = PlacedOutline(outline, 10, 5, 1)
    occupied = container([PlacedOutline(outline, 30, 5, 3), candidate, candidate])
    before = deepcopy((candidate, occupied, [p.points for p in occupied], outline.points))
    identities = tuple(id(p) for p in occupied)
    assert placement_conflicts_any(candidate, occupied) is True
    assert conflicting_pairs(occupied) == ((1, 2),)
    assert (candidate, occupied, [p.points for p in occupied], outline.points) == before
    assert tuple(id(p) for p in occupied) == identities
    assert all(p.outline is outline for p in occupied)
