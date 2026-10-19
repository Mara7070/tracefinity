"""Deterministic composition of rectangular fit and polygon collision checks."""

from copy import deepcopy

import pytest

from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates
from app.services.object_placement import nonconflicting_fit_candidates


def rectangle(width=1, height=1):
    return CanonicalOutline.from_points(((0, 0), (width, 0), (width, height), (0, height)))


@pytest.fixture
def triangle():
    return CanonicalOutline.from_points(((0, 0), (6, 0), (0, 4)))


@pytest.mark.parametrize("occupied", [(), []])
@pytest.mark.parametrize("width,height,inset", [(8, 8, 1), (6, 4, 0), (4, 6, 0), (1, 1, 1)])
def test_empty_occupied_matches_direct_fit(triangle, occupied, width, height, inset):
    assert nonconflicting_fit_candidates(triangle, width, height, occupied, inset) == fit_candidates(
        triangle, width, height, inset
    )


def test_preserves_all_quarter_turns_in_order(triangle):
    occupied = [PlacedOutline(rectangle(), 20, 20)]
    result = nonconflicting_fit_candidates(triangle, 8, 8, occupied, 1)
    assert tuple(p.quarter_turns for p in result) == (0, 1, 2, 3)
    assert tuple(p.bounds for p in result) == (
        (1, 1, 7, 5), (1, 1, 5, 7), (1, 1, 7, 5), (1, 1, 5, 7)
    )


def test_filters_one_of_two_orientations(triangle):
    # Only turns 0 and 2 fit. Turn 0 occupies the lower-left corner;
    # turn 2 has its diagonal above this small square.
    occupied = [PlacedOutline(rectangle(0.25, 0.25), 0.25, 0.25)]
    result = nonconflicting_fit_candidates(triangle, 6, 4, occupied)
    assert tuple(p.quarter_turns for p in result) == (2,)
    assert result[0].points == ((0, 4), (6, 0), (6, 4))


def test_preserves_order_after_filtering(triangle):
    # Even turns extend to x=6; odd turns end at x=4.
    occupied = [PlacedOutline(rectangle(0.25, 4), 5, 0)]
    result = nonconflicting_fit_candidates(triangle, 6, 6, occupied)
    assert tuple(p.quarter_turns for p in result) == (1, 3)


def test_filters_all_orientations(triangle):
    assert nonconflicting_fit_candidates(triangle, 6, 6, [PlacedOutline(rectangle(6, 6))]) == ()


@pytest.mark.parametrize("container", [tuple, list])
def test_multiple_occupied_placements(triangle, container):
    # The vertical strip removes even turns; the upper-left square removes
    # turn 3, leaving turn 1's empty upper-left corner.
    occupied = container([
        PlacedOutline(rectangle(0.25, 4), 5, 0),
        PlacedOutline(rectangle(0.25, 0.25), 0.25, 5.25),
    ])
    result = nonconflicting_fit_candidates(triangle, 6, 6, occupied)
    assert tuple(p.quarter_turns for p in result) == (1,)


@pytest.mark.parametrize("x,y", [(2, 0), (2, 2)])
def test_boundary_contact_conflicts(x, y):
    assert nonconflicting_fit_candidates(rectangle(2, 2), 2, 2, [PlacedOutline(rectangle(), x, y)]) == ()


@pytest.mark.parametrize(
    "index,value",
    [(index, value) for index in range(3)
     for value in (float("nan"), float("inf"), -float("inf"), -1, 0)
     if index != 2 or value != 0],
)
def test_dimension_and_inset_validation_delegates(triangle, index, value):
    args = [6, 6, 0]
    args[index] = value
    with pytest.raises(ValueError) as direct:
        fit_candidates(triangle, *args)
    with pytest.raises(ValueError) as filtered:
        nonconflicting_fit_candidates(triangle, args[0], args[1], [], args[2])
    assert str(filtered.value) == str(direct.value)


@pytest.mark.parametrize("invalid_candidate", [False, True])
def test_invalid_polygon_geometry_propagates(triangle, invalid_candidate):
    crossing = CanonicalOutline.from_points(((0, 0), (4, 4), (0, 4), (3, 0)))
    outline = crossing if invalid_candidate else triangle
    occupied = [PlacedOutline(triangle if invalid_candidate else crossing, 20, 20)]
    with pytest.raises(ValueError, match="Placement polygon must be nonempty and valid"):
        nonconflicting_fit_candidates(outline, 6, 6, occupied)


@pytest.mark.parametrize("container", [tuple, list])
def test_inputs_are_unchanged(triangle, container):
    occupied = container([PlacedOutline(rectangle(), 20, 20, 1), PlacedOutline(rectangle(), 30, 30)])
    before = deepcopy((triangle, occupied, tuple(p.points for p in occupied)))
    original_placements = tuple(occupied)
    result = nonconflicting_fit_candidates(triangle, 8, 8, occupied, 1)
    assert (triangle, occupied, tuple(p.points for p in occupied)) == before
    assert all(a is b for a, b in zip(occupied, original_placements))
    assert isinstance(result, tuple)
    assert all(p.outline is triangle for p in result)
