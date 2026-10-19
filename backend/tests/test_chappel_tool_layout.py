"""Explicit Chappel envelopes preserve canonical Tool layout semantics."""

from copy import deepcopy

import pytest

from app.services import chappel_tool_layout
from app.models.schemas import FingerHole, Point, Tool
from app.services.chappel_dimensions import CHAPPEL
from app.services.chappel_tool_layout import (
    first_nonconflicting_tool_in_chappel_organizer as first,
)
from app.services.chappel_tool_layout import fit_tool_in_chappel_organizer as fit
from app.services.chappel_tool_layout import minimal_chappel_organizer_sizes as minimal
from app.services.chappel_tool_layout import (
    nonconflicting_tool_in_chappel_organizer as nonconflicting,
)
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates
from app.services.object_placement import (
    first_nonconflicting_fit_candidate,
    nonconflicting_fit_candidates,
)
from app.services.tool_layout import (
    canonical_outline_for_tool,
    fit_tool_candidates,
    nonconflicting_tool_candidates,
)


def make_tool(coordinates):
    return Tool(id="trace", name="Trace", points=[Point(x=x, y=y) for x, y in coordinates])


@pytest.fixture
def tool():
    return make_tool([(-2, -1), (58, -1), (-2, 19)])


def placement(coordinates, x=0, y=0):
    return PlacedOutline(CanonicalOutline.from_points(coordinates), x, y)


@pytest.mark.parametrize("cells", [(1, 1), (2, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("inset", [0.0, 0.5, 28.0])
@pytest.mark.parametrize("container", [tuple, list])
def test_exact_envelope_and_direct_delegates(tool, cells, inset, container):
    width, height = CHAPPEL.organizer_footprint_xy(*cells)
    occupied = container([placement([(0, 0), (1, 0), (0, 1)], 100, 100)])
    assert fit(tool, *cells, inset) == fit_tool_candidates(tool, width, height, inset)
    outline = canonical_outline_for_tool(tool)
    assert fit(tool, *cells, inset) == fit_candidates(outline, width, height, inset)
    expected = nonconflicting_tool_candidates(tool, width, height, occupied, inset)
    assert expected == nonconflicting_fit_candidates(outline, width, height, occupied, inset)
    assert nonconflicting(tool, *cells, occupied, inset) == expected
    assert first(tool, *cells, occupied, inset) == first_nonconflicting_fit_candidate(
        canonical_outline_for_tool(tool), width, height, occupied, inset
    )
    assert first(tool, *cells, occupied, inset) == (expected[0] if expected else None)


@pytest.mark.parametrize("cells,turns", [
    ((1, 1), ()), ((2, 1), (0, 2)), ((1, 2), (1, 3)), ((2, 3), (0, 1, 2, 3)),
])
def test_xy_orientation_order_and_alignment(tool, cells, turns):
    candidates = fit(tool, *cells, 0.5)
    assert tuple(p.quarter_turns for p in candidates) == turns
    assert all(p.bounds[:2] == (0.5, 0.5) for p in candidates)
    assert first(tool, *cells, []) == (fit(tool, *cells)[0] if turns else None)


def test_exact_organizer_clearance():
    tool = make_tool([(0, 0), (54.7, 0), (54.7, 54.7), (0, 54.7)])
    assert fit(tool, 1, 1) == ()
    tool = make_tool([(0, 0), (54.6, 0), (54.6, 54.6), (0, 54.6)])
    assert len(fit(tool, 1, 1)) == 1


def rectangle_tool(width, height):
    return make_tool([(0, 0), (width, 0), (width, height), (0, height)])


def test_minimal_one_cell_excludes_larger_fitting_envelopes():
    assert minimal(rectangle_tool(20, 10), 4, 3) == ((1, 1),)


def test_minimal_rectangle_returns_both_ties_in_xy_order():
    tool = rectangle_tool(60, 20)
    assert fit(tool, 1, 1) == ()
    assert fit(tool, 1, 2) and fit(tool, 2, 1)
    assert minimal(tool, 3, 3) == ((1, 2), (2, 1))


def test_minimal_asymmetric_bounds_allow_only_one_envelope():
    tool = rectangle_tool(120, 20)
    assert tuple(p.quarter_turns for p in fit(tool, 3, 1)) == (0,)
    assert minimal(tool, 4, 1) == ((3, 1),)


def test_minimal_elongated_envelope_beats_larger_area_square():
    tool = rectangle_tool(100, 20)
    assert fit(tool, 1, 2) and fit(tool, 2, 1) and fit(tool, 2, 2)
    assert minimal(tool, 4, 4) == ((1, 2), (2, 1))


@pytest.mark.parametrize("bounds", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_minimal_never_expands_too_small_bounds(bounds):
    tool = rectangle_tool(120, 20)
    assert fit(tool, 3, 1) and fit(tool, 1, 3)
    assert minimal(tool, *bounds) == ()


def test_minimal_positive_inset_increases_envelope():
    tool = rectangle_tool(54, 20)
    assert fit(tool, 1, 1)
    assert fit(tool, 1, 1, 0.5) == ()
    assert fit(tool, 1, 2, 0.5) and fit(tool, 2, 1, 0.5)
    assert minimal(tool, 3, 3) == ((1, 1),)
    assert minimal(tool, 3, 3, 0.5) == ((1, 2), (2, 1))


@pytest.mark.parametrize("qualifiers,expected", [
    (set(), ()),
    ({(1, 1), (2, 3)}, ((1, 1),)),
    ({(1, 3), (2, 1), (2, 2), (2, 3)}, ((2, 1),)),
    ({(1, 2), (2, 1), (2, 3)}, ((1, 2), (2, 1))),
])
def test_minimal_delegates_every_bounded_pair(monkeypatch, qualifiers, expected):
    # Deliberately invalid geometry: only the fit delegate may interpret it.
    tool = make_tool([])
    inset = 0.75
    calls = []

    def fake_fit(actual_tool, cells_x, cells_y, actual_inset):
        assert actual_tool is tool
        assert actual_inset is inset
        calls.append((cells_x, cells_y))
        return (object(),) if (cells_x, cells_y) in qualifiers else ()

    monkeypatch.setattr(chappel_tool_layout, "fit_tool_in_chappel_organizer", fake_fit)
    assert minimal(tool, 2, 3, inset) == expected
    assert sorted(calls) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0, -1])
def test_minimal_strict_max_bound_validation(tool, axis, invalid, monkeypatch):
    def unexpected_fit(*args):
        pytest.fail("Invalid bounds must be rejected before fitting")

    monkeypatch.setattr(chappel_tool_layout, "fit_tool_in_chappel_organizer", unexpected_fit)
    bounds = [2, 3]
    bounds[axis] = invalid
    before = bounds.copy()
    with pytest.raises(ValueError, match="cell bounds must be Python ints >= 1"):
        minimal(tool, *bounds)
    assert bounds == before


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("x,y,w,h,turns", [
    (20, 20, 1, 1, (0, 1, 2, 3)),
    (0.1, 0.1, 0.1, 0.1, (2,)),
    (6, 0, 0.25, 4, (1, 3)),  # Exact boundary contact filters turns 0 and 2.
    (0, 0, 6, 6, ()),
])
def test_collision_filtering_and_first(container, x, y, w, h, turns):
    tool = make_tool([(0, 0), (6, 0), (0, 4)])
    occupied = container([placement([(0, 0), (w, 0), (w, h), (0, h)], x, y)])
    candidates = nonconflicting(tool, 1, 1, occupied)
    assert tuple(p.quarter_turns for p in candidates) == turns
    assert first(tool, 1, 1, occupied) == (candidates[0] if candidates else None)


ADAPTERS = [lambda tool, x, y, inset=0: fit(tool, x, y, inset),
            lambda tool, x, y, inset=0: minimal(tool, x, y, inset),
            lambda tool, x, y, inset=0: nonconflicting(tool, x, y, [], inset),
            lambda tool, x, y, inset=0: first(tool, x, y, [], inset)]


@pytest.mark.parametrize("adapter", ADAPTERS[:1] + ADAPTERS[2:])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0, -1])
def test_cell_validation_unchanged(tool, adapter, axis, invalid):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError) as expected:
        CHAPPEL.organizer_footprint_xy(*cells)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        adapter(tool, *cells)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("inset", [-1, float("nan"), float("inf"), -float("inf")])
def test_inset_validation_unchanged(tool, adapter, inset):
    with pytest.raises(ValueError) as expected:
        fit_tool_candidates(tool, *CHAPPEL.organizer_footprint_xy(2, 3), inset)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        adapter(tool, 2, 3, inset)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("coordinates", [[], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)]])
def test_invalid_outer_ring_unchanged(adapter, coordinates):
    tool = make_tool(coordinates)
    with pytest.raises(ValueError) as expected:
        canonical_outline_for_tool(tool)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        adapter(tool, 1, 1)


@pytest.mark.parametrize("adapter", [nonconflicting, first])
@pytest.mark.parametrize("invalid_tool", [False, True])
def test_invalid_polygon_unchanged(tool, adapter, invalid_tool):
    crossing = [(0, 0), (4, 4), (0, 4), (3, 0)]
    occupied = [placement(crossing, 100, 100)]
    if invalid_tool:
        tool.points = make_tool(crossing).points
        occupied = [placement([(0, 0), (1, 0), (0, 1)], 100, 100)]
    with pytest.raises(ValueError) as expected:
        nonconflicting_tool_candidates(tool, *CHAPPEL.organizer_footprint_xy(2, 3), occupied)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        adapter(tool, 2, 3, occupied)
    # Collision validation remains lazy when no orientation fits.
    assert adapter(tool, 1, 1, occupied, 28) == (() if adapter is nonconflicting else None)


@pytest.mark.parametrize("container", [tuple, list])
def test_ignored_fields_and_input_immutability(tool, container):
    occupied = container([placement([(0, 0), (1, 0), (0, 1)], 100, 100)])
    expected = (fit(tool, 2, 3), nonconflicting(tool, 2, 3, occupied), first(tool, 2, 3, occupied),
                minimal(tool, 2, 3, 0.5))
    tool.interior_rings = [[Point(x=0, y=0)]]
    tool.finger_holes = [FingerHole(id="hole", x=0, y=0, radius=100)]
    tool.source_image_transform = [2, 0, 0, 2, 100, 200]
    tool.tags = ["tag"]
    tool.project_ids = ["project"]
    tool.smoothed = False
    tool.smooth_level = 0.9
    before = deepcopy((tool.model_dump(), occupied))
    points_list, points, placements = tool.points, tuple(tool.points), tuple(occupied)
    assert (fit(tool, 2, 3), nonconflicting(tool, 2, 3, occupied), first(tool, 2, 3, occupied),
            minimal(tool, 2, 3, 0.5)) == expected
    assert (tool.model_dump(), occupied) == before
    assert tool.points is points_list
    assert all(a is b for a, b in zip(tool.points, points))
    assert all(a is b for a, b in zip(occupied, placements))
