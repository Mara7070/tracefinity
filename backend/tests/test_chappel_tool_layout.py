"""Explicit Chappel envelopes preserve canonical Tool layout semantics."""

from copy import deepcopy

import pytest

from app.models.schemas import FingerHole, Point, Tool
from app.services import chappel_tool_layout
from app.services.chappel_dimensions import CHAPPEL
from app.services.chappel_tool_layout import edge_anchor_tool_candidates as edge_candidates
from app.services.chappel_tool_layout import (
    first_minimal_chappel_organizer_fit as first_minimal,
)
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


@pytest.mark.parametrize("cells", [(1, 1), (2, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("inset", [0, 0.5, 28])
def test_edge_empty_matches_single_fit(tool, cells, inset):
    assert edge_candidates(tool, *cells, [], inset, 1) == fit(tool, *cells, inset)


@pytest.mark.parametrize("container", [list, tuple])
def test_edge_rectangle_spacing_dedup_and_immutability(container):
    tool = rectangle_tool(10, 5)
    occupied = container([placement([(0, 0), (10, 0), (10, 5), (0, 5)])] * 2)
    before = deepcopy((tool.model_dump(), occupied))
    result = edge_candidates(tool, 1, 1, occupied, gap=1)
    assert [(p.quarter_turns, p.bounds[:2]) for p in result] == [
        (turn, anchor) for turn in (0, 1) for anchor in ((11, 0), (0, 6), (11, 6))
    ]
    assert len({p.points for p in result}) == len(result)
    assert edge_candidates(tool, 1, 1, occupied) == ()
    assert (tool.model_dump(), occupied) == before


def test_edge_offset_outline_and_container_overflow():
    tool = make_tool([(-12, 8), (-2, 8), (-2, 13), (-12, 13)])
    occupied = [placement([(0, 0), (45, 0), (45, 45), (0, 45)])]
    result = edge_candidates(tool, 1, 1, occupied, inset=2, gap=1)
    assert [(p.quarter_turns, p.bounds) for p in result] == [
        (0, (2, 46, 12, 51)), (1, (46, 2, 51, 12))
    ]


def test_edge_asymmetric_orientation_and_position_order(tool):
    # A small offset triangle keeps all four rotations distinct.
    tool = make_tool([(-2, 3), (4, 3), (-2, 7)])
    occupied = [placement([(0, 0), (10, 0), (10, 10), (0, 10)])]
    result = edge_candidates(tool, 1, 1, occupied, inset=2, gap=1)
    assert [(p.quarter_turns, p.bounds[:2]) for p in result] == [
        (turn, anchor) for turn in range(4) for anchor in ((11, 2), (2, 11), (11, 11))
    ]


def test_edge_occupied_anchor_equal_to_inset_is_deduped():
    occupied = [placement([(-2, -2), (0, -2), (0, 0), (-2, 0)])]
    tool = rectangle_tool(5, 5)
    assert edge_candidates(tool, 1, 1, occupied, inset=1, gap=1) == fit(tool, 1, 1, 1)


@pytest.mark.parametrize("gap", [True, False, -1, float("nan"), float("inf"),
                                 -float("inf"), "1", None, 1j])
def test_edge_invalid_gap(gap):
    with pytest.raises(ValueError, match="Gap must be a finite non-negative real"):
        edge_candidates(rectangle_tool(5, 5), 1, 1, [], gap=gap)


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


@pytest.mark.parametrize("width,height,bounds,envelope,turn", [
    (20, 10, (4, 3), (1, 1), 0),
    (60, 20, (3, 3), (1, 2), 1),
    (120, 20, (4, 1), (3, 1), 0),
])
def test_first_minimal_envelope_and_fit_order(width, height, bounds, envelope, turn):
    tool = rectangle_tool(width, height)
    before = deepcopy(tool.model_dump())
    candidates = fit(tool, *envelope, 0.5)
    result = first_minimal(tool, *bounds, 0.5)
    assert result == (*envelope, candidates[0])
    assert result[2].quarter_turns == turn
    assert tool.model_dump() == before


def test_first_minimal_no_fitting_envelope():
    assert first_minimal(rectangle_tool(120, 20), 2, 2) is None


@pytest.mark.parametrize("sizes", [(), ((1, 2), (2, 1)), ((2, 1), (1, 2))])
@pytest.mark.parametrize("empty_fit", [False, True])
def test_first_minimal_exact_delegation(monkeypatch, sizes, empty_fit):
    tool = make_tool([])  # Validation belongs exclusively to the delegates.
    inset = 0.75
    calls = []
    candidates = (placement([(0, 0), (6, 0), (0, 4)]),
                  placement([(0, 0), (3, 0), (0, 2)]))
    before = deepcopy(candidates)

    def fake_minimal(actual_tool, max_x, max_y, actual_inset):
        assert actual_tool is tool
        assert actual_inset is inset
        calls.append(("minimal", max_x, max_y))
        return sizes

    def fake_fit(actual_tool, cells_x, cells_y, actual_inset):
        assert actual_tool is tool
        assert actual_inset is inset
        calls.append(("fit", cells_x, cells_y))
        return () if empty_fit else candidates

    monkeypatch.setattr(chappel_tool_layout, "minimal_chappel_organizer_sizes", fake_minimal)
    monkeypatch.setattr(chappel_tool_layout, "fit_tool_in_chappel_organizer", fake_fit)
    if sizes and empty_fit:
        with pytest.raises(RuntimeError, match="minimum Chappel envelope"):
            first_minimal(tool, 3, 4, inset)
    else:
        result = first_minimal(tool, 3, 4, inset)
        if sizes:
            assert result[:2] == (1, 2)
            assert result[2] is candidates[0]
        else:
            assert result is None
    assert calls == [("minimal", 3, 4)] + ([("fit", 1, 2)] if sizes else [])
    assert candidates == before


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0, -1])
def test_first_minimal_bound_validation_unchanged(tool, axis, invalid):
    bounds = [2, 3]
    bounds[axis] = invalid
    with pytest.raises(ValueError) as expected:
        minimal(tool, *bounds)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        first_minimal(tool, *bounds)


@pytest.mark.parametrize("delegate", ["minimal_chappel_organizer_sizes", "fit_tool_in_chappel_organizer"])
def test_first_minimal_propagates_delegate_exception(monkeypatch, tool, delegate):
    error = ValueError("delegated validation failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(chappel_tool_layout, "minimal_chappel_organizer_sizes", lambda *args: ((1, 1),))
    monkeypatch.setattr(chappel_tool_layout, delegate, fail)
    with pytest.raises(ValueError) as raised:
        first_minimal(tool, 2, 3)
    assert raised.value is error


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
            lambda tool, x, y, inset=0: first(tool, x, y, [], inset),
            lambda tool, x, y, inset=0: edge_candidates(tool, x, y, [], inset)]


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


@pytest.mark.parametrize("adapter", ADAPTERS + [first_minimal])
@pytest.mark.parametrize("inset", [-1, float("nan"), float("inf"), -float("inf")])
def test_inset_validation_unchanged(tool, adapter, inset):
    with pytest.raises(ValueError) as expected:
        fit_tool_candidates(tool, *CHAPPEL.organizer_footprint_xy(2, 3), inset)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        adapter(tool, 2, 3, inset)


@pytest.mark.parametrize("adapter", ADAPTERS + [first_minimal])
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
                minimal(tool, 2, 3, 0.5), first_minimal(tool, 2, 3, 0.5))
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
            minimal(tool, 2, 3, 0.5), first_minimal(tool, 2, 3, 0.5)) == expected
    assert (tool.model_dump(), occupied) == before
    assert tool.points is points_list
    assert all(a is b for a, b in zip(tool.points, points))
    assert all(a is b for a, b in zip(occupied, placements))


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("fail_at", [None, 1, 2])
def test_greedy_exact_delegation_and_immutability(monkeypatch, container, fail_at):
    tools = container([rectangle_tool(9, 4), rectangle_tool(3, 8), rectangle_tool(6, 2)])
    candidates = tuple(
        (placement([(0, 0), (size, 0), (0, size)]),
         placement([(0, 0), (1, 0), (0, 1)], 40, 40))
        for size in (2, 3, 4)
    )
    before = deepcopy(([tool.model_dump() for tool in tools], candidates))
    tool_refs = tuple(tools)
    point_lists = [tool.points for tool in tools]
    calls = []
    gap, inset = 1.25, 0.75

    def delegate(tool, cells_x, cells_y, occupied, actual_inset, actual_gap):
        index = len(calls)
        assert tool is tools[index]
        assert (cells_x, cells_y) == (2, 3)
        assert actual_inset is inset
        assert actual_gap is gap
        assert isinstance(occupied, list)
        assert len(occupied) == index
        assert all(p is candidates[i][0] for i, p in enumerate(occupied))
        calls.append(tool)
        return () if index == fail_at else candidates[index]

    monkeypatch.setattr(chappel_tool_layout, "edge_anchor_tool_candidates", delegate)
    result = chappel_tool_layout.place_tools_in_chappel_organizer(tools, 2, 3, gap, inset)
    assert len(calls) == (3 if fail_at is None else fail_at + 1)
    if fail_at is None:
        assert isinstance(result, tuple)
        assert all(p is candidates[i][0] for i, p in enumerate(result))
    else:
        assert result is None
    assert ([tool.model_dump() for tool in tools], candidates) == before
    assert all(a is b for a, b in zip(tools, tool_refs))
    assert all(tool.points is points for tool, points in zip(tools, point_lists))


@pytest.mark.parametrize("dimensions,expected", [
    ([(20, 10), (20, 10), (10, 10)],
     [(0, 0, 20, 10), (21, 0, 41, 10), (42, 0, 52, 10)]),
    ([(40, 20), (40, 20)], [(0, 0, 40, 20), (0, 21, 40, 41)]),
])
def test_greedy_rectangles_and_repeatability(dimensions, expected):
    tools = [rectangle_tool(*size) for size in dimensions]
    before = deepcopy(tools)
    result = chappel_tool_layout.place_tools_in_chappel_organizer(tools, 1, 1, 1)
    assert [p.bounds for p in result] == expected
    assert chappel_tool_layout.place_tools_in_chappel_organizer(tools, 1, 1, 1) == result
    assert tools == before


def test_greedy_empty_skips_delegate(monkeypatch):
    def unexpected(*args):
        pytest.fail("Empty input must not delegate")

    monkeypatch.setattr(chappel_tool_layout, "edge_anchor_tool_candidates", unexpected)
    assert chappel_tool_layout.place_tools_in_chappel_organizer([], 0, 0, -1) == ()


@pytest.mark.parametrize("tool,cells,inset,gap", [
    (rectangle_tool(5, 5), (0, 1), 0, 1),
    (rectangle_tool(5, 5), (1, 1), -1, 1),
    (rectangle_tool(5, 5), (1, 1), 0, -1),
    (make_tool([]), (1, 1), 0, 1),
])
def test_greedy_delegated_validation(tool, cells, inset, gap):
    with pytest.raises(ValueError) as expected:
        edge_candidates(tool, *cells, [], inset, gap)
    with pytest.raises(type(expected.value)) as raised:
        chappel_tool_layout.place_tools_in_chappel_organizer([tool], *cells, gap, inset)
    assert str(raised.value) == str(expected.value)


def test_greedy_preserves_delegate_exception_identity(monkeypatch):
    error = ValueError("delegated failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(chappel_tool_layout, "edge_anchor_tool_candidates", fail)
    with pytest.raises(ValueError) as raised:
        chappel_tool_layout.place_tools_in_chappel_organizer([rectangle_tool(5, 5)], 1, 1, 1)
    assert raised.value is error


minimal_tools = chappel_tool_layout.minimal_chappel_organizer_sizes_for_tools


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("dimensions,bounds,expected", [
    ([(20, 10), (20, 10)], (3, 3), ((1, 1),)),
    ([(40, 20), (40, 20)], (3, 3), ((1, 1),)),
    ([(50, 50), (50, 50)], (3, 3), ((1, 2), (2, 1))),
    ([(100, 20), (100, 20)], (3, 3), ((1, 2), (2, 1))),
    ([(120, 20), (120, 20)], (2, 2), ()),
])
def test_minimal_tools_real_rectangles(container, dimensions, bounds, expected):
    tools = container(rectangle_tool(*size) for size in dimensions)
    before = deepcopy(tools)
    refs = tuple(tools)
    point_lists = [tool.points for tool in tools]
    assert minimal_tools(tools, *bounds, 1) == expected
    assert minimal_tools(tools, *bounds, 1) == expected
    assert tools == before
    assert all(a is b for a, b in zip(tools, refs))
    assert all(tool.points is points for tool, points in zip(tools, point_lists))
    if dimensions[0] == (100, 20):
        assert chappel_tool_layout.place_tools_in_chappel_organizer(tools, 2, 2, 1) is not None
    if not expected:
        assert chappel_tool_layout.place_tools_in_chappel_organizer(tools, 3, 1, 1) is not None


@pytest.mark.parametrize("qualifiers,expected", [
    (set(), ()),
    ({(1, 1), (2, 3)}, ((1, 1),)),
    ({(1, 3), (2, 1), (2, 2)}, ((2, 1),)),
    ({(1, 2), (2, 1), (2, 3)}, ((1, 2), (2, 1))),
])
def test_minimal_tools_exact_delegation(monkeypatch, qualifiers, expected):
    tools = [make_tool([]), rectangle_tool(5, 5)]
    gap, inset = 1.25, 0.75
    calls = []

    def delegate(actual_tools, x, y, actual_gap, actual_inset):
        assert actual_tools is tools
        assert actual_gap is gap
        assert actual_inset is inset
        calls.append((x, y))
        # Even a falsey result qualifies when it is not None.
        return () if (x, y) in qualifiers else None

    monkeypatch.setattr(chappel_tool_layout, "place_tools_in_chappel_organizer", delegate)
    assert minimal_tools(tools, 2, 3, gap, inset) == expected
    assert calls == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0, -1, None, "2"])
def test_minimal_tools_strict_bounds(monkeypatch, axis, invalid):
    def unexpected(*args):
        pytest.fail("Invalid bounds must not delegate")

    monkeypatch.setattr(chappel_tool_layout, "place_tools_in_chappel_organizer", unexpected)
    bounds = [2, 3]
    bounds[axis] = invalid
    with pytest.raises(ValueError, match="cell bounds must be Python ints >= 1"):
        minimal_tools([rectangle_tool(5, 5)], *bounds, 1)


@pytest.mark.parametrize("tools", [[], ()])
def test_minimal_tools_rejects_empty(monkeypatch, tools):
    def unexpected(*args):
        pytest.fail("Empty input must not delegate")

    monkeypatch.setattr(chappel_tool_layout, "place_tools_in_chappel_organizer", unexpected)
    with pytest.raises(ValueError, match="tools must be nonempty"):
        minimal_tools(tools, 2, 3, 1)


@pytest.mark.parametrize("tool,gap,inset", [
    (rectangle_tool(5, 5), -1, 0),
    (rectangle_tool(5, 5), True, 0),
    (rectangle_tool(5, 5), float("nan"), 0),
    (rectangle_tool(5, 5), 1, -1),
    (rectangle_tool(5, 5), 1, float("inf")),
    (make_tool([]), 1, 0),
])
def test_minimal_tools_delegated_validation(tool, gap, inset):
    tools = [tool]
    with pytest.raises(ValueError) as expected:
        chappel_tool_layout.place_tools_in_chappel_organizer(tools, 1, 1, gap, inset)
    with pytest.raises(type(expected.value)) as raised:
        minimal_tools(tools, 2, 3, gap, inset)
    assert str(raised.value) == str(expected.value)


def test_minimal_tools_preserves_exception_identity(monkeypatch):
    error = ValueError("delegated validation failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(chappel_tool_layout, "place_tools_in_chappel_organizer", fail)
    with pytest.raises(ValueError) as raised:
        minimal_tools([rectangle_tool(5, 5)], 2, 3, 1)
    assert raised.value is error
