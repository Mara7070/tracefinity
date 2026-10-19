"""Integration between persisted Tool models and pure layout primitives."""

from copy import deepcopy

import pytest

from app.models.schemas import FingerHole, Point, Tool
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates
from app.services.object_placement import nonconflicting_fit_candidates
from app.services.tool_layout import (
    canonical_outline_for_tool,
    fit_tool_candidates,
    nonconflicting_tool_candidates,
)


@pytest.fixture
def tool():
    return Tool(
        id="triangle",
        name="Traced triangle",
        points=[Point(x=-2, y=-1), Point(x=4, y=-1), Point(x=-1, y=3)],
    )


def test_canonicalizes_persisted_ring_without_mutation():
    points = [Point(x=x, y=y) for x, y in ((4, 3), (4, -1), (-2, -1), (-2, 3), (4, 3))]
    tool = Tool(id="rectangle", name="Rectangle", points=points)
    before = tool.model_dump()
    original_list = tool.points
    original_points = tuple(tool.points)

    outline = canonical_outline_for_tool(tool)

    assert outline.points == ((-2, -1), (4, -1), (4, 3), (-2, 3))
    assert outline.signed_area == 24
    assert tool.model_dump() == before
    assert tool.points is original_list
    assert all(current is original for current, original in zip(tool.points, original_points))
    assert [(point.x, point.y) for point in points] == [(4, 3), (4, -1), (-2, -1), (-2, 3), (4, 3)]


@pytest.mark.parametrize("coordinates", [[], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)]])
def test_invalid_outer_geometry_propagates_validation(coordinates):
    tool = Tool(id="invalid", name="Invalid", points=[Point(x=x, y=y) for x, y in coordinates])
    with pytest.raises(ValueError) as expected:
        CanonicalOutline.from_points(coordinates)
    with pytest.raises(ValueError) as actual:
        canonical_outline_for_tool(tool)
    assert str(actual.value) == str(expected.value)


def test_ignores_holes_and_other_tool_fields(tool):
    expected = canonical_outline_for_tool(tool)
    tool.interior_rings = [[Point(x=0, y=0), Point(x=1, y=0), Point(x=0, y=1)]]
    tool.finger_holes = [FingerHole(id="hole", x=1, y=1, radius=0.5)]
    tool.source_image_transform = [2, 0, 0, 2, 100, 200]
    tool.smoothed = False
    tool.smooth_level = 0.9
    tool.project_ids = ["project"]
    tool.tags = ["tag"]
    assert canonical_outline_for_tool(tool) == expected


@pytest.mark.parametrize(
    "width,height,inset,turns",
    [(10, 10, 0, (0, 1, 2, 3)), (7, 5, 0.5, (0, 2)), (5, 7, 0.5, (1, 3)),
     (6, 4, 0.25, ()), (3, 3, 0, ()), (10, 10, 5, ())],
)
def test_fit_matches_primitive_points_order_and_inset(tool, width, height, inset, turns):
    before = tool.model_dump()
    expected = fit_candidates(canonical_outline_for_tool(tool), width, height, inset)
    actual = fit_tool_candidates(tool, width, height, inset)
    assert isinstance(actual, tuple)
    assert actual == expected
    assert tuple(candidate.points for candidate in actual) == tuple(candidate.points for candidate in expected)
    assert tuple(candidate.quarter_turns for candidate in actual) == turns
    assert all(candidate.bounds[:2] == (inset, inset) for candidate in actual)
    assert tool.model_dump() == before


def test_default_inset(tool):
    assert fit_tool_candidates(tool, 10, 10) == fit_candidates(canonical_outline_for_tool(tool), 10, 10)


@pytest.mark.parametrize(
    "index,value",
    [(index, value) for index in range(3)
     for value in [float("nan"), float("inf"), -float("inf"), -1]] + [(0, 0), (1, 0)],
)
def test_invalid_dimensions_and_inset_delegate_validation(tool, index, value):
    args = [10, 10, 0]
    args[index] = value
    with pytest.raises(ValueError) as expected:
        fit_candidates(canonical_outline_for_tool(tool), *args)
    with pytest.raises(ValueError) as actual:
        fit_tool_candidates(tool, *args)
    assert type(actual.value) is type(expected.value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("occupied", [(), []])
@pytest.mark.parametrize("width,height,inset", [(10, 10, 0), (7, 5, 0.5), (5, 7, 0.5), (3, 3, 0)])
def test_nonconflicting_empty_matches_fit(tool, occupied, width, height, inset):
    assert nonconflicting_tool_candidates(tool, width, height, occupied, inset) == fit_tool_candidates(
        tool, width, height, inset
    )


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("x,y,w,h,width,height,turns", [
    (20, 20, 1, 1, 6, 6, (0, 1, 2, 3)),
    (0.1, 0.1, 0.1, 0.1, 6, 4, (2,)),
    (5, 0, 0.25, 4, 6, 6, (1, 3)),
    (0, 0, 6, 6, 6, 6, ()),
])
def test_nonconflicting_filters_in_order(container, x, y, w, h, width, height, turns):
    tool = Tool(id="triangle", name="Triangle", points=[
        Point(x=0, y=0), Point(x=6, y=0), Point(x=0, y=4),
    ])
    outline = CanonicalOutline.from_points(((0, 0), (w, 0), (w, h), (0, h)))
    occupied = container([PlacedOutline(outline, x, y)])
    actual = nonconflicting_tool_candidates(tool, width, height, occupied)
    assert isinstance(actual, tuple)
    assert tuple(p.quarter_turns for p in actual) == turns
    assert actual == nonconflicting_fit_candidates(
        canonical_outline_for_tool(tool), width, height, occupied
    )
    assert actual == nonconflicting_tool_candidates(tool, width, height, occupied)


def test_nonconflicting_ignores_non_outer_fields(tool):
    expected = nonconflicting_tool_candidates(tool, 10, 10, [], 0.5)
    tool.interior_rings = [[Point(x=100, y=100)]]
    tool.finger_holes = [FingerHole(id="hole", x=100, y=100, radius=50)]
    tool.source_image_transform = [2, 0, 0, 2, 100, 200]
    tool.smoothed = False
    tool.smooth_level = 0.9
    tool.project_ids = ["project"]
    tool.tags = ["tag"]
    tool.name = "Renamed"
    assert nonconflicting_tool_candidates(tool, 10, 10, [], 0.5) == expected


@pytest.mark.parametrize("coordinates", [
    [], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
])
def test_nonconflicting_invalid_tool_propagates(coordinates):
    tool = Tool(id="invalid", name="Invalid", points=[Point(x=x, y=y) for x, y in coordinates])
    with pytest.raises(ValueError) as expected:
        canonical_outline_for_tool(tool)
    with pytest.raises(ValueError) as actual:
        nonconflicting_tool_candidates(tool, 10, 10, [])
    assert type(actual.value) is type(expected.value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("invalid_tool", [False, True])
@pytest.mark.parametrize("container", [tuple, list])
def test_nonconflicting_invalid_collision_geometry(tool, invalid_tool, container):
    crossing = [(0, 0), (4, 4), (0, 4), (3, 0)]
    occupied_outline = CanonicalOutline.from_points(crossing)
    if invalid_tool:
        occupied_outline = canonical_outline_for_tool(tool)
        tool.points = [Point(x=x, y=y) for x, y in crossing]
    occupied = container([PlacedOutline(occupied_outline, 20, 20)])
    with pytest.raises(ValueError) as expected:
        nonconflicting_fit_candidates(canonical_outline_for_tool(tool), 10, 10, occupied)
    with pytest.raises(ValueError) as actual:
        nonconflicting_tool_candidates(tool, 10, 10, occupied)
    assert type(actual.value) is type(expected.value)
    assert str(actual.value) == str(expected.value)
    assert nonconflicting_tool_candidates(tool, 1, 1, occupied) == ()


@pytest.mark.parametrize("index,value", [
    (index, value) for index in range(3)
    for value in [float("nan"), float("inf"), -float("inf"), -1]
] + [(0, 0), (1, 0)])
def test_nonconflicting_validation_delegates(tool, index, value):
    args = [10, 10, 0]
    args[index] = value
    with pytest.raises(ValueError) as expected:
        nonconflicting_fit_candidates(canonical_outline_for_tool(tool), args[0], args[1], [], args[2])
    with pytest.raises(ValueError) as actual:
        nonconflicting_tool_candidates(tool, args[0], args[1], [], args[2])
    assert type(actual.value) is type(expected.value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("container", [tuple, list])
def test_nonconflicting_preserves_inputs(tool, container):
    occupied = container([PlacedOutline(canonical_outline_for_tool(tool), 20, 20, 1)])
    before = deepcopy((tool.model_dump(), occupied, tuple(p.points for p in occupied)))
    original_points = tuple(tool.points)
    original_list = tool.points
    original_placements = tuple(occupied)
    assert nonconflicting_tool_candidates(tool, 10, 10, occupied, 0.5)
    assert (tool.model_dump(), occupied, tuple(p.points for p in occupied)) == before
    assert tool.points is original_list
    assert all(a is b for a, b in zip(tool.points, original_points))
    assert all(a is b for a, b in zip(occupied, original_placements))
