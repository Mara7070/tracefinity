"""Integration between persisted Tool models and pure layout primitives."""

import pytest

from app.models.schemas import FingerHole, Point, Tool
from app.services.object_geometry import CanonicalOutline
from app.services.object_layout import fit_candidates
from app.services.tool_layout import canonical_outline_for_tool, fit_tool_candidates


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
