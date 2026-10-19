"""Physical and topology contracts for native Chappel blanks and mating feet."""

import pytest
from build123d import Compound, Location, Solid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS_Solid

from app.services.chappel_brep import (
    chappel_organizer_blank,
    chappel_v2_foot_array,
    chappel_v2_foot_cell,
)
from app.services.chappel_dimensions import CHAPPEL


@pytest.mark.parametrize("cells_x, cells_y, expected_xy", [
    (1, 1, (54.6, 54.6)),
    (2, 1, (109.6, 54.6)),
    (1, 2, (54.6, 109.6)),
    (2, 3, (109.6, 164.6)),
    (3, 2, (164.6, 109.6)),
])
def test_native_blank(cells_x, cells_y, expected_xy):
    blank = chappel_organizer_blank(cells_x, cells_y)

    assert isinstance(blank, Solid)
    assert isinstance(blank.wrapped, TopoDS_Solid)
    assert len(blank.solids()) == 1
    assert blank.is_valid  # build123d exposes OCCT validity as a property.
    bounds = blank.bounding_box()
    assert tuple(bounds.min) == pytest.approx((0, 0, 0), rel=0, abs=1e-9)
    expected = (*expected_xy, 25.0)
    assert tuple(bounds.max) == pytest.approx(expected, rel=0, abs=1e-9)
    assert tuple(bounds.size) == pytest.approx(
        (*CHAPPEL.organizer_footprint_xy(cells_x, cells_y), CHAPPEL.organizer_height),
        rel=0, abs=1e-9,
    )
    assert blank.volume == pytest.approx(
        expected[0] * expected[1] * expected[2], rel=0, abs=1e-7,
    )


def test_asymmetric_axes():
    xy = chappel_organizer_blank(2, 3).bounding_box().size
    yx = chappel_organizer_blank(3, 2).bounding_box().size
    assert xy.X < xy.Y
    assert yx.X > yx.Y
    assert (xy.X, xy.Y, xy.Z) == pytest.approx(
        (yx.Y, yx.X, yx.Z), rel=0, abs=1e-9,
    )


def test_repeated_calls_are_independent():
    first = chappel_organizer_blank(2, 3)
    second = chappel_organizer_blank(2, 3)
    assert first is not second
    assert not first.wrapped.IsSame(second.wrapped)
    first.move(Location((7, 8, 9)))
    assert tuple(first.bounding_box().min) == pytest.approx((7, 8, 9), abs=1e-9)
    assert tuple(second.bounding_box().min) == pytest.approx((0, 0, 0), abs=1e-9)
    assert second.is_valid


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0.0, 0, -1, -4, "1", None])
@pytest.mark.parametrize("axis", [0, 1])
def test_cell_validation_propagates(invalid, axis):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError) as dimensions_error:
        CHAPPEL.organizer_footprint_xy(*cells)
    with pytest.raises(ValueError) as blank_error:
        chappel_organizer_blank(*cells)
    assert type(blank_error.value) is type(dimensions_error.value)
    assert blank_error.value.args == dimensions_error.value.args


def test_native_v2_foot_cell():
    cell = chappel_v2_foot_cell()
    assert isinstance(cell, Solid)
    assert isinstance(cell.wrapped, TopoDS_Solid)
    assert len(cell.solids()) == 1
    assert BRepCheck_Analyzer(cell.wrapped).IsValid()
    bounds = cell.bounding_box()
    assert tuple(bounds.min) == pytest.approx((-26.98845, -26.98845, 0), rel=0, abs=5e-7)
    assert tuple(bounds.max) == pytest.approx((26.98845, 26.98845, 4), rel=0, abs=5e-7)
    assert tuple(bounds.size) == pytest.approx((53.9769, 53.9769, 4), rel=0, abs=5e-7)
    assert 0 < cell.volume < 53.9769 ** 2 * 4
    # A floor remains beneath an open cavity, rather than a full prism.
    assert cell.is_inside((0, 0, 1.44))
    assert not cell.is_inside((0, 0, 1.46))
    assert not cell.is_inside((0, 0, 3.99))
    assert cell.is_inside((26, 0, 3.99))
    # Recovered outer section sizes, including the transition kink.
    for z, half_size in [(0, 46.1167 / 2), (3.75, 53.6163 / 2), (4, 53.9769 / 2)]:
        vertices = [v for v in cell.vertices() if abs(v.Z - z) < 1e-7]
        assert max(v.X for v in vertices) == pytest.approx(half_size, rel=0, abs=5e-7)


@pytest.mark.parametrize("nx, ny", [(1, 1), (2, 3), (3, 2)])
def test_v2_array_positions_and_bounds(nx, ny):
    array = chappel_v2_foot_array(nx, ny)
    assert isinstance(array, Compound)
    solids = array.solids()
    assert len(solids) == nx * ny
    assert BRepCheck_Analyzer(array.wrapped).IsValid()
    centers = sorted(tuple(s.bounding_box().center()) for s in solids)
    expected = sorted((27.3 + 55 * x, 27.3 + 55 * y, 2) for x in range(nx) for y in range(ny))
    for actual, wanted in zip(centers, expected, strict=True):
        assert actual == pytest.approx(wanted, rel=0, abs=5e-7)
    cell_volume = chappel_v2_foot_cell().volume
    assert all(s.volume == pytest.approx(cell_volume, rel=0, abs=5e-7) for s in solids)
    margin = (54.6 - 53.9769) / 2
    width, depth = CHAPPEL.organizer_footprint_xy(nx, ny)
    bounds = array.bounding_box()
    assert tuple(bounds.min) == pytest.approx((margin, margin, 0), rel=0, abs=5e-7)
    assert tuple(bounds.max) == pytest.approx((width - margin, depth - margin, 4), rel=0, abs=5e-7)
    if nx != ny:
        assert (bounds.size.X < bounds.size.Y) == (nx < ny)


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0.0, 0, -1, -4, "1", None])
@pytest.mark.parametrize("axis", [0, 1])
def test_v2_array_validation_propagates(invalid, axis):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError) as expected:
        CHAPPEL.organizer_footprint_xy(*cells)
    with pytest.raises(ValueError) as actual:
        chappel_v2_foot_array(*cells)
    assert type(actual.value) is type(expected.value)
    assert actual.value.args == expected.value.args


@pytest.mark.parametrize("factory", [chappel_v2_foot_cell, lambda: chappel_v2_foot_array(2, 3)])
def test_v2_calls_are_independent(factory):
    first, second = factory(), factory()
    original = tuple(second.bounding_box().min)
    assert first is not second
    for a in first.solids():
        for b in second.solids():
            assert not a.wrapped.IsSame(b.wrapped)
    first.move(Location((7, 8, 9)))
    assert tuple(second.bounding_box().min) == pytest.approx(original, rel=0, abs=5e-7)
    assert tuple(first.bounding_box().min) == pytest.approx(
        tuple(a + b for a, b in zip(original, (7, 8, 9), strict=True)), rel=0, abs=5e-7,
    )
