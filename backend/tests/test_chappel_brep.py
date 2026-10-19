"""Physical and topology contracts for the native organizer outer blank."""

import pytest
from build123d import Location, Solid
from OCP.TopoDS import TopoDS_Solid

from app.services.chappel_brep import chappel_organizer_blank
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
