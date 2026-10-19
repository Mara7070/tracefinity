"""Numerical and validation contracts for the non-CAD Chappel vocabulary."""

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from app.services.chappel_dimensions import CHAPPEL, ChappelDimensions


@pytest.mark.parametrize("cells, ordinary, organizer", [
    (1, 54.8, 54.6),
    (2, 109.8, 109.6),
    (3, 164.8, 164.6),
    (4, 219.8, 219.6),
])
def test_exact_footprints(cells, ordinary, organizer):
    for result, expected in (
        (CHAPPEL.bin_footprint(cells), ordinary),
        (CHAPPEL.organizer_footprint(cells), organizer),
    ):
        assert result == expected
        assert type(result) is float


@pytest.mark.parametrize("cells_x, cells_y, ordinary, organizer", [
    (1, 1, (54.8, 54.8), (54.6, 54.6)),
    (2, 3, (109.8, 164.8), (109.6, 164.6)),
    (4, 1, (219.8, 54.8), (219.6, 54.6)),
    (1, 4, (54.8, 219.8), (54.6, 219.6)),
])
def test_xy_footprints(cells_x, cells_y, ordinary, organizer):
    for result, expected in (
        (CHAPPEL.bin_footprint_xy(cells_x, cells_y), ordinary),
        (CHAPPEL.organizer_footprint_xy(cells_x, cells_y), organizer),
    ):
        assert type(result) is tuple
        assert result == expected
        assert all(type(value) is float for value in result)


def test_constants_and_singleton():
    assert isinstance(CHAPPEL, ChappelDimensions)
    assert CHAPPEL == ChappelDimensions()
    assert CHAPPEL.pitch == 55.0
    assert CHAPPEL.bin_footprint_clearance == 0.2
    assert CHAPPEL.organizer_footprint_clearance == 0.4
    assert CHAPPEL.organizer_height == 25.0
    assert CHAPPEL.short_height == 39.15
    assert CHAPPEL.full_height == 75.0
    assert CHAPPEL.preferred_interface_variant == "short40"


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0.0, 0, -1, -4, "1", None])
@pytest.mark.parametrize("method", ["bin_footprint", "organizer_footprint"])
def test_invalid_single_axis(method, invalid):
    with pytest.raises(ValueError):
        getattr(CHAPPEL, method)(invalid)


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0.0, 0, -1, -4, "1", None])
@pytest.mark.parametrize("method", ["bin_footprint_xy", "organizer_footprint_xy"])
@pytest.mark.parametrize("axis", [0, 1])
def test_invalid_xy_axis(method, invalid, axis):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError):
        getattr(CHAPPEL, method)(*cells)


def test_dataclass_is_frozen():
    assert is_dataclass(ChappelDimensions)
    for dimensions in (CHAPPEL, ChappelDimensions()):
        for field in fields(dimensions):
            with pytest.raises(FrozenInstanceError):
                setattr(dimensions, field.name, getattr(dimensions, field.name))
        with pytest.raises(FrozenInstanceError):
            del dimensions.pitch
