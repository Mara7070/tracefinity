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


@pytest.mark.parametrize("name, expected", [
    ("grid_height", 4.0),
    ("foot_bottom_size", 45.4909),
    ("foot_chamfer", 2.9698),
    ("foot_floor_thickness", 1.45),
    ("foot_inner_at_floor", 44.2897),
    ("foot_inner_chamfer", 1.9445),
    ("foot_transition_z", 3.75),
    ("foot_outer_at_transition", 52.9905),
    ("foot_outer_at_grid_top", 53.3511),
    ("v2_foot_size_delta", 0.6258),
])
def test_recovered_foot_constants(name, expected):
    for dimensions in (CHAPPEL, ChappelDimensions()):
        assert getattr(dimensions, name) == expected
        assert type(getattr(dimensions, name)) is float
        with pytest.raises(FrozenInstanceError):
            setattr(dimensions, name, 0)


@pytest.mark.parametrize("name, expected", [
    ("insert_bottom_inset", 8.0),
    ("insert_mid_inset", 6.0),
    ("insert_bottom_ramp_z", 1.0),
    ("insert_mid_top_z", 10.0),
    ("insert_shoulder_top_z", 13.0),
    ("insert_mid_corner_radius", 1.0),
    ("insert_top_corner_radius", 3.0),
])
def test_recovered_insert_constants(name, expected):
    for dimensions in (CHAPPEL, ChappelDimensions()):
        assert getattr(dimensions, name) == expected
        assert type(getattr(dimensions, name)) is float
        with pytest.raises(FrozenInstanceError):
            setattr(dimensions, name, 0)
