"""Physical and topology contracts for native Chappel blanks and mating feet."""

import pytest
from build123d import Compound, GeomType, Location, Solid
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.TopoDS import TopoDS_Solid

from app.services.chappel_brep import (
    chappel_adapter_blank,
    chappel_insert_blank,
    chappel_insert_with_pocket,
    chappel_insert_with_pockets,
    chappel_organizer_blank,
    chappel_v2_foot_array,
    chappel_v2_foot_cell,
)
from app.services.chappel_dimensions import CHAPPEL
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.polygon_scaler import PolygonScaler, ScaledPolygon


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


@pytest.mark.parametrize("factory", [chappel_organizer_blank, chappel_insert_blank, chappel_adapter_blank])
def test_repeated_calls_are_independent(factory):
    first = factory(2, 3)
    second = factory(2, 3)
    assert first is not second
    assert not first.wrapped.IsSame(second.wrapped)
    first.move(Location((7, 8, 9)))
    tolerance = 1e-9 if factory is chappel_organizer_blank else 5e-7
    assert tuple(first.bounding_box().min) == pytest.approx((7, 8, 9), rel=0, abs=tolerance)
    assert tuple(second.bounding_box().min) == pytest.approx((0, 0, 0), rel=0, abs=tolerance)
    assert second.is_valid


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0.0, 0, -1, -4, "1", None])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("factory", [chappel_organizer_blank, chappel_insert_blank, chappel_adapter_blank])
def test_cell_validation_propagates(invalid, axis, factory):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError) as dimensions_error:
        CHAPPEL.organizer_footprint_xy(*cells)
    with pytest.raises(ValueError) as blank_error:
        factory(*cells)
    assert type(blank_error.value) is type(dimensions_error.value)
    assert blank_error.value.args == dimensions_error.value.args


@pytest.mark.parametrize("nx, ny", [(1, 1), (2, 1), (2, 3)])
@pytest.mark.parametrize("thickness", [None, 0.75, 3])
def test_native_adapter_blank(nx, ny, thickness):
    blank = (chappel_adapter_blank(nx, ny) if thickness is None
             else chappel_adapter_blank(nx, ny, thickness))
    thickness = 1.5 if thickness is None else thickness
    width, depth = CHAPPEL.bin_footprint_xy(nx, ny)
    assert isinstance(blank, Solid)
    assert isinstance(blank.wrapped, TopoDS_Solid)
    assert len(blank.solids()) == 1
    assert blank.is_valid
    assert BRepCheck_Analyzer(blank.wrapped).IsValid()
    assert tuple(blank.bounding_box().min) == pytest.approx((0, 0, 0), rel=0, abs=5e-7)
    assert tuple(blank.bounding_box().max) == pytest.approx(
        (width, depth, 4 + thickness), rel=0, abs=5e-7,
    )
    # Actual horizontal sections prove the deck covers the ordinary footprint,
    # including the overhang at Z=4, with no interior holes above the feet.
    assert (width, depth) == pytest.approx((55 * nx - 0.2, 55 * ny - 0.2))
    for z in (4, 4 + thickness / 2, 4 + thickness):
        operation = BRepAlgoAPI_Section(
            blank.wrapped, gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), False,
        )
        operation.Build()
        assert operation.IsDone()
        section = Compound(operation.Shape())
        assert tuple(section.bounding_box().min) == pytest.approx((0, 0, z), rel=0, abs=5e-7)
        assert tuple(section.bounding_box().max) == pytest.approx((width, depth, z), rel=0, abs=5e-7)
        if z > 4:
            assert len(section.edges()) == 4
    assert not blank.is_inside((0.1, 0.1, 3.999))
    assert blank.is_inside((0.1, 0.1, 4.001))

    # A slab below the foot cavity floor must contain one disconnected foot
    # per cell; centres, bounds and empty inter-cell gaps come from topology.
    slab = Solid.make_box(width, depth, 0.5).moved(Location((0, 0, 0.5)))
    lower = blank.intersect(slab)
    assert len(lower.solids()) == nx * ny
    centers = sorted(tuple(s.bounding_box().center()) for s in lower.solids())
    expected = sorted((27.3 + 55 * x, 27.3 + 55 * y, 0.75)
                      for x in range(nx) for y in range(ny))
    for actual, wanted in zip(centers, expected, strict=True):
        assert actual == pytest.approx(wanted, rel=0, abs=5e-7)
    bounds = Compound(lower.solids()).bounding_box()
    assert bounds.min.X > 0 and bounds.min.Y > 0
    assert bounds.max.X < width and bounds.max.Y < depth
    for x in range(nx):
        for y in range(ny):
            cx, cy = 27.3 + 55 * x, 27.3 + 55 * y
            assert blank.is_inside((cx, cy, 1))
            assert not blank.is_inside((cx, cy, 3.999))
            assert blank.is_inside((cx + 26, cy, 3.999))
            assert blank.is_inside((cx, cy, 4.001))
            if x + 1 < nx:
                assert not blank.is_inside((cx + 27.5, cy, 1))
            if y + 1 < ny:
                assert not blank.is_inside((cx, cy + 27.5, 1))
    assert blank.volume == pytest.approx(
        chappel_v2_foot_array(nx, ny).volume + width * depth * thickness,
        rel=0, abs=1e-6,
    )


@pytest.mark.parametrize("thickness", [
    True, False, 0, -1, float("nan"), float("inf"), -float("inf"), "1.5", None, 1j,
])
def test_adapter_invalid_deck_thickness(thickness):
    with pytest.raises(ValueError, match="^deck_thickness must be a finite positive real number$"):
        chappel_adapter_blank(1, 1, thickness)


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


@pytest.mark.parametrize("nx, ny, width, depth", [
    (1, 1, 54.6, 54.6), (2, 3, 109.6, 164.6), (3, 2, 164.6, 109.6),
])
def test_native_insert_sections(nx, ny, width, depth):
    blank = chappel_insert_blank(nx, ny)
    assert isinstance(blank, Solid)
    assert isinstance(blank.wrapped, TopoDS_Solid)
    assert len(blank.solids()) == 1
    assert BRepCheck_Analyzer(blank.wrapped).IsValid()
    bounds = blank.bounding_box()
    assert tuple(bounds.min) == pytest.approx((0, 0, 0), rel=0, abs=5e-7)
    assert tuple(bounds.max) == pytest.approx((width, depth, 25), rel=0, abs=5e-7)
    if nx != ny:
        assert (bounds.size.X < bounds.size.Y) == (nx < ny)
    assert 0 < blank.volume < width * depth * 25
    assert len(blank.shells()) == 1
    # Intersect the actual B-rep, including caps, kinks and intermediate ramps.
    for z, inset in [
        (0, 8), (0.0001, 7.9998), (0.5, 7), (0.9999, 6.0002),
        (1, 6), (1.0001, 6), (5, 6), (9.9999, 6), (10, 6),
        (10.0001, 5.9998), (11.5, 3), (12.9999, 0.0002),
        (13, 0), (13.0001, 0), (24.9999, 0), (25, 0),
    ]:
        operation = BRepAlgoAPI_Section(
            blank.wrapped, gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), False,
        )
        operation.Build()
        assert operation.IsDone()
        section = Compound(operation.Shape())
        assert section.edges()
        section_bounds = section.bounding_box()
        assert tuple(section_bounds.min) == pytest.approx(
            (inset / 2, inset / 2, z), rel=0, abs=5e-7,
        )
        assert tuple(section_bounds.max) == pytest.approx(
            (width - inset / 2, depth - inset / 2, z), rel=0, abs=5e-7,
        )
        # No internal cavity: each interior slice contains the payload centre.
        assert blank.is_inside((width / 2, depth / 2, z))
    # Circular corner radii are measured from section curves, not input wires.
    for z, radius in [(0, 0), (1, 1), (5, 1), (10, 1), (13, 3), (25, 3)]:
        operation = BRepAlgoAPI_Section(
            blank.wrapped, gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), False,
        )
        operation.Build()
        section = Compound(operation.Shape())
        arcs = [edge for edge in section.edges() if edge.geom_type == GeomType.CIRCLE]
        assert len(arcs) == (4 if radius else 0)
        for arc in arcs:
            assert arc.radius == pytest.approx(radius, rel=0, abs=5e-7)


@pytest.fixture
def pocket_placement():
    return PlacedOutline(CanonicalOutline.from_points(
        ((0, 0), (20, 0), (20, 10), (0, 10)),
    ), x=30, y=40)


@pytest.mark.parametrize("depth", [5.0, 12.0, 23.0])
def test_native_pocket(pocket_placement, depth):
    blank = chappel_insert_blank(2, 3)
    result = chappel_insert_with_pocket(pocket_placement, 2, 3, depth)
    assert isinstance(result, Solid)
    assert isinstance(result.wrapped, TopoDS_Solid)
    assert len(result.solids()) == 1
    assert BRepCheck_Analyzer(result.wrapped).IsValid()
    for attr in ("min", "max"):
        assert tuple(getattr(result.bounding_box(), attr)) == pytest.approx(
            tuple(getattr(blank.bounding_box(), attr)), rel=0, abs=5e-7,
        )
    assert blank.volume - result.volume == pytest.approx(200 * depth, rel=0, abs=1e-6)
    floor = 25 - depth
    assert not result.is_inside((40, 45, 24.99))
    assert not result.is_inside((40, 45, floor + 0.001))
    assert result.is_inside((40, 45, floor - 0.001))
    assert result.is_inside((29.99, 45, 24.99))


def test_pocket_uses_transformed_placement(pocket_placement):
    placement = PlacedOutline(pocket_placement.outline, x=70, y=60, quarter_turns=1)
    result = chappel_insert_with_pocket(placement, 2, 3, 5)
    assert not result.is_inside((65, 70, 24))
    assert result.is_inside((75, 65, 24))
    assert result.is_inside((40, 45, 24))
    assert result.is_inside((65, 70, 19.99))


def test_pocket_clearance_matches_existing_buffer(pocket_placement):
    expected = PolygonScaler().add_clearance(
        ScaledPolygon("expected", list(pocket_placement.points), ""), 1.0,
    )
    result = chappel_insert_with_pocket(pocket_placement, 2, 3, 5, 1.0)
    # At a height inside the pocket, the section contains an outer envelope and
    # the cavity wire. Measure the latter from the actual subtracted geometry.
    operation = BRepAlgoAPI_Section(
        result.wrapped, gp_Pln(gp_Pnt(0, 0, 23), gp_Dir(0, 0, 1)), False,
    )
    operation.Build()
    edges = Compound(operation.Shape()).edges()
    cavity_edges = [e for e in edges if 20 < e.center().X < 60 and 30 < e.center().Y < 60]
    bounds = Compound(cavity_edges).bounding_box()
    xs, ys = zip(*expected.points_mm)
    assert tuple(bounds.min) == pytest.approx((min(xs), min(ys), 23), abs=5e-7)
    assert tuple(bounds.max) == pytest.approx((max(xs), max(ys), 23), abs=5e-7)
    assert chappel_insert_blank(2, 3).volume - result.volume == pytest.approx(
        22 * 12 * 5, rel=0, abs=1e-6,
    )


def test_containment_is_depth_aware(pocket_placement):
    placement = PlacedOutline(pocket_placement.outline, x=1, y=40)
    assert chappel_insert_with_pocket(placement, 2, 3, 12).is_valid
    with pytest.raises(ValueError, match="^Pocket is not fully contained within the Chappel insert$"):
        chappel_insert_with_pocket(placement, 2, 3, 20)


def test_clearance_can_breach_envelope(pocket_placement):
    placement = PlacedOutline(pocket_placement.outline, x=0.5, y=40)
    assert chappel_insert_with_pocket(placement, 2, 3, 5).is_valid
    with pytest.raises(ValueError, match="Pocket is not fully contained"):
        chappel_insert_with_pocket(placement, 2, 3, 5, 1.0)


def test_rounded_corner_breach(pocket_placement):
    placement = PlacedOutline(pocket_placement.outline, x=0.1, y=0.1)
    with pytest.raises(ValueError, match="Pocket is not fully contained"):
        chappel_insert_with_pocket(placement, 2, 3, 5)


def test_pocket_preserves_concave_outline_detail():
    outline = CanonicalOutline.from_points((
        (0, 0), (20, 0), (20, 10), (10.1, 10),
        (10.1, 9.8), (9.9, 9.8), (9.9, 10), (0, 10),
    ))
    placement = PlacedOutline(outline, x=30, y=40)
    result = chappel_insert_with_pocket(placement, 2, 3, 5)
    assert result.is_inside((40, 49.9, 24))
    assert not result.is_inside((40, 49.7, 24))
    assert chappel_insert_blank(2, 3).volume - result.volume == pytest.approx(
        outline.signed_area * 5, rel=0, abs=1e-6,
    )


@pytest.mark.parametrize("depth", [True, False, None, "5", 1j, float("nan"), float("inf"), -float("inf"), 0, -1, 23.001])
@pytest.mark.parametrize("factory", [chappel_insert_with_pocket, lambda p, *a: chappel_insert_with_pockets([p], *a)])
def test_invalid_pocket_depth(pocket_placement, depth, factory):
    with pytest.raises(ValueError, match="depth must"):
        factory(pocket_placement, 2, 3, depth)


@pytest.mark.parametrize("clearance", [True, False, None, "1", 1j, float("nan"), float("inf"), -float("inf"), -0.001])
@pytest.mark.parametrize("factory", [chappel_insert_with_pocket, lambda p, *a: chappel_insert_with_pockets([p], *a)])
def test_invalid_pocket_clearance(pocket_placement, clearance, factory):
    with pytest.raises(ValueError, match="clearance must"):
        factory(pocket_placement, 2, 3, 5, clearance)


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.5, 0, -1, "1", None])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("factory", [chappel_insert_with_pocket, lambda p, *a: chappel_insert_with_pockets([p], *a)])
def test_pocket_cell_validation(pocket_placement, invalid, axis, factory):
    cells = [2, 3]
    cells[axis] = invalid
    with pytest.raises(ValueError) as expected:
        CHAPPEL.organizer_footprint_xy(*cells)
    with pytest.raises(ValueError) as actual:
        factory(pocket_placement, *cells, 5)
    assert actual.value.args == expected.value.args


def test_pocket_calls_are_independent_and_preserve_input(pocket_placement):
    original = (pocket_placement.outline.points, pocket_placement.points, pocket_placement.bounds)
    first = chappel_insert_with_pocket(pocket_placement, 2, 3, 5, 1)
    second = chappel_insert_with_pocket(pocket_placement, 2, 3, 5, 1)
    assert not first.wrapped.IsSame(second.wrapped)
    first.move(Location((7, 8, 9)))
    assert tuple(second.bounding_box().min) == pytest.approx((0, 0, 0), abs=5e-7)
    assert (pocket_placement.outline.points, pocket_placement.points, pocket_placement.bounds) == original


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("clearance", [0, 1])
def test_composed_pockets(pocket_placement, count, container, clearance):
    placements = container([
        pocket_placement,
        PlacedOutline(pocket_placement.outline, x=80, y=60, quarter_turns=1),
        PlacedOutline(pocket_placement.outline, x=50, y=110, quarter_turns=3),
    ][:count])
    original = tuple((p, p.points, p.bounds) for p in placements)
    blank = chappel_insert_blank(2, 3)
    result = chappel_insert_with_pockets(placements, 2, 3, 23, clearance)
    reverse = chappel_insert_with_pockets(tuple(reversed(placements)), 2, 3, 23, clearance)
    assert isinstance(result.wrapped, TopoDS_Solid)
    assert len(result.solids()) == 1
    assert BRepCheck_Analyzer(result.wrapped).IsValid()
    assert BRepCheck_Analyzer(reverse.wrapped).IsValid()
    for attr in ("min", "max"):
        assert tuple(getattr(result.bounding_box(), attr)) == pytest.approx(
            tuple(getattr(blank.bounding_box(), attr)), abs=5e-7,
        )
    assert blank.volume - result.volume == pytest.approx(
        count * (20 + 2 * clearance) * (10 + 2 * clearance) * 23, abs=1e-6,
    )
    assert reverse.volume == pytest.approx(result.volume, abs=1e-6)
    assert result.cut(reverse).volume == pytest.approx(0, abs=1e-7)
    assert reverse.cut(result).volume == pytest.approx(0, abs=1e-7)
    for placement in placements:
        single = chappel_insert_with_pocket(placement, 2, 3, 23, clearance)
        min_x, min_y, max_x, max_y = placement.bounds
        x, y = (min_x + max_x) / 2, (min_y + max_y) / 2
        assert not result.is_inside((x, y, 2.001))
        assert not result.is_inside((x, y, 24.999))
        assert result.is_inside((x, y, 1.999))
        assert result.is_inside((min_x - clearance - 0.01, y, 24))
        if count == 1:
            assert single.volume == pytest.approx(result.volume, abs=1e-6)
            assert single.cut(result).volume == pytest.approx(0, abs=1e-7)
            assert result.cut(single).volume == pytest.approx(0, abs=1e-7)
    assert tuple((p, p.points, p.bounds) for p in placements) == original


@pytest.mark.parametrize("offset, clearance, message", [
    (19, 0, "^Pocket placements"), (20, 0, "^Pocket placements"),
    (21, 1, "^Clearance-expanded pockets"), (22, 1, "^Clearance-expanded pockets"),
])
def test_composed_collision(pocket_placement, offset, clearance, message):
    other = PlacedOutline(pocket_placement.outline, x=30 + offset, y=40)
    with pytest.raises(ValueError, match=message):
        chappel_insert_with_pockets([pocket_placement, other], 2, 3, 5, clearance)


@pytest.mark.parametrize("reverse", [False, True])
def test_composed_deep_breach(pocket_placement, reverse):
    edge = PlacedOutline(pocket_placement.outline, x=1, y=80)
    placements = [pocket_placement, edge]
    if reverse:
        placements.reverse()
    assert chappel_insert_with_pockets(placements, 2, 3, 12).is_valid
    with pytest.raises(ValueError, match="^Pocket is not fully contained within the Chappel insert$"):
        chappel_insert_with_pockets(placements, 2, 3, 20)


@pytest.mark.parametrize("placements", [[], ()])
def test_composed_empty(placements):
    with pytest.raises(ValueError, match="^At least one pocket placement is required$"):
        chappel_insert_with_pockets(placements, 2, 3, 5)


def test_composition_buffers_each_global_ring_once(pocket_placement, monkeypatch):
    placements = [pocket_placement, PlacedOutline(
        pocket_placement.outline, x=80, y=60, quarter_turns=1,
    )]
    calls = []
    original = PolygonScaler.add_clearance

    def record(self, polygon, clearance):
        calls.append((tuple(polygon.points_mm), clearance))
        return original(self, polygon, clearance)

    monkeypatch.setattr(PolygonScaler, "add_clearance", record)
    chappel_insert_with_pockets(placements, 2, 3, 5, 1)
    assert calls == [(p.points, 1) for p in placements]
