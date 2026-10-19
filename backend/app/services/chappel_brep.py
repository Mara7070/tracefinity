"""Native Chappel B-rep blanks and recovered v2/40 mating feet."""

from math import isfinite
from numbers import Real

from build123d import Compound, Face, Location, Plane, Solid, Wire

from app.services.chappel_dimensions import CHAPPEL
from app.services.object_collision import conflicting_pairs
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.polygon_scaler import PolygonScaler, ScaledPolygon


def chappel_organizer_blank(cells_x: int, cells_y: int) -> Solid:
    """Return a rectangular solid with its lower-left-bottom corner at (0, 0, 0).

    Dimensions are in millimetres. This outer blank has no mating or interior
    features and is not a finished printable organizer. Cell validation belongs
    to CHAPPEL and propagates unchanged.
    """
    width, depth = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return Solid.make_box(width, depth, CHAPPEL.organizer_height)


def chappel_insert_blank(cells_x: int, cells_y: int) -> Solid:
    """Return the recovered solid 25-series external envelope in the XY frame.

    Sections follow Mara7070/cad-apprenticeship/interfaces/chappel_insert.py:
    centred lower sections, sharp bottom corners, and ruled transitions to the
    rounded full footprint. This payload has no donor bowl cavity or mating foot.
    CHAPPEL owns cell validation; dimensions are in millimetres.
    """
    d = CHAPPEL
    width, depth = d.organizer_footprint_xy(cells_x, cells_y)
    sections = []
    for inset, z, radius in (
        (d.insert_bottom_inset, 0, 0),
        (d.insert_mid_inset, d.insert_bottom_ramp_z, d.insert_mid_corner_radius),
        (d.insert_mid_inset, d.insert_mid_top_z, d.insert_mid_corner_radius),
        (0, d.insert_shoulder_top_z, d.insert_top_corner_radius),
        (0, d.organizer_height, d.insert_top_corner_radius),
    ):
        section = Wire.make_rect(
            width - inset, depth - inset,
            Plane(origin=(width / 2, depth / 2, z)),
        )
        if radius:
            section = section.fillet_2d(radius, section.vertices())
        sections.append(section)
    return Solid.make_loft(sections, ruled=True)


def chappel_insert_with_pocket(
    placement: PlacedOutline, cells_x: int, cells_y: int, depth: float,
    clearance: float = 0.0,
) -> Solid:
    """Cut one placed outer ring downward from the recovered insert top.

    Placement coordinates are millimetres in the blank's XY frame. Clearance
    is applied once after placement, without further smoothing or reduction.
    The physical prism must fit the native envelope over its entire depth;
    a 2 mm structural floor remains beneath the deepest permitted pocket.
    """
    return _insert_with_pockets((placement,), cells_x, cells_y, depth, clearance)


def chappel_insert_with_pockets(
    placements: tuple[PlacedOutline, ...] | list[PlacedOutline],
    cells_x: int, cells_y: int, depth: float, clearance: float = 0.0,
) -> Solid:
    """Cut explicit placements after raw/expanded collision and native containment checks."""
    if not placements:
        raise ValueError("At least one pocket placement is required")
    return _insert_with_pockets(tuple(placements), cells_x, cells_y, depth, clearance)


def _insert_with_pockets(
    placements: tuple[PlacedOutline, ...], cells_x: int, cells_y: int,
    depth: float, clearance: float,
) -> Solid:
    blank = chappel_insert_blank(cells_x, cells_y)
    top = CHAPPEL.organizer_height
    if (isinstance(depth, bool) or not isinstance(depth, Real)
            or not isfinite(depth) or not 0 < depth <= top - 2.0):
        raise ValueError("depth must be finite and satisfy 0 < depth <= 23.0 mm")
    if (isinstance(clearance, bool) or not isinstance(clearance, Real)
            or not isfinite(clearance) or clearance < 0):
        raise ValueError("clearance must be a finite nonnegative real number")

    if conflicting_pairs(placements):
        raise ValueError("Pocket placements must not overlap or touch")
    polygons = [
        PolygonScaler().add_clearance(
            ScaledPolygon("pocket", list(placement.points), ""), clearance,
        )
        for placement in placements
    ]
    expanded = tuple(
        PlacedOutline(CanonicalOutline.from_points(polygon.points_mm))
        for polygon in polygons
    )
    if conflicting_pairs(expanded):
        raise ValueError("Clearance-expanded pockets must not overlap or touch")
    # Validate every physical prism against the original blank before cutting.
    cutters = [
        _contained_pocket_cutter(polygon.points_mm, blank, depth)
        for polygon in polygons
    ]
    result = blank
    for cutter in cutters:
        result = result.cut(cutter)
    solids = result.solids()
    if len(solids) != 1 or not result.is_valid:
        raise ValueError("Pocket must produce one valid Chappel insert solid")
    return solids[0]


def _contained_pocket_cutter(
    points: list[tuple[float, float]], blank: Solid, depth: float,
) -> Solid:
    top = CHAPPEL.organizer_height
    floor = top - depth
    face = Face(Wire.make_polygon(
        [(x, y, floor) for x, y in points], close=True,
    ))
    physical = Solid.extrude(face, (0, 0, depth))
    # Measure escaped material, rather than approximating the tapered envelope
    # with XY bounds. The absolute tolerance is in cubic millimetres.
    escaped = physical.cut(blank)
    if not physical.is_valid or escaped.volume > 1e-7:
        raise ValueError("Pocket is not fully contained within the Chappel insert")

    # Only the boolean cutter overruns the top; validation used physical depth.
    return Solid.extrude(face, (0, 0, depth + 0.01))


def _chamfered_square(size: float, chamfer: float, z: float) -> Wire:
    """Recovered eight-vertex section, centred on the archaeology origin."""
    h = size / 2
    c = chamfer
    return Wire.make_polygon([
        (-h + c, -h, z), (h - c, -h, z),
        (h, -h + c, z), (h, h - c, z),
        (h - c, h, z), (-h + c, h, z),
        (-h, h - c, z), (-h, -h + c, z),
    ], close=True)


def chappel_v2_foot_cell() -> Solid:
    """Return the recovered preferred v2/40 foot, centred in XY, z=0..4.

    Grammar and dimensions come from
    Mara7070/cad-apprenticeship/interfaces/chappel.py (Chappel donor corpus).
    The +0.6258 mm v2/40 size delta was physically coupon-tested and preferred
    over the 75-series foot. This is an isolated mating foot, not a body.
    """
    d = CHAPPEL
    delta = d.v2_foot_size_delta
    outer = Solid.make_loft([
        _chamfered_square(size + delta, d.foot_chamfer, z)
        for size, z in (
            (d.foot_bottom_size, 0),
            (d.foot_outer_at_transition, d.foot_transition_z),
            (d.foot_outer_at_grid_top, d.grid_height),
        )
    ], ruled=True)
    # Extend only the cutter by the recovered boolean overcut, never the foot.
    cutter_top = d.grid_height + 0.01
    inner_size = d.foot_inner_at_floor + delta
    inner = Solid.make_loft([
        _chamfered_square(inner_size, d.foot_inner_chamfer, d.foot_floor_thickness),
        _chamfered_square(
            inner_size + 2 * (cutter_top - d.foot_floor_thickness),
            d.foot_inner_chamfer, cutter_top,
        ),
    ], ruled=True)
    return outer.cut(inner)


def chappel_v2_foot_array(cells_x: int, cells_y: int) -> Compound | Solid:
    """Return unfused feet in the organizer lower-left frame, X then Y.

    CHAPPEL owns validation. The first centre is (27.3, 27.3); subsequent
    centres use the 55 mm pitch. No organizer body is included.
    """
    CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    origin = CHAPPEL.organizer_footprint(1) / 2
    return Compound([
        chappel_v2_foot_cell().moved(Location((
            origin + ix * CHAPPEL.pitch, origin + iy * CHAPPEL.pitch, 0,
        )))
        for ix in range(cells_x)
        for iy in range(cells_y)
    ])
