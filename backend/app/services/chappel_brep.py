"""Native Chappel B-rep blanks and recovered v2/40 mating feet."""

from build123d import Compound, Location, Solid, Wire

from app.services.chappel_dimensions import CHAPPEL


def chappel_organizer_blank(cells_x: int, cells_y: int) -> Solid:
    """Return a rectangular solid with its lower-left-bottom corner at (0, 0, 0).

    Dimensions are in millimetres. This outer blank has no mating or interior
    features and is not a finished printable organizer. Cell validation belongs
    to CHAPPEL and propagates unchanged.
    """
    width, depth = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return Solid.make_box(width, depth, CHAPPEL.organizer_height)


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
