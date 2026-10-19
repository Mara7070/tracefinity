"""Native B-rep outer blanks for the Chappel organizer/25 series."""

from build123d import Solid

from app.services.chappel_dimensions import CHAPPEL


def chappel_organizer_blank(cells_x: int, cells_y: int) -> Solid:
    """Return a rectangular solid with its lower-left-bottom corner at (0, 0, 0).

    Dimensions are in millimetres. This outer blank has no mating or interior
    features and is not a finished printable organizer. Cell validation belongs
    to CHAPPEL and propagates unchanged.
    """
    width, depth = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return Solid.make_box(width, depth, CHAPPEL.organizer_height)
