"""Shapely adapter for exact occupied-outline collision checks."""

from shapely.geometry import Polygon

from app.services.object_geometry import PlacedOutline


def placements_conflict(a: PlacedOutline, b: PlacedOutline) -> bool:
    """Return whether occupied polygons intersect, including boundary contact.

    Invalid or empty polygons raise ValueError; geometry is never repaired or
    buffered. All rotation and translation come from the placement points.
    """
    polygons = (Polygon(a.points), Polygon(b.points))
    for polygon in polygons:
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Placement polygon must be nonempty and valid")
    return bool(polygons[0].intersects(polygons[1]))
