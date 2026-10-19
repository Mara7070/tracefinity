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


def placement_conflicts_any(
    candidate: PlacedOutline, occupied: tuple[PlacedOutline, ...] | list[PlacedOutline]
) -> bool:
    """Check occupied placements left to right, stopping at the first conflict.

    Invalid geometry in evaluated pairs raises ValueError via placements_conflict.
    """
    return any(placements_conflict(candidate, placement) for placement in occupied)


def conflicting_pairs(
    placements: tuple[PlacedOutline, ...] | list[PlacedOutline],
) -> tuple[tuple[int, int], ...]:
    """Return conflicting index pairs (i < j) in lexicographic order.

    Invalid geometry in evaluated pairs raises ValueError via placements_conflict.
    """
    return tuple(
        (i, j)
        for i in range(len(placements))
        for j in range(i + 1, len(placements))
        if placements_conflict(placements[i], placements[j])
    )
