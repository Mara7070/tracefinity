"""Deterministic single-outline placements inside a rectangular usable region."""

from math import isfinite

from app.services.object_geometry import CanonicalOutline, PlacedOutline


def fit_candidates(
    outline: CanonicalOutline, width: float, height: float, inset: float = 0.0
) -> tuple[PlacedOutline, ...]:
    """Return distinct fitting quarter turns, aligned to the inset lower-left.

    Dimensions use the outline's units. Fit and point deduplication use exact
    comparisons; equivalent placements retain their first quarter turn.
    """
    width, height, inset = float(width), float(height), float(inset)
    if not isfinite(width) or width <= 0:
        raise ValueError("Width must be finite and positive")
    if not isfinite(height) or height <= 0:
        raise ValueError("Height must be finite and positive")
    if not isfinite(inset) or inset < 0:
        raise ValueError("Inset must be finite and non-negative")

    usable_width, usable_height = width - 2 * inset, height - 2 * inset
    if usable_width <= 0 or usable_height <= 0:
        return ()

    candidates = []
    seen = set()
    for turns in range(4):
        rotated = outline.rotated_quarter_turns(turns)
        if rotated.width <= usable_width and rotated.height <= usable_height:
            min_x, min_y, _, _ = rotated.bounds
            candidate = PlacedOutline(outline, inset - min_x, inset - min_y, turns)
            points = candidate.points
            if points not in seen:
                seen.add(points)
                candidates.append(candidate)
    return tuple(candidates)
