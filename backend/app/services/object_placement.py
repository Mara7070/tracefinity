"""Filter rectangular fit candidates using occupied-outline collision checks."""

from app.services.object_collision import placements_conflict
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates


def nonconflicting_fit_candidates(
    outline: CanonicalOutline,
    width: float,
    height: float,
    occupied: tuple[PlacedOutline, ...] | list[PlacedOutline],
    inset: float = 0.0,
) -> tuple[PlacedOutline, ...]:
    """Return fitting placements in their original order, excluding conflicts.

    Boundary contact counts as conflict. Validation and geometry errors propagate
    from the fit and collision primitives without repair.
    """
    candidates = fit_candidates(outline, width, height, inset)
    return tuple(
        candidate
        for candidate in candidates
        if not any(placements_conflict(candidate, other) for other in occupied)
    )
