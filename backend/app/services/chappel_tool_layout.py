"""Fit persisted tool outer rings into explicitly sized Chappel organizers."""

from app.models.schemas import Tool
from app.services.chappel_dimensions import CHAPPEL
from app.services.object_geometry import PlacedOutline
from app.services.object_placement import first_nonconflicting_fit_candidate
from app.services.tool_layout import (
    canonical_outline_for_tool,
    fit_tool_candidates,
    nonconflicting_tool_candidates,
)


def fit_tool_in_chappel_organizer(
    tool: Tool, cells_x: int, cells_y: int, inset: float = 0.0
) -> tuple[PlacedOutline, ...]:
    """Return fitting quarter turns in the explicit organizer/25-series envelope."""
    width, height = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return fit_tool_candidates(tool, width, height, inset)


def minimal_chappel_organizer_sizes(
    tool: Tool, max_cells_x: int, max_cells_y: int, inset: float = 0.0
) -> tuple[tuple[int, int], ...]:
    """Return all bounded minimum-cell-area fitting envelopes in lexicographic order."""
    for maximum in (max_cells_x, max_cells_y):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("max cells must be a Python int >= 1")

    minimum_area = None
    sizes = []
    for cells_x in range(1, max_cells_x + 1):
        for cells_y in range(1, max_cells_y + 1):
            if not fit_tool_in_chappel_organizer(tool, cells_x, cells_y, inset):
                continue
            area = cells_x * cells_y
            if minimum_area is None or area < minimum_area:
                minimum_area = area
                sizes = [(cells_x, cells_y)]
            elif area == minimum_area:
                sizes.append((cells_x, cells_y))
    return tuple(sizes)


def nonconflicting_tool_in_chappel_organizer(
    tool: Tool,
    cells_x: int,
    cells_y: int,
    occupied: tuple[PlacedOutline, ...] | list[PlacedOutline],
    inset: float = 0.0,
) -> tuple[PlacedOutline, ...]:
    """Return fitting outer-ring placements that do not contact occupied outlines."""
    width, height = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return nonconflicting_tool_candidates(tool, width, height, occupied, inset)


def first_nonconflicting_tool_in_chappel_organizer(
    tool: Tool,
    cells_x: int,
    cells_y: int,
    occupied: tuple[PlacedOutline, ...] | list[PlacedOutline],
    inset: float = 0.0,
) -> PlacedOutline | None:
    """Return the first surviving placement in canonical quarter-turn order."""
    width, height = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return first_nonconflicting_fit_candidate(
        canonical_outline_for_tool(tool), width, height, occupied, inset
    )
