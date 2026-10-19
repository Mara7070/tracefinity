"""Fit persisted tool outer rings into explicitly sized Chappel organizers."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from app.models.schemas import Tool
from app.services.chappel_dimensions import CHAPPEL
from app.services.object_collision import placements_conflict
from app.services.object_geometry import PlacedOutline
from app.services.object_placement import first_nonconflicting_fit_candidate
from app.services.tool_layout import (
    canonical_outline_for_tool,
    fit_tool_candidates,
    nonconflicting_tool_candidates,
)


def edge_anchor_tool_candidates(
    tool: Tool,
    cells_x: int,
    cells_y: int,
    occupied: tuple[PlacedOutline, ...] | list[PlacedOutline],
    inset: float = 0.0,
    gap: float = 0.0,
) -> tuple[PlacedOutline, ...]:
    """Return distinct, noncontacting quarter turns at lower-left edge anchors.

    Gap is an explicit offset from occupied bounding-box maxima, not a physical
    spacing policy. Zero-gap anchors can be rejected because touching conflicts.
    Orientations precede positions, which are ordered by (y, x).
    """
    if isinstance(gap, bool) or not isinstance(gap, Real) or not isfinite(gap) or gap < 0:
        raise ValueError("Gap must be a finite non-negative real")
    width, height = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    outline = canonical_outline_for_tool(tool)
    inset = float(inset)
    if not isfinite(inset) or inset < 0:
        raise ValueError("Inset must be finite and non-negative")
    if width <= 2 * inset or height <= 2 * inset:
        return ()

    xs, ys = {inset}, {inset}
    for other in occupied:
        _, _, max_x, max_y = other.bounds
        xs.add(max_x + gap)
        ys.add(max_y + gap)

    candidates = []
    seen = set()
    for turns in range(4):
        min_x, min_y, _, _ = outline.rotated_quarter_turns(turns).bounds
        for y in sorted(ys):
            for x in sorted(xs):
                candidate = PlacedOutline(outline, x - min_x, y - min_y, turns)
                left, bottom, right, top = candidate.bounds
                if not (inset <= left and inset <= bottom
                        and right <= width - inset and top <= height - inset):
                    continue
                points = candidate.points
                if points in seen:
                    continue
                if any(placements_conflict(candidate, other) for other in occupied):
                    continue
                seen.add(points)
                candidates.append(candidate)
    return tuple(candidates)


def place_tools_in_chappel_organizer(
    tools: Sequence[Tool],
    cells_x: int,
    cells_y: int,
    gap: float,
    inset: float = 0.0,
) -> tuple[PlacedOutline, ...] | None:
    """Greedily place tools in caller order using each first edge-anchor candidate."""
    occupied: list[PlacedOutline] = []
    for tool in tools:
        candidates = edge_anchor_tool_candidates(tool, cells_x, cells_y, occupied, inset, gap)
        if not candidates:
            return None
        occupied.append(candidates[0])
    return tuple(occupied)


def fit_tool_in_chappel_organizer(
    tool: Tool, cells_x: int, cells_y: int, inset: float = 0.0
) -> tuple[PlacedOutline, ...]:
    """Return fitting quarter turns in the explicit organizer/25-series envelope."""
    width, height = CHAPPEL.organizer_footprint_xy(cells_x, cells_y)
    return fit_tool_candidates(tool, width, height, inset)


def minimal_chappel_organizer_sizes(
    tool: Tool, max_cells_x: int, max_cells_y: int, inset: float = 0.0
) -> tuple[tuple[int, int], ...]:
    """Return every bounded fitting envelope with minimum cell area, in XY order."""
    for bound in (max_cells_x, max_cells_y):
        if type(bound) is not int or bound < 1:
            raise ValueError("cell bounds must be Python ints >= 1")

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


def first_minimal_chappel_organizer_fit(
    tool: Tool, max_cells_x: int, max_cells_y: int, inset: float = 0.0
) -> tuple[int, int, PlacedOutline] | None:
    """Select the lexicographically first minimum envelope, then its first fit."""
    sizes = minimal_chappel_organizer_sizes(tool, max_cells_x, max_cells_y, inset)
    if not sizes:
        return None
    cells_x, cells_y = min(sizes)
    candidates = fit_tool_in_chappel_organizer(tool, cells_x, cells_y, inset)
    if not candidates:
        raise RuntimeError("Selected minimum Chappel envelope yielded no fit candidate")
    return cells_x, cells_y, candidates[0]


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
