"""Adapt persisted tool outer rings to canonical geometry and layout."""

from app.models.schemas import Tool
from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates
from app.services.object_placement import nonconflicting_fit_candidates


def canonical_outline_for_tool(tool: Tool) -> CanonicalOutline:
    """Copy the tool's outer ring, ignoring holes and other tool fields."""
    return CanonicalOutline.from_points((point.x, point.y) for point in tool.points)


def fit_tool_candidates(
    tool: Tool, width: float, height: float, inset: float = 0.0
) -> tuple[PlacedOutline, ...]:
    """Enumerate fitting placements using the canonical layout primitive."""
    return fit_candidates(canonical_outline_for_tool(tool), width, height, inset)


def nonconflicting_tool_candidates(
    tool: Tool,
    width: float,
    height: float,
    occupied: tuple[PlacedOutline, ...] | list[PlacedOutline],
    inset: float = 0.0,
) -> tuple[PlacedOutline, ...]:
    """Filter fitting outer-ring placements through the collision composition."""
    return nonconflicting_fit_candidates(
        canonical_outline_for_tool(tool), width, height, occupied, inset
    )
