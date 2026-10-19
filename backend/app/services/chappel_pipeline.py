"""Build explicit Chappel pockets and export one manufacturing derivative."""

import os
from collections.abc import Sequence
from math import isfinite
from numbers import Real
from pathlib import Path

from app.models.schemas import Tool
from app.services.chappel_brep import chappel_insert_with_pockets
from app.services.chappel_dimensions import CHAPPEL
from app.services.chappel_export import export_chappel_step, export_chappel_stl
from app.services.chappel_tool_layout import (
    first_minimal_chappel_organizer_fit,
    place_tools_in_chappel_organizer,
)
from app.services.object_geometry import PlacedOutline


def export_chappel_insert(
    placements: tuple[PlacedOutline, ...] | list[PlacedOutline],
    cells_x: int,
    cells_y: int,
    depth: float,
    output_path: str | os.PathLike[str],
    clearance: float = 0.0,
) -> Path:
    """Build accepted native geometry and export according to the explicit suffix.

    Unsupported suffixes fail before CAD work. Geometry and exporter errors
    propagate unchanged; the caller supplies the destination and placements.
    """
    suffix = Path(output_path).suffix.lower()
    if suffix in {".step", ".stp"}:
        exporter = export_chappel_step
    elif suffix == ".stl":
        exporter = export_chappel_stl
    else:
        raise ValueError("output_path must have a .step, .stp or .stl suffix")
    shape = chappel_insert_with_pockets(placements, cells_x, cells_y, depth, clearance)
    return exporter(shape, output_path)


def export_chappel_tools(
    tools: Sequence[Tool],
    cells_x: int,
    cells_y: int,
    gap: float,
    depth: float,
    output_path: str | os.PathLike[str],
    inset: float = 0.0,
    clearance: float = 0.0,
) -> Path:
    """Greedily place ordered tools in an explicit envelope, then export.

    Gap controls placement anchors; clearance independently expands CAD pockets.
    Delegate errors propagate unchanged except for an explicit no-fit result.
    """
    placements = place_tools_in_chappel_organizer(tools, cells_x, cells_y, gap, inset)
    if placements is None:
        raise ValueError(
            "ordered Tool set does not fit the supplied Chappel envelope under the greedy policy"
        )
    return export_chappel_insert(placements, cells_x, cells_y, depth, output_path, clearance)


def export_minimal_chappel_tool(
    tool: Tool,
    max_cells_x: int,
    max_cells_y: int,
    depth: float,
    output_path: str | os.PathLike[str],
    inset: float = 0.0,
    clearance: float = 0.0,
) -> Path:
    """Select an envelope reserving a conservative physical margin at all depths."""
    if (isinstance(inset, bool) or not isinstance(inset, Real)
            or not isfinite(inset) or inset < 0):
        raise ValueError("Inset must be finite and non-negative")
    if (isinstance(clearance, bool) or not isinstance(clearance, Real)
            or not isfinite(clearance) or clearance < 0):
        raise ValueError("clearance must be a finite nonnegative real number")
    # At maximum depth the 2 mm floor keeps cutters above the bottom ramp.
    # Reserve an axis-aligned region inside both rounded mid and full sections;
    # recovered insets reduce the full footprint per axis, hence the half inset.
    physical_inset = max(
        CHAPPEL.insert_mid_inset / 2 + CHAPPEL.insert_mid_corner_radius,
        CHAPPEL.insert_top_corner_radius,
    )
    effective_inset = max(inset, physical_inset + clearance)
    selected = first_minimal_chappel_organizer_fit(tool, max_cells_x, max_cells_y, effective_inset)
    if selected is None:
        raise ValueError("tool does not fit within the supplied Chappel cell bounds")
    cells_x, cells_y, placement = selected
    return export_chappel_insert((placement,), cells_x, cells_y, depth, output_path, clearance)
