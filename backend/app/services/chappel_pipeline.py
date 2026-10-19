"""Build explicit Chappel pockets and export one manufacturing derivative."""

import os
from pathlib import Path

from app.services.chappel_brep import chappel_insert_with_pockets
from app.services.chappel_export import export_chappel_step, export_chappel_stl
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
