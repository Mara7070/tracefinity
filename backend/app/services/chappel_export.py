"""Native manufacturing derivatives of already-built Chappel geometry."""

import os
from pathlib import Path

from build123d import Solid, export_step


def export_chappel_step(shape: Solid, output_path: str | os.PathLike[str]) -> Path:
    """Write one valid native solid directly to an existing directory as STEP.

    Geometry and layout policy belong to the services that build the solid.
    Existing destinations may be replaced; exporter and filesystem errors propagate.
    """
    if not isinstance(shape, Solid) or shape.is_null:
        raise ValueError("shape must be a valid native Solid containing exactly one solid")
    if len(shape.solids()) != 1 or not shape.is_valid:
        raise ValueError("shape must be a valid native Solid containing exactly one solid")
    try:
        path = Path(os.fspath(output_path))
    except TypeError as exc:
        raise ValueError("output_path must be a string path or path-like object") from exc
    if path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("output_path must have a .step or .stp suffix")
    if not path.parent.is_dir():
        raise ValueError("output_path parent must exist and be a directory")

    if not export_step(shape, path):
        raise RuntimeError("STEP exporter reported failure")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("STEP export did not produce a nonempty file")
    return path
