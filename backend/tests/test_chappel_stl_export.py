"""Mesh-derivative contracts for native Chappel solids."""

from pathlib import Path

import pytest
import trimesh
from build123d import Compound, Location, Solid
from OCP.BRepCheck import BRepCheck_Analyzer

from app.services import chappel_export
from app.services.chappel_brep import chappel_insert_blank, chappel_insert_with_pockets
from app.services.chappel_export import export_chappel_stl
from app.services.object_geometry import CanonicalOutline, PlacedOutline


@pytest.fixture(scope="module")
def blank():
    return chappel_insert_blank(2, 3)


@pytest.fixture(scope="module")
def pocketed():
    outline = CanonicalOutline.from_points(((0, 0), (20, 0), (20, 10), (0, 10)))
    return chappel_insert_with_pockets([
        PlacedOutline(outline, x=30, y=40),
        PlacedOutline(outline, x=80, y=60, quarter_turns=1),
    ], 2, 3, depth=12)


def _snapshot(shape):
    bounds = shape.bounding_box()
    return (
        tuple(bounds.min), tuple(bounds.max), shape.volume,
        tuple(len(getattr(shape, kind)()) for kind in
              ("solids", "shells", "faces", "wires", "edges", "vertices")),
        BRepCheck_Analyzer(shape.wrapped).IsValid(),
    )


def _load_mesh(path: Path, source: Solid) -> trimesh.Trimesh:
    assert path.is_file() and path.stat().st_size > 0
    mesh = trimesh.load(path, force="mesh", process=True)
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight
    assert mesh.is_volume
    assert len(mesh.split()) == 1
    source_bounds = source.bounding_box()
    expected = (tuple(source_bounds.min), tuple(source_bounds.max))
    assert tuple(mesh.bounds[0]) == pytest.approx(expected[0], rel=0, abs=2e-3)
    assert tuple(mesh.bounds[1]) == pytest.approx(expected[1], rel=0, abs=2e-3)
    assert mesh.volume == pytest.approx(source.volume, rel=2e-5, abs=0.05)
    return mesh


@pytest.mark.parametrize("name", ["part.stl", "part.STL"])
@pytest.mark.parametrize("model", ["blank", "pocketed"])
def test_native_stl_derivative(request, tmp_path, model, name):
    source = request.getfixturevalue(model)
    before = _snapshot(source)
    wrapped = source.wrapped
    path = tmp_path / name
    result = export_chappel_stl(source, path)
    assert isinstance(result, Path) and result == path
    _load_mesh(result, source)
    assert source.wrapped.IsSame(wrapped)
    assert _snapshot(source) == before


def test_pockets_survive_mesh_derivative(tmp_path, blank, pocketed):
    blank_mesh = _load_mesh(export_chappel_stl(blank, tmp_path / "blank.stl"), blank)
    pocket_mesh = _load_mesh(export_chappel_stl(pocketed, tmp_path / "pocketed.stl"), pocketed)
    brep_removed = blank.volume - pocketed.volume
    mesh_removed = blank_mesh.volume - pocket_mesh.volume
    assert brep_removed == pytest.approx(20 * 10 * 12 * 2, rel=0, abs=1e-6)
    assert mesh_removed == pytest.approx(brep_removed, rel=2e-5, abs=0.1)
    assert len(pocket_mesh.faces) > len(blank_mesh.faces)


def test_replace_existing_destination(tmp_path, blank, pocketed):
    path = tmp_path / "replace.stl"
    first = _load_mesh(export_chappel_stl(blank, path), blank)
    first_volume = first.volume
    assert export_chappel_stl(pocketed, path) == path
    second = _load_mesh(path, pocketed)
    assert second.volume < first_volume


@pytest.mark.parametrize("name", ["bad.step", "bad", "bad.stl.zip"])
def test_bad_suffix(tmp_path, blank, name):
    path = tmp_path / name
    with pytest.raises(ValueError, match="suffix"):
        export_chappel_stl(blank, path)
    assert not path.exists()


@pytest.mark.parametrize("path", [None, 42, object(), b"bytes.stl"])
def test_non_path(blank, path):
    with pytest.raises(ValueError, match="path"):
        export_chappel_stl(blank, path)


def test_invalid_parents(tmp_path, blank):
    parent_file = tmp_path / "file"
    parent_file.write_text("unchanged")
    for parent in [tmp_path / "missing", parent_file]:
        with pytest.raises(ValueError, match="parent"):
            export_chappel_stl(blank, parent / "part.stl")
    assert not (tmp_path / "missing").exists()
    assert parent_file.read_text() == "unchanged"


def test_reject_wrong_or_multisolid_shape(tmp_path, blank):
    other = Solid.make_box(1, 1, 1).moved(Location((200, 0, 0)))
    compound = Compound([blank, other])
    multi = Solid(compound.wrapped)
    path = tmp_path / "reject.stl"
    for shape in [None, object(), blank.faces()[0], compound, multi, Solid()]:
        with pytest.raises(ValueError, match="shape"):
            export_chappel_stl(shape, path)
        assert not path.exists()


@pytest.mark.parametrize("behavior", ["exception", "false", "missing", "empty"])
def test_exporter_failures(tmp_path, monkeypatch, blank, behavior):
    path = tmp_path / "failed.stl"

    def fail(shape, destination):
        assert shape is blank and destination == path
        if behavior == "exception":
            raise OSError("export failed")
        if behavior == "empty":
            destination.touch()
        return behavior != "false"

    monkeypatch.setattr(chappel_export, "export_stl", fail)
    with pytest.raises(OSError if behavior == "exception" else RuntimeError):
        export_chappel_stl(blank, path)
