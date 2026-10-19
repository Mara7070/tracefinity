"""Native STEP round-trip and bounded export failure contracts."""

from pathlib import Path

import pytest
from build123d import Compound, Location, Solid, import_step
from OCP.BRep import BRep_Builder
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid

from app.services import chappel_export
from app.services.chappel_brep import chappel_insert_blank, chappel_insert_with_pockets
from app.services.chappel_export import export_chappel_step
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


def _round_trip(path, source):
    assert path.is_file() and path.stat().st_size > 0
    imported = import_step(path)
    assert len(imported.solids()) == 1
    assert BRepCheck_Analyzer(imported.wrapped).IsValid()
    solid = imported.solids()[0]
    assert BRepCheck_Analyzer(solid.wrapped).IsValid()
    for end in ("min", "max"):
        assert tuple(getattr(solid.bounding_box(), end)) == pytest.approx(
            tuple(getattr(source.bounding_box(), end)), rel=0, abs=1e-6,
        )
    assert solid.volume == pytest.approx(source.volume, rel=1e-9)
    return solid


@pytest.mark.parametrize("suffix", [".step", ".STEP", ".stp"])
@pytest.mark.parametrize("model", ["blank", "pocketed"])
def test_native_round_trip(request, tmp_path, model, suffix):
    source = request.getfixturevalue(model)
    before = _snapshot(source)
    wrapped = source.wrapped
    path = tmp_path / (model + suffix)
    result = export_chappel_step(source, path)
    assert isinstance(result, Path) and result == path
    solid = _round_trip(result, source)
    if model == "pocketed":
        for x, y in [(40, 45), (75, 70)]:
            assert not solid.is_inside((x, y, 24))
            assert not solid.is_inside((x, y, 13.001))
            assert solid.is_inside((x, y, 12.999))
        for point in [(29.99, 45, 24), (69.99, 70, 24), (55, 55, 24)]:
            assert solid.is_inside(point)
    assert source.wrapped.IsSame(wrapped)
    assert _snapshot(source) == before


def test_replace_and_relative_string_path(tmp_path, monkeypatch, blank, pocketed):
    monkeypatch.chdir(tmp_path)
    path = Path("requested.step")
    export_chappel_step(blank, str(path))
    _round_trip(path, blank)
    assert export_chappel_step(pocketed, str(path)) == path
    imported = _round_trip(path, pocketed)
    assert imported.volume < blank.volume
    assert not imported.is_inside((40, 45, 24))


@pytest.mark.parametrize("name", ["bad.stl", "bad", "bad.step.zip"])
def test_bad_suffix(tmp_path, blank, name):
    path = tmp_path / name
    with pytest.raises(ValueError, match="suffix"):
        export_chappel_step(blank, path)
    assert not path.exists()


@pytest.mark.parametrize("path", [None, 42, object(), b"bytes.step"])
def test_non_path(blank, path):
    with pytest.raises(ValueError, match="path"):
        export_chappel_step(blank, path)


def test_invalid_parents(tmp_path, blank):
    parent_file = tmp_path / "file"
    parent_file.write_text("unchanged")
    for parent in [tmp_path / "missing", parent_file]:
        with pytest.raises(ValueError, match="parent"):
            export_chappel_step(blank, parent / "part.step")
    assert not (tmp_path / "missing").exists()
    assert parent_file.read_text() == "unchanged"


def test_reject_native_topology_and_wrong_types_without_writing(tmp_path, blank):
    other = Solid.make_box(1, 1, 1).moved(Location((200, 0, 0)))
    compound = Compound([blank, other])
    multi = Solid(compound.wrapped)
    assert len(multi.solids()) == 2
    native = TopoDS_Solid()
    builder = BRep_Builder()
    builder.MakeSolid(native)
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    builder.Add(shell, blank.faces()[0].wrapped)
    builder.Add(native, shell)
    invalid = Solid(native)
    assert not BRepCheck_Analyzer(invalid.wrapped).IsValid()
    path = tmp_path / "reject.step"
    for shape in [None, object(), blank.faces()[0], compound, multi, invalid, Solid()]:
        with pytest.raises(ValueError):
            export_chappel_step(shape, path)
        assert not path.exists()


@pytest.mark.parametrize("behavior", ["exception", "false", "missing", "empty"])
def test_exporter_failures(tmp_path, monkeypatch, blank, behavior):
    path = tmp_path / "failed.step"

    def fail(shape, destination):
        assert shape is blank and destination == path
        if behavior == "exception":
            raise OSError("export failed")
        if behavior == "empty":
            destination.touch()
        return behavior != "false"

    monkeypatch.setattr(chappel_export, "export_step", fail)
    with pytest.raises(OSError if behavior == "exception" else RuntimeError):
        export_chappel_step(blank, path)
