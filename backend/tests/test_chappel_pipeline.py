"""Explicit build/export delegation and real manufacturing artifact evidence."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import trimesh
from build123d import import_step
from OCP.BRepCheck import BRepCheck_Analyzer

from app.models.schemas import Point, Tool
from app.services import chappel_pipeline
from app.services.chappel_brep import chappel_insert_blank, chappel_insert_with_pockets
from app.services.chappel_tool_layout import minimal_chappel_organizer_sizes
from app.services.object_geometry import CanonicalOutline, PlacedOutline


@pytest.fixture
def placements():
    outline = CanonicalOutline.from_points(((0, 0), (20, 0), (20, 10), (0, 10)))
    return [PlacedOutline(outline, x=30, y=40),
            PlacedOutline(outline, x=80, y=60, quarter_turns=1)]


@pytest.mark.parametrize("suffix,selected", [
    (".step", "step"), (".STEP", "step"), (".stp", "step"),
    (".stl", "stl"), (".STL", "stl"),
])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("string_path", [False, True])
def test_exact_delegation(monkeypatch, tmp_path, placements, suffix, selected,
                          container, string_path):
    placements = container(placements)
    before = deepcopy(placements)
    original_objects = tuple(placements)
    shape = object()
    returned = Path("exporter-result")
    builder = Mock(return_value=shape)
    step = Mock(return_value=returned)
    stl = Mock(return_value=returned)
    monkeypatch.setattr(chappel_pipeline, "chappel_insert_with_pockets", builder)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_step", step)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_stl", stl)
    destination = tmp_path / ("explicit" + suffix)
    if string_path:
        destination = str(destination)
    result = chappel_pipeline.export_chappel_insert(
        placements, 2, 3, 12.5, destination, clearance=0.25,
    )
    assert result is returned
    builder.assert_called_once_with(placements, 2, 3, 12.5, 0.25)
    assert builder.call_args.args[0] is placements
    (step if selected == "step" else stl).assert_called_once_with(shape, destination)
    (stl if selected == "step" else step).assert_not_called()
    assert placements == before
    assert all(a is b for a, b in zip(placements, original_objects))


@pytest.mark.parametrize("name", ["part", "part.zip", "part.step.zip", ".stl", "part.3mf"])
def test_bad_suffix_before_build(monkeypatch, tmp_path, name):
    builder = Mock(side_effect=AssertionError("must not build"))
    monkeypatch.setattr(chappel_pipeline, "chappel_insert_with_pockets", builder)
    with pytest.raises(ValueError, match="suffix"):
        chappel_pipeline.export_chappel_insert([], 0, 0, -1, tmp_path / name)
    builder.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".step", ".stl"])
@pytest.mark.parametrize("stage", ["builder", "exporter"])
def test_exception_identity(monkeypatch, tmp_path, placements, suffix, stage):
    failure = OSError("original failure")
    builder = Mock(side_effect=failure) if stage == "builder" else Mock(return_value=object())
    exporter = Mock(side_effect=failure)
    monkeypatch.setattr(chappel_pipeline, "chappel_insert_with_pockets", builder)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_step", exporter)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_stl", exporter)
    with pytest.raises(OSError) as caught:
        chappel_pipeline.export_chappel_insert(placements, 2, 3, 12, tmp_path / ("part" + suffix))
    assert caught.value is failure
    builder.assert_called_once()
    if stage == "builder":
        exporter.assert_not_called()
    else:
        exporter.assert_called_once()


@pytest.mark.parametrize("case", ["collision", "breach", "cells_x", "cells_y", "depth", "clearance", "empty"])
def test_real_geometry_errors_unchanged(tmp_path, placements, case):
    cells_x, cells_y, depth, clearance = 2, 3, 12, 0
    if case == "collision":
        placements.append(placements[0])
    elif case == "breach":
        placements.append(PlacedOutline(placements[0].outline, x=1, y=80))
        depth = 20
    elif case == "cells_x":
        cells_x = 0
    elif case == "cells_y":
        cells_y = 0
    elif case == "depth":
        depth = 24
    elif case == "clearance":
        clearance = -1
    else:
        placements.clear()
    before = deepcopy(placements)
    with pytest.raises(ValueError) as direct:
        chappel_insert_with_pockets(placements, cells_x, cells_y, depth, clearance)
    path = tmp_path / "invalid.step"
    with pytest.raises(type(direct.value)) as delegated:
        chappel_pipeline.export_chappel_insert(placements, cells_x, cells_y, depth, path, clearance)
    assert delegated.value.args == direct.value.args
    assert placements == before
    assert not path.exists()


@pytest.mark.parametrize("suffix", [".step", ".stl"])
def test_real_two_pocket_artifact(tmp_path, placements, suffix):
    before = deepcopy(placements)
    blank = chappel_insert_blank(2, 3)
    expected_volume = blank.volume - 2 * 20 * 10 * 12
    destination = tmp_path / ("pocketed" + suffix)
    result = chappel_pipeline.export_chappel_insert(placements, 2, 3, 12, destination)
    assert result == destination
    assert result.is_file() and result.stat().st_size > 0
    bounds = blank.bounding_box()
    if suffix == ".step":
        imported = import_step(result)
        assert len(imported.solids()) == 1
        assert BRepCheck_Analyzer(imported.wrapped).IsValid()
        solid = imported.solids()[0]
        assert solid.is_valid
        assert solid.volume == pytest.approx(expected_volume, rel=1e-9)
        for end in ("min", "max"):
            assert tuple(getattr(solid.bounding_box(), end)) == pytest.approx(
                tuple(getattr(bounds, end)), rel=0, abs=1e-6,
            )
        for x, y in [(40, 45), (75, 70)]:
            assert not solid.is_inside((x, y, 24))
            assert not solid.is_inside((x, y, 13.001))
            assert solid.is_inside((x, y, 12.999))
    else:
        mesh = trimesh.load(result, force="mesh", process=True)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight and mesh.is_volume
        assert len(mesh.split()) == 1
        assert mesh.volume == pytest.approx(expected_volume, rel=2e-5, abs=0.05)
        assert tuple(mesh.bounds[0]) == pytest.approx(tuple(bounds.min), rel=0, abs=2e-3)
        assert tuple(mesh.bounds[1]) == pytest.approx(tuple(bounds.max), rel=0, abs=2e-3)
    assert placements == before
    assert list(tmp_path.iterdir()) == [destination]


@pytest.fixture
def rectangular_tool():
    return Tool(id="rectangle", name="Rectangle", points=[
        Point(x=x, y=y) for x, y in ((0, 0), (60, 0), (60, 20), (0, 20))
    ])


@pytest.mark.parametrize("string_path", [False, True])
@pytest.mark.parametrize("inset,clearance,expected_inset", [
    (0, 0, 3.0), (0.5, 0.25, 3.25), (5, 0.25, 5),
])
def test_minimal_exact_delegation(monkeypatch, tmp_path, rectangular_tool, placements, string_path,
                                  inset, clearance, expected_inset):
    tool = rectangular_tool
    before = deepcopy(tool)
    placement = placements[0]
    placement_before = deepcopy(placement)
    selector = Mock(return_value=(2, 3, placement))
    returned = Path("selected-result")
    exporter = Mock(return_value=returned)
    monkeypatch.setattr(chappel_pipeline, "first_minimal_chappel_organizer_fit", selector)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    destination = tmp_path / "minimal.step"
    if string_path:
        destination = str(destination)
    result = chappel_pipeline.export_minimal_chappel_tool(tool, 4, 5, 12.5, destination, inset, clearance)
    selector.assert_called_once_with(tool, 4, 5, expected_inset)
    assert selector.call_args.args[0] is tool
    exporter.assert_called_once_with((placement,), 2, 3, 12.5, destination, clearance)
    assert exporter.call_args.args[0][0] is placement
    assert exporter.call_args.args[4] is destination
    assert result is returned
    assert tool == before
    assert placement == placement_before


def test_minimal_no_fit(monkeypatch, tmp_path, rectangular_tool):
    exporter = Mock()
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    with pytest.raises(ValueError) as caught:
        chappel_pipeline.export_minimal_chappel_tool(rectangular_tool, 1, 1, 12, tmp_path / "no.step")
    assert str(caught.value) == "tool does not fit within the supplied Chappel cell bounds"
    exporter.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["selector", "exporter"])
def test_minimal_exception_identity(monkeypatch, tmp_path, rectangular_tool, placements, stage):
    failure = ValueError("delegate failure")
    selector = Mock(side_effect=failure) if stage == "selector" else Mock(return_value=(1, 2, placements[0]))
    exporter = Mock(side_effect=failure)
    monkeypatch.setattr(chappel_pipeline, "first_minimal_chappel_organizer_fit", selector)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    with pytest.raises(ValueError) as caught:
        chappel_pipeline.export_minimal_chappel_tool(rectangular_tool, 3, 3, 12, tmp_path / "part.step")
    assert caught.value is failure
    selector.assert_called_once()
    if stage == "selector":
        exporter.assert_not_called()
    else:
        exporter.assert_called_once()


def test_minimal_suffix_validation_after_selection(monkeypatch, tmp_path, rectangular_tool):
    selector = Mock(wraps=chappel_pipeline.first_minimal_chappel_organizer_fit)
    exporter = Mock(wraps=chappel_pipeline.export_chappel_insert)
    builder = Mock(side_effect=AssertionError("must not build"))
    monkeypatch.setattr(chappel_pipeline, "first_minimal_chappel_organizer_fit", selector)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    monkeypatch.setattr(chappel_pipeline, "chappel_insert_with_pockets", builder)
    with pytest.raises(ValueError, match="output_path must have a .step, .stp or .stl suffix"):
        chappel_pipeline.export_minimal_chappel_tool(rectangular_tool, 3, 3, 12, tmp_path / "part.zip")
    selector.assert_called_once()
    exporter.assert_called_once()
    assert exporter.call_args.args[1:3] == (1, 2)
    builder.assert_not_called()


@pytest.mark.parametrize("case", ["bounds", "tool", "inset", "depth", "clearance"])
def test_minimal_real_validation_errors(tmp_path, rectangular_tool, case):
    tool = rectangular_tool
    max_x, max_y, inset, depth, clearance = 3, 3, 0, 12, 0
    if case == "bounds":
        max_x = 0
    elif case == "tool":
        tool = tool.model_copy(update={"points": []})
    elif case == "inset":
        inset = -1
    elif case == "depth":
        depth = 24
    else:
        clearance = -1
    before = deepcopy(tool)
    destination = tmp_path / "invalid.step"
    with pytest.raises(ValueError) as direct:
        selected = chappel_pipeline.first_minimal_chappel_organizer_fit(tool, max_x, max_y, inset)
        cells_x, cells_y, placement = selected
        chappel_pipeline.export_chappel_insert((placement,), cells_x, cells_y, depth, destination, clearance)
    with pytest.raises(type(direct.value)) as delegated:
        chappel_pipeline.export_minimal_chappel_tool(tool, max_x, max_y, depth, destination, inset, clearance)
    assert delegated.value.args == direct.value.args
    assert tool == before
    assert not destination.exists()


@pytest.mark.parametrize("suffix", [".step", ".stl"])
@pytest.mark.parametrize("depth,small", [(12, False), (23, True)])
def test_minimal_real_artifact(tmp_path, rectangular_tool, suffix, depth, small):
    if small:
        rectangular_tool = rectangular_tool.model_copy(update={"points": [
            Point(x=x, y=y) for x, y in ((0, 0), (20, 0), (20, 10), (0, 10))
        ]})
    before = deepcopy(rectangular_tool)
    assert minimal_chappel_organizer_sizes(rectangular_tool, 3, 3) == (
        ((1, 1),) if small else ((1, 2), (2, 1))
    )
    blank = chappel_insert_blank(1, 1 if small else 2)
    bounds = blank.bounding_box()
    expected_volume = blank.volume - (20 * 10 if small else 60 * 20) * depth
    destination = tmp_path / ("minimal" + suffix)
    result = chappel_pipeline.export_minimal_chappel_tool(rectangular_tool, 3, 3, depth, destination)
    assert result == destination
    assert result.is_file() and result.stat().st_size > 0
    if suffix == ".step":
        imported = import_step(result)
        assert BRepCheck_Analyzer(imported.wrapped).IsValid()
        assert len(imported.solids()) == 1
        solid = imported.solids()[0]
        assert solid.is_valid and solid.volume > 0
        assert solid.volume == pytest.approx(expected_volume, rel=1e-9)
        for end in ("min", "max"):
            assert tuple(getattr(solid.bounding_box(), end)) == pytest.approx(
                tuple(getattr(bounds, end)), rel=0, abs=1e-6,
            )
    else:
        mesh = trimesh.load(result, force="mesh", process=True)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight and mesh.is_volume
        assert len(mesh.split()) == 1
        assert mesh.volume == pytest.approx(expected_volume, rel=2e-5, abs=0.05)
        assert tuple(mesh.bounds[0]) == pytest.approx(tuple(bounds.min), rel=0, abs=2e-3)
        assert tuple(mesh.bounds[1]) == pytest.approx(tuple(bounds.max), rel=0, abs=2e-3)
    assert rectangular_tool == before


@pytest.fixture
def ordered_tools():
    return [Tool(id=name, name=name, points=[
        Point(x=x, y=y) for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]) for name, width, height in [("wide", 20, 10), ("narrow", 12, 8)]]


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("string_path", [False, True])
def test_tools_exact_delegation(monkeypatch, tmp_path, ordered_tools, placements,
                                container, string_path):
    tools = container(ordered_tools)
    before = deepcopy(tools)
    originals = tuple(tools)
    placement_before = deepcopy(placements)
    original_placements = tuple(placements)
    placer = Mock(return_value=placements)
    returned = Path("tools-result")
    exporter = Mock(return_value=returned)
    monkeypatch.setattr(chappel_pipeline, "place_tools_in_chappel_organizer", placer)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    destination = tmp_path / "tools.step"
    if string_path:
        destination = str(destination)
    result = chappel_pipeline.export_chappel_tools(
        tools, 2, 3, 1.5, 12.5, destination, 2.0, 0.25,
    )
    placer.assert_called_once_with(tools, 2, 3, 1.5, 2.0)
    assert placer.call_args.args[0] is tools
    exporter.assert_called_once_with(placements, 2, 3, 12.5, destination, 0.25)
    assert exporter.call_args.args[0] is placements
    assert exporter.call_args.args[4] is destination
    assert result is returned
    assert tools == before and placements == placement_before
    assert all(a is b for a, b in zip(tools, originals, strict=True))
    assert all(a is b for a, b in zip(placements, original_placements, strict=True))


@pytest.mark.parametrize("mocked", [False, True])
def test_tools_no_fit(monkeypatch, tmp_path, rectangular_tool, mocked):
    placer = Mock(return_value=None) if mocked else Mock(
        wraps=chappel_pipeline.place_tools_in_chappel_organizer,
    )
    exporter = Mock()
    monkeypatch.setattr(chappel_pipeline, "place_tools_in_chappel_organizer", placer)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    tools = [rectangular_tool]
    before = deepcopy(tools)
    with pytest.raises(ValueError) as caught:
        chappel_pipeline.export_chappel_tools(tools, 1, 1, 1, 12, tmp_path / "no.step")
    assert str(caught.value) == (
        "ordered Tool set does not fit the supplied Chappel envelope under the greedy policy"
    )
    placer.assert_called_once_with(tools, 1, 1, 1, 0.0)
    exporter.assert_not_called()
    assert tools == before
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["placement", "export"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_tools_exception_identity(monkeypatch, tmp_path, ordered_tools, placements,
                                  stage, error_type):
    failure = error_type("original delegate failure")
    placer = Mock(side_effect=failure) if stage == "placement" else Mock(return_value=placements)
    exporter = Mock(side_effect=failure)
    monkeypatch.setattr(chappel_pipeline, "place_tools_in_chappel_organizer", placer)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    with pytest.raises(error_type) as caught:
        chappel_pipeline.export_chappel_tools(ordered_tools, 1, 1, 2, 12, tmp_path / "part.step")
    assert caught.value is failure
    placer.assert_called_once()
    if stage == "placement":
        exporter.assert_not_called()
    else:
        exporter.assert_called_once()


def test_tools_bad_suffix_delegated_before_cad(monkeypatch, tmp_path, ordered_tools):
    exporter = Mock(wraps=chappel_pipeline.export_chappel_insert)
    builder = Mock(side_effect=AssertionError("must not build"))
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    monkeypatch.setattr(chappel_pipeline, "chappel_insert_with_pockets", builder)
    with pytest.raises(ValueError, match="output_path must have a .step, .stp or .stl suffix"):
        chappel_pipeline.export_chappel_tools(ordered_tools, 1, 1, 2, 12, tmp_path / "part.zip", 5)
    exporter.assert_called_once()
    builder.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".step", ".stl"])
@pytest.mark.parametrize("clearance", [0.0, 0.25])
def test_tools_real_artifact(tmp_path, ordered_tools, suffix, clearance):
    before = deepcopy(ordered_tools)
    originals = tuple(ordered_tools)
    blank = chappel_insert_blank(1, 1)
    expected_volume = blank.volume - 12 * (
        (20 + 2 * clearance) * (10 + 2 * clearance)
        + (12 + 2 * clearance) * (8 + 2 * clearance)
    )
    destination = tmp_path / ("tools" + suffix)
    result = chappel_pipeline.export_chappel_tools(
        ordered_tools, 1, 1, 2, 12, destination, inset=5, clearance=clearance,
    )
    assert result == destination
    assert result.is_file() and result.stat().st_size > 0
    if suffix == ".step":
        imported = import_step(result)
        assert BRepCheck_Analyzer(imported.wrapped).IsValid()
        assert len(imported.solids()) == 1
        solid = imported.solids()[0]
        assert solid.is_valid
        assert solid.volume == pytest.approx(expected_volume, rel=1e-9)
        # Greedy order puts the second rectangle to the right of the first.
        for x, y in [(15, 10), (33, 9)]:
            assert not solid.is_inside((x, y, 24))
            assert solid.is_inside((x, y, 12.999))
    else:
        mesh = trimesh.load(result, force="mesh", process=True)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight and mesh.is_volume
        assert len(mesh.split()) == 1
        assert mesh.volume == pytest.approx(expected_volume, rel=2e-5, abs=0.05)
    assert ordered_tools == before
    assert all(a is b for a, b in zip(ordered_tools, originals, strict=True))


def test_tools_real_clearance_collision(tmp_path, ordered_tools):
    before = deepcopy(ordered_tools)
    placements = chappel_pipeline.place_tools_in_chappel_organizer(ordered_tools, 1, 1, 1, 5)
    assert placements is not None
    assert chappel_insert_with_pockets(placements, 1, 1, 12).is_valid
    with pytest.raises(ValueError, match="^Clearance-expanded pockets") as direct:
        chappel_insert_with_pockets(placements, 1, 1, 12, 0.75)
    destination = tmp_path / "collision.step"
    with pytest.raises(type(direct.value)) as delegated:
        chappel_pipeline.export_chappel_tools(ordered_tools, 1, 1, 1, 12, destination, 5, 0.75)
    assert delegated.value.args == direct.value.args
    assert ordered_tools == before
    assert not destination.exists()


@pytest.mark.parametrize("field,message", [
    ("inset", "Inset must be finite and non-negative"),
    ("clearance", "clearance must be a finite nonnegative real number"),
])
@pytest.mark.parametrize("value", [True, False, -1, float("nan"), float("inf"),
                                  -float("inf"), "0", None, 1j])
def test_minimal_invalid_margin_before_selection(monkeypatch, tmp_path, rectangular_tool,
                                                field, message, value):
    selector = Mock()
    exporter = Mock()
    monkeypatch.setattr(chappel_pipeline, "first_minimal_chappel_organizer_fit", selector)
    monkeypatch.setattr(chappel_pipeline, "export_chappel_insert", exporter)
    with pytest.raises(ValueError, match=f"^{message}$"):
        chappel_pipeline.export_minimal_chappel_tool(
            rectangular_tool, 3, 3, 12, tmp_path / "invalid.step", **{field: value},
        )
    selector.assert_not_called()
    exporter.assert_not_called()
