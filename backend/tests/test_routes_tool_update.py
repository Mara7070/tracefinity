"""Tool update canonicalization and persistence through the real route."""

import asyncio

import pytest
from fastapi import HTTPException

from app.models.schemas import FingerHole, Point, Tool, ToolUpdateRequest
from app.services.tool_store import ToolStore


def _points(coordinates):
    return [Point(x=x, y=y) for x, y in coordinates]


@pytest.fixture
def stored_tool(tmp_path, monkeypatch):
    # Route import initializes paper detection; these tests do not need a model.
    monkeypatch.setattr("app.services.onnx_check.is_onnx_available", lambda: False)
    from app.api import routes

    store = ToolStore(tmp_path)
    tool = Tool(
        id="legacy-tool", name="Original",
        points=_points([(4, 3), (4, -1), (-2, -1), (-2, 3), (4, 3)]),
        finger_holes=[FingerHole(id="hole", x=0, y=0, radius=1)],
        interior_rings=[_points([(1, 1), (1, 0), (0, 0), (1, 1)])],
        smoothed=False, smooth_level=0.7,
        source_session_id="session", source_polygon_id="polygon",
        source_image_path="image.png", source_image_width=800, source_image_height=600,
        source_image_transform=[2, 0, 0, 2, 100, 200],
        category="Hand tools", drawer="Top", tags=["old"], project_ids=["project"],
        review_status="reviewed", needs_cleanup=True,
        thumbnail_path="thumbnail.png", created_at="2026-01-01T00:00:00Z",
    )
    store.set(tool.id, tool)
    monkeypatch.setattr(routes, "get_stores", lambda user_id: (None, store, None))
    return store, tool


def _update(tool, **fields):
    from app.api import routes

    return asyncio.run(routes.update_tool(
        None, tool.id, ToolUpdateRequest(**fields), user_id="test-user",
    ))


def _persisted(store, tool):
    return ToolStore(store.file_path.parent).get(tool.id).model_dump()


def test_points_update_persists_canonical_ring_and_preserves_other_fields(stored_tool):
    store, tool = stored_tool
    before = tool.model_dump()
    replacement = _points([(8, 5), (8, -2), (-3, -2), (-3, 5), (8, 5)])

    assert _update(tool, points=replacement).status == "ok"

    expected = {**before, "points": [p.model_dump() for p in _points([
        (-3, -2), (8, -2), (8, 5), (-3, 5),
    ])]}
    assert store.get(tool.id).model_dump() == expected
    assert _persisted(store, tool) == expected


@pytest.mark.parametrize("points_field", [{}, {"points": None}])
def test_metadata_update_preserves_legacy_ring(stored_tool, points_field):
    store, tool = stored_tool
    before = tool.model_dump()
    original_points = tool.points
    changes = dict(name="Renamed", category=None, drawer=None, tags=["new"],
                   project_ids=[], review_status=None, needs_cleanup=False)

    assert _update(tool, **changes, **points_field).status == "ok"

    assert tool.points is original_points
    expected = {**before, **changes}
    assert store.get(tool.id).model_dump() == expected
    assert _persisted(store, tool) == expected


@pytest.mark.parametrize("coordinates", [
    [], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (0, 0), (0, 0)],
])
def test_invalid_points_reject_without_mutating_store(stored_tool, coordinates):
    store, tool = stored_tool
    before = tool.model_dump()
    disk_before = store.file_path.read_bytes()

    with pytest.raises(HTTPException) as error:
        _update(tool, points=_points(coordinates), name="Must not persist")

    assert error.value.status_code == 400
    assert error.value.detail
    assert store.get(tool.id).model_dump() == before
    assert store.file_path.read_bytes() == disk_before
    assert _persisted(store, tool) == before


def test_empty_update_preserves_entire_legacy_tool(stored_tool):
    store, tool = stored_tool
    before = tool.model_dump()

    assert _update(tool).status == "ok"

    assert store.get(tool.id).model_dump() == before
    assert _persisted(store, tool) == before
