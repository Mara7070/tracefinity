from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.config import ensure_user_dirs
from app.main import app
from app.models.schemas import FingerHole, Point, Polygon, Session
from app.services.tool_namer import (
    FallbackToolNamer,
    OllamaToolNamer,
    OpenRouterToolNamer,
    ToolNamerConfig,
    create_tool_namer,
    name_polygons,
    parse_label_response,
    validate_label,
)
from app.services.tool_store import ToolStore


def _square_poly(label: str = "tool 1") -> Polygon:
    return Polygon(
        id="poly-1",
        label=label,
        points=[
            Point(x=10, y=10),
            Point(x=90, y=10),
            Point(x=90, y=90),
            Point(x=10, y=90),
        ],
    )


class StaticNamer:
    def __init__(self, label: str):
        self.label = label
        self.calls: list[bytes] = []

    async def name(self, image_png: bytes) -> str | None:
        self.calls.append(image_png)
        return self.label


class FailingNamer:
    async def name(self, _image_png: bytes) -> str | None:
        raise RuntimeError("namer unavailable")


def test_disabled_provider_keeps_fallback_labels(tmp_path):
    image_path = tmp_path / "tool.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 255, dtype=np.uint8))

    result = asyncio.run(
        name_polygons(str(image_path), [_square_poly()], namer=FallbackToolNamer())
    )

    assert result[0].label == "tool 1"


def test_create_tool_namer_dispatches_by_provider():
    assert isinstance(
        create_tool_namer(ToolNamerConfig(provider="none")),
        FallbackToolNamer,
    )
    assert isinstance(
        create_tool_namer(ToolNamerConfig(provider="ollama")),
        OllamaToolNamer,
    )
    assert isinstance(
        create_tool_namer(ToolNamerConfig(provider="openrouter")),
        OpenRouterToolNamer,
    )
    assert isinstance(
        create_tool_namer(ToolNamerConfig(provider="unsupported")),
        FallbackToolNamer,
    )


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}

    def json(self):
        return self._json_body


def test_openrouter_namer_falls_through_to_next_model_on_429(monkeypatch):
    calls: list[str] = []

    async def fake_post(_self, _url, json, headers):
        calls.append(json["model"])
        if json["model"] == "model-a":
            return _FakeResponse(429, headers={"retry-after": "0"})
        return _FakeResponse(200, {"choices": [{"message": {"content": '{"name":"hacksaw"}'}}]})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    namer = OpenRouterToolNamer(
        ToolNamerConfig(
            provider="openrouter",
            openrouter_api_key="key",
            openrouter_model="model-a,model-b",
        )
    )

    label = asyncio.run(namer.name(b"\x89PNG"))

    assert label == "hacksaw"
    assert calls == ["model-a", "model-a", "model-b"]


def test_openrouter_namer_returns_none_when_content_missing(monkeypatch):
    async def fake_post(_self, _url, json, headers):
        return _FakeResponse(200, {"choices": [{"message": {}}]})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    namer = OpenRouterToolNamer(
        ToolNamerConfig(provider="openrouter", openrouter_api_key="key", openrouter_model="model-a")
    )

    assert asyncio.run(namer.name(b"\x89PNG")) is None


def test_parse_valid_ollama_json_label():
    assert parse_label_response('{"name":"Needle Nose Pliers"}') == "needle nose pliers"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "unknown",
        "object",
        "this name is far too long to be a useful short tool label",
        "one two three four five",
    ],
)
def test_rejects_unusable_names(name):
    assert validate_label(name) is None


def test_name_polygons_uses_namer_for_generic_labels_only(tmp_path):
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (90, 90), (0, 0, 0), -1)
    cv2.rectangle(image, (130, 10), (210, 90), (0, 0, 0), -1)
    image_path = tmp_path / "tools.png"
    cv2.imwrite(str(image_path), image)

    polygons = [
        _square_poly("tool 1"),
        Polygon(
            id="poly-2",
            label="marked caliper",
            points=[
                Point(x=130, y=10),
                Point(x=210, y=10),
                Point(x=210, y=90),
                Point(x=130, y=90),
            ],
        ),
    ]
    namer = StaticNamer("needle nose pliers")

    result = asyncio.run(name_polygons(str(image_path), polygons, namer=namer))

    assert len(namer.calls) == 1
    assert namer.calls[0].startswith(b"\x89PNG")
    assert [p.label for p in result] == ["needle nose pliers", "marked caliper"]


def test_name_polygons_keeps_fallback_when_namer_fails(tmp_path):
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (90, 90), (0, 0, 0), -1)
    image_path = tmp_path / "tool.png"
    cv2.imwrite(str(image_path), image)

    result = asyncio.run(
        name_polygons(str(image_path), [_square_poly()], namer=FailingNamer())
    )

    assert result[0].label == "tool 1"


def test_trace_route_applies_tool_namer_before_persisting(tmp_path, monkeypatch):
    storage_path = tmp_path / "storage"
    user_path = storage_path / "default"
    ensure_user_dirs(user_path)
    routes._store_cache.clear()
    routes._project_store_cache.clear()
    monkeypatch.setattr(routes.settings, "storage_path", storage_path)
    monkeypatch.setattr(routes.settings, "tracers", "isnet")
    monkeypatch.setattr(routes.settings, "google_api_key", None)
    monkeypatch.setattr(routes.settings, "openrouter_api_key", None)

    sessions, _, _ = routes.get_stores("default")
    image_path = user_path / "processed" / "corrected.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 255, dtype=np.uint8))
    sessions.set(
        "session-1",
        Session(
            id="session-1",
            created_at=datetime.now(UTC).isoformat(),
            corrected_image_path="default/processed/corrected.png",
            scale_factor=1.0,
        ),
    )

    class FakeTracer:
        async def trace_tools(self, *_args, **_kwargs):
            return [_square_poly("tool 1")], None

    async def fake_name_polygons(_image_path, polygons):
        polygons[0].label = "hacksaw"
        return polygons

    monkeypatch.setattr(routes, "_get_tracer", lambda _tracer_id=None: FakeTracer())
    monkeypatch.setattr(routes, "name_polygons", fake_name_polygons)

    client = TestClient(app)
    response = client.post(
        "/api/sessions/session-1/trace",
        json={"provider": "google", "tracer": "isnet"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"polygons", "mask_url"}
    assert body["polygons"][0]["label"] == "hacksaw"
    assert sessions.get("session-1").polygons[0].label == "hacksaw"


def test_save_tools_from_session_persists_polygon_label(tmp_path, monkeypatch):
    storage_path = tmp_path / "storage"
    user_path = storage_path / "default"
    ensure_user_dirs(user_path)
    routes._store_cache.clear()
    routes._project_store_cache.clear()
    monkeypatch.setattr(routes.settings, "storage_path", storage_path)

    sessions, tools, _ = routes.get_stores("default")
    image_path = user_path / "processed" / "corrected.png"
    cv2.imwrite(str(image_path), np.full((120, 120, 3), 255, dtype=np.uint8))
    sessions.set(
        "session-1",
        Session(
            id="session-1",
            created_at=datetime.now(UTC).isoformat(),
            corrected_image_path="default/processed/corrected.png",
            scale_factor=1.0,
            polygons=[_square_poly("hacksaw")],
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/api/sessions/session-1/save-tools",
        json={"polygon_ids": ["poly-1"]},
    )

    assert response.status_code == 200
    tool_id = response.json()["tool_ids"][0]
    assert tools.get(tool_id).name == "hacksaw"


@pytest.mark.parametrize("reverse", [False, True], ids=["ccw", "cw"])
@pytest.mark.parametrize("start", range(4))
@pytest.mark.parametrize("closed", [False, True], ids=["open", "closed-with-duplicate"])
def test_save_tools_canonical_outer_ring(tmp_path, monkeypatch, reverse, start, closed):
    monkeypatch.setattr(routes.settings, "storage_path", tmp_path)
    monkeypatch.setattr(routes, "_store_cache", {})
    ensure_user_dirs(tmp_path / "default")
    sessions, tools, _ = routes.get_stores("default")
    vertices = [(20, 40), (100, 40), (80, 80), (20, 80)]
    if reverse:
        vertices.reverse()
    vertices = vertices[start:] + vertices[:start]
    if closed:
        vertices = vertices[:1] + vertices + vertices[:1]
    poly = Polygon(
        id="poly-1", label="probe",
        points=[Point(x=x, y=y) for x, y in vertices],
        finger_holes=[FingerHole(
            id="hole-1", x=40, y=60, radius=3, shape="rectangle",
            width=7, height=5, rotation=27, depth_override=4,
        )],
        interior_rings=[[
            Point(x=50, y=50), Point(x=40, y=50),
            Point(x=40, y=60), Point(x=50, y=50),
        ]],
    )
    sessions.set("session-1", Session(id="session-1", scale_factor=0.5, polygons=[poly]))
    response = TestClient(app).post("/api/sessions/session-1/save-tools", json={})
    assert response.status_code == 200, response.text
    (tool_id,) = response.json()["tool_ids"]
    # Reload the persisted representation, including its Point list type.
    tool = ToolStore(tools.file_path.parent).get(tool_id)
    assert isinstance(tool.points, list)
    assert all(isinstance(p, Point) for p in tool.points)
    ring = [(p.x, p.y) for p in tool.points]
    assert ring == [(-20, -10), (20, -10), (10, 10), (-20, 10)]
    assert ring[0] == min(ring)
    assert sum(x * ny - nx * y for (x, y), (nx, ny) in zip(ring, ring[1:] + ring[:1])) > 0
    # The original 80 x 40 px bounds become 40 x 20 mm, centered at zero.
    assert (min(p.x for p in tool.points), max(p.x for p in tool.points)) == (-20, 20)
    assert (min(p.y for p in tool.points), max(p.y for p in tool.points)) == (-10, 10)
    assert tool.finger_holes == [poly.finger_holes[0].model_copy(update={"x": -10, "y": 0})]
    # Preserve the interior ring's winding, cyclic start and closing duplicate.
    assert tool.interior_rings == [[
        Point(x=-5, y=-5), Point(x=-10, y=-5),
        Point(x=-10, y=0), Point(x=-5, y=-5),
    ]]
    assert tool.source_image_transform == [0.5, 0, 0, 0.5, -30, -30]
    assert sessions.get("session-1").polygons == [poly]
