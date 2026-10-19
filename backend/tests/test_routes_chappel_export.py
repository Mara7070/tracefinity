"""Authenticated persisted-Tool manufacturing downloads and response lifetime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.schemas import Point, Tool
from app.services.tool_store import ToolStore


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.onnx_check.is_onnx_available", lambda: False)
    from app import auth
    from app.api import routes

    monkeypatch.setattr(routes.settings, "auth_mode", "native")
    monkeypatch.setattr(routes.settings, "storage_path", tmp_path)
    monkeypatch.setattr(auth, "resolve_account", lambda request: (
        SimpleNamespace(storage_namespace="owner")
        if request.cookies.get("tracefinity_auth") == "valid" else None
    ))
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "outputs").mkdir()
    store = ToolStore(owner)
    tool = Tool(id="rectangle-tool", name="Rectangle", points=[
        Point(x=x, y=y) for x, y in ((0, 0), (60, 0), (60, 20), (0, 20))
    ])
    store.set(tool.id, tool)
    stores = Mock(side_effect=lambda user_id: (
        (None, store, None) if user_id == "owner" else (None, ToolStore(tmp_path), None)
    ))
    monkeypatch.setattr(routes, "get_stores", stores)
    before = tool.model_dump()
    disk_before = store.file_path.read_bytes()
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    with TestClient(app) as client:
        client.cookies.set("tracefinity_auth", "valid")
        yield SimpleNamespace(client=client, routes=routes, tool=tool, store=store,
                              stores=stores, outputs=owner / "outputs")
    assert tool.model_dump() == before
    assert store.get(tool.id) is tool
    assert store.file_path.read_bytes() == disk_before
    assert list((owner / "outputs").iterdir()) == []


def export(api, **overrides):
    params = dict(max_cells_x=3, max_cells_y=4, depth=12.5, format="step")
    params.update(overrides)
    return api.client.post(f"/api/tools/{api.tool.id}/export-chappel", params=params)


@pytest.mark.parametrize("format,media", [("step", "model/step"), ("stl", "application/sla")])
def test_authenticated_delegation_and_download(api, monkeypatch, format, media):
    def write(tool, max_x, max_y, depth, path, inset, clearance):
        assert path.parent == api.outputs
        assert path.suffix == f".{format}"
        path.write_bytes(b"manufacturing artifact")
        return path
    service = Mock(side_effect=write)
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", service)
    response = export(api, format=format, inset=0.5, clearance=0.25)
    assert response.status_code == 200
    assert response.content == b"manufacturing artifact"
    assert response.headers["content-type"] == media
    assert response.headers["content-disposition"] == (
        f'attachment; filename="tracefinity-rectangl-chappel.{format}"'
    )
    service.assert_called_once_with(api.tool, 3, 4, 12.5, service.call_args.args[4], 0.5, 0.25)
    assert service.call_args.args[0] is api.store.get(api.tool.id)
    api.stores.assert_called_once_with("owner")
    assert not service.call_args.args[4].exists()


def test_auth_required(api, monkeypatch):
    service = Mock()
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", service)
    api.client.cookies.clear()
    assert export(api).status_code == 401
    service.assert_not_called()
    api.stores.assert_not_called()


def test_optional_defaults(api, monkeypatch):
    def write(*args):
        args[4].write_bytes(b"artifact")
        return args[4]
    service = Mock(side_effect=write)
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", service)
    assert export(api).status_code == 200
    assert service.call_args.args[5:] == (0.0, 0.0)


def test_missing_and_other_user_tool(api, monkeypatch):
    service = Mock()
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", service)
    response = api.client.post("/api/tools/missing/export-chappel", params=dict(
        max_cells_x=3, max_cells_y=3, depth=12, format="step"))
    assert response.status_code == 404
    assert response.json() == {"detail": "tool not found"}
    from app import auth
    monkeypatch.setattr(auth, "resolve_account", lambda request: SimpleNamespace(storage_namespace="other"))
    assert export(api).status_code == 404
    service.assert_not_called()


@pytest.mark.parametrize("params", [dict(format="3mf"), dict(format="STEP"),
                                      dict(max_cells_x="bad"), dict(max_cells_y=1.5),
                                      dict(depth="bad")])
def test_query_validation_before_export(api, monkeypatch, params):
    service = Mock()
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", service)
    assert export(api, **params).status_code == 422
    service.assert_not_called()


@pytest.mark.parametrize("message", ["tool does not fit within the supplied Chappel cell bounds",
                                     "depth must be finite and positive"])
def test_service_value_error_preserves_detail_and_cleans_partial_file(api, monkeypatch, message):
    def fail(*args):
        args[4].write_bytes(b"partial")
        raise ValueError(message)
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", fail)
    response = export(api)
    assert response.status_code == 400
    assert response.json() == {"detail": message}
    assert list(api.outputs.iterdir()) == []


def test_unexpected_service_error_propagates_and_cleans(api, monkeypatch):
    failure = OSError("export failed")
    def fail(*args):
        args[4].write_bytes(b"partial")
        raise failure
    monkeypatch.setattr(api.routes, "export_minimal_chappel_tool", fail)
    with pytest.raises(OSError) as caught:
        export(api)
    assert caught.value is failure
    assert list(api.outputs.iterdir()) == []


@pytest.mark.parametrize("format", ["step", "stl"])
def test_real_export_download(api, format):
    response = export(api, format=format, depth=12, inset=2)
    assert response.status_code == 200
    if format == "step":
        assert response.content.startswith(b"ISO-10303-21;")
    else:
        import io
        import trimesh
        mesh = trimesh.load(io.BytesIO(response.content), file_type="stl", force="mesh")
        assert mesh.is_watertight and mesh.is_volume
    assert list(api.outputs.iterdir()) == []


@pytest.mark.parametrize("params", [dict(max_cells_x=0), dict(inset=-1),
                                      dict(clearance=-1), dict(depth=24),
                                      dict(max_cells_x=1, max_cells_y=1)])
def test_real_service_invalid_input(api, params):
    response = export(api, **params)
    assert response.status_code == 400
    assert response.json()["detail"]
