import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from flux_mcp.server import BearerTokenVerifier, FluxMCPServer
from flux_mcp.service import FluxMCPService

from .conftest import png


def make_server(app_config, transport, mode="stdio", auth=True, token=None):
    app_config.server.transport = mode
    app_config.server.auth_enabled = auth
    app_config.flux.mcp_token = SecretStr(token) if token else None
    service = FluxMCPService(app_config, transport)
    return FluxMCPServer(app_config.server, service)


@pytest.fixture
def server(app_config, transport):
    server = make_server(app_config, transport)
    server.service.initialize()
    server.service.flux.comfy.poll_interval = 0
    return server


async def test_tools_and_instructions(server):
    async with Client(server.create_mcp_app()) as client:
        names = {t.name for t in await client.list_tools()}
        assert names == {"generate_image", "edit_image", "list_models", "server_status"}
        assert "generate_image" in client.instructions


async def test_generate_image_returns_preview_and_url(server):
    async with Client(server.create_mcp_app()) as client:
        result = await client.call_tool("generate_image", {"prompt": "a red fox", "seed": 3})
    text, preview = result.content
    assert "seed 3" in text.text and "http://flux.test:8300/images/" in text.text
    assert preview.type == "image" and preview.mime_type == "image/jpeg"
    assert result.structured_content["images"][0]["url"].endswith(".png")


async def test_edit_image_chains_from_generate(server):
    async with Client(server.create_mcp_app()) as client:
        first = await client.call_tool("generate_image", {"prompt": "a red fox"})
        image_id = first.structured_content["images"][0]["id"]
        edited = await client.call_tool("edit_image", {"prompt": "make it blue", "image": image_id})
    assert edited.structured_content["images"][0]["id"] != image_id


async def test_bad_input_is_a_tool_error(server):
    async with Client(server.create_mcp_app()) as client:
        with pytest.raises(ToolError, match="width must be"):
            await client.call_tool("generate_image", {"prompt": "x", "width": 99999})


async def test_status_and_models(server):
    async with Client(server.create_mcp_app()) as client:
        status = (await client.call_tool("server_status", {})).structured_content
        models = (await client.call_tool("list_models", {})).structured_content
    assert status["devices"][0]["vram_free_gib"] == 42.0
    assert models["workflows"] == ["edit", "t2i"]


async def test_image_route(app_config, transport):
    server = make_server(app_config, transport, mode="http", token="secret")
    app = server.create_asgi_app()

    stored = server.service.flux.store.save(png(16, 16))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://t") as http:
        ok = await http.get(f"/images/{stored.id}.png")
        missing = await http.get(f"/images/{'0' * 32}.png")
        traversal = await http.get("/images/..%2Fsecret.png")
        unauth = await http.post("/mcp", json={})
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert missing.status_code == 404 and traversal.status_code == 404
    assert unauth.status_code == 401


async def test_bearer_verifier():
    verifier = BearerTokenVerifier(SecretStr("secret"))
    assert await verifier.verify_token("secret") is not None
    assert await verifier.verify_token("wrong") is None


def test_http_without_token_refuses_to_start(app_config, transport):
    with pytest.raises(ValueError, match="FLUX_MCP_TOKEN"):
        make_server(app_config, transport, mode="http").validate_configuration()
    make_server(app_config, transport, mode="http", auth=False).validate_configuration()
    make_server(app_config, transport, mode="stdio").validate_configuration()
