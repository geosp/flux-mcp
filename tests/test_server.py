import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from flux_mcp.auth import BearerTokenVerifier
from flux_mcp.server import FluxMCPServer
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
        assert names == {"generate_image", "edit_image", "upscale_image", "remove_background",
                         "list_models", "server_status"}
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
        edited = await client.call_tool("edit_image", {"prompt": "make it blue", "images": [image_id]})
        combined = await client.call_tool("edit_image", {
            "prompt": "the fox from the first image in the scene from the second image",
            "images": [image_id, edited.structured_content["images"][0]["id"]],
        })
        upscaled = await client.call_tool("upscale_image", {"image": image_id, "factor": 4})
        cutout = await client.call_tool("remove_background", {"image": image_id})
    assert edited.structured_content["images"][0]["id"] != image_id
    assert upscaled.structured_content["width"] == 4096 and upscaled.structured_content["seed"] is None
    assert cutout.content[0].text.startswith("Cut out 1 image(s) at 1024x1024.")
    assert len(combined.structured_content["images"]) == 1


async def test_bad_input_is_a_tool_error(server):
    async with Client(server.create_mcp_app()) as client:
        with pytest.raises(ToolError, match="width must be"):
            await client.call_tool("generate_image", {"prompt": "x", "width": 99999})


async def test_status_and_models(server):
    async with Client(server.create_mcp_app()) as client:
        status = (await client.call_tool("server_status", {})).structured_content
        models = (await client.call_tool("list_models", {})).structured_content
    assert status["devices"][0]["vram_free_gib"] == 42.0
    assert models["workflows"] == ["edit", "edit_hq", "remove_background", "t2i", "t2i_hq", "upscale"]


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


async def upload_client(app_config, transport, **kwargs):
    server = make_server(app_config, transport, mode="http", token="secret", **kwargs)
    app = server.create_asgi_app()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://t")


async def test_upload_stores_image(app_config, transport):
    server, http = await upload_client(app_config, transport)
    async with http:
        response = await http.post("/upload", headers={"Authorization": "Bearer secret"},
                                   files={"file": ("photo.jpg", png(40, 30), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert (data["width"], data["height"]) == (40, 30)
    assert data["url"] == f"http://flux.test:8300/images/{data['id']}.png"
    assert server.service.flux.store.path_for(data["id"]) is not None


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"},
                                     {"Authorization": "Basic secret"}])
async def test_upload_requires_token(app_config, transport, headers):
    _, http = await upload_client(app_config, transport)
    async with http:
        response = await http.post("/upload", headers=headers, files={"file": ("a.png", png(8, 8))})
    assert response.status_code == 401


async def test_upload_without_auth_when_disabled(app_config, transport):
    _, http = await upload_client(app_config, transport, auth=False)
    async with http:
        response = await http.post("/upload", files={"file": ("a.png", png(8, 8))})
    assert response.status_code == 200


async def test_upload_rejects_bad_files(app_config, transport):
    app_config.flux.max_input_bytes = 2000
    _, http = await upload_client(app_config, transport)
    auth = {"Authorization": "Bearer secret"}
    async with http:
        text = await http.post("/upload", headers=auth, files={"file": ("a.txt", b"hello")})
        missing = await http.post("/upload", headers=auth, files={"other": ("a.png", png(8, 8))})
        big = await http.post("/upload", headers=auth,
                              files={"file": ("a.png", png(8, 8) + b"\0" * 5000)})
    assert text.status_code == 400 and "not a readable image" in text.json()["error"]
    assert missing.status_code == 400
    assert big.status_code == 413


async def test_upload_page(app_config, transport):
    _, http = await upload_client(app_config, transport)
    async with http:
        page = await http.get("/upload")
    assert page.status_code == 200 and "Upload a reference image" in page.text


async def test_generate_with_boxes(server, fake_comfy):
    async with Client(server.create_mcp_app()) as client:
        await client.call_tool("generate_image", {"prompt": "a fox and a sign", "boxes": [
            {"bbox": [100, 50, 900, 450], "desc": "a red fox"},
            {"bbox": [600, 550, 950, 950], "desc": "a wooden sign", "text": "Welcome"},
        ]})
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "x", "boxes": [
                {"bbox": [900, 0, 100, 500], "desc": "upside down"}]})
    request = fake_comfy.prompts["p1"]["21"]["inputs"]["prompt"]
    assert '"desc": "a wooden sign \\"Welcome\\""' in request
