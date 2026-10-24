import contextlib

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
                         "get_image", "unload_models", "list_models", "server_status",
                         "create_upload_link", "check_upload"}
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


@contextlib.asynccontextmanager
async def http_server(app_config, transport, **config):
    """An HTTP-mode server with an MCP client and a plain HTTP client on its ASGI app"""
    for key, value in config.items():
        setattr(app_config.flux, key, value)
    server = make_server(app_config, transport, mode="http", token="secret")
    app = server.create_asgi_app()
    server.service.flux.comfy.poll_interval = 0
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://flux.test:8300"
    ) as http:
        yield server, Client(server._mcp_app), http


async def test_master_token_upload_route_is_gone(app_config, transport):
    async with http_server(app_config, transport) as (_, _, http):
        assert (await http.post("/mcp", json={})).status_code == 401
        response = await http.post("/upload", files={"file": ("a.png", png(8, 8))},
                                   headers={"Authorization": "Bearer secret"})
        assert response.status_code in (404, 405)
        assert (await http.get("/upload")).status_code in (404, 405)


async def test_upload_link_flow(app_config, transport):
    async with http_server(app_config, transport) as (server, client, http):
        async with client:
            link = await client.call_tool("create_upload_link", {"max_files": 2})
            url = link.structured_content["upload_url"]
            upload_id = link.structured_content["upload_id"]
            assert url.startswith("http://flux.test:8300/upload/") and upload_id not in url
            assert "curl -F file=@" in link.content[0].text

            empty = await client.call_tool("check_upload", {"upload_id": upload_id})
            assert empty.structured_content == {"files": [], "remaining": 2, "expired": False}

            page = await http.get(url)
            assert page.status_code == 200 and "Upload reference images" in page.text
            assert 'name="referrer" content="no-referrer"' in page.text
            assert "Authorization" not in page.text
            uploaded = await http.post(url, files=[("file", ("../../fox.jpg", png(40, 30))),
                                                   ("file", ("bg.png", png(64, 48, "blue")))])
            assert uploaded.status_code == 200
            assert uploaded.json() == {"files": [{"filename": "fox.jpg", "size": len(png(40, 30))},
                                                 {"filename": "bg.png",
                                                  "size": len(png(64, 48, "blue"))}],
                                       "remaining": 0}

            # used up: the link no longer works, even for its form
            assert (await http.post(url, files={"file": ("x.png", png(8, 8))})).status_code == 404
            assert (await http.get(url)).status_code == 404

            checked = await client.call_tool("check_upload", {"upload_id": upload_id, "wait_s": 5})
            files = {f["filename"]: f for f in checked.structured_content["files"]}
            assert set(files) == {"fox.jpg", "bg.png"}
            assert (files["fox.jpg"]["width"], files["fox.jpg"]["height"]) == (40, 30)
            assert checked.structured_content["remaining"] == 0

            edited = await client.call_tool("edit_image", {
                "prompt": "the fox from the first image on the second",
                "images": [files["fox.jpg"]["id"], files["bg.png"]["id"]]})
            assert len(edited.structured_content["images"]) == 1
            shown = await client.call_tool("get_image", {"image": files["fox.jpg"]["id"]})
            assert shown.structured_content["metadata"]["upload_id"] == upload_id

            image = await http.get(files["bg.png"]["url"].removeprefix("http://flux.test:8300"))
            assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert (await http.post("/mcp", json={})).status_code == 401


async def test_upload_link_limits(app_config, transport):
    async with http_server(app_config, transport, upload_link_max_files=3,
                           max_input_bytes=2000) as (server, client, http):
        store = server.service.flux.store
        async with client:
            with pytest.raises(ToolError, match="at most 3"):
                await client.call_tool("create_upload_link", {"max_files": 4})
            link = await client.call_tool("create_upload_link", {"max_files": 2})
            url = link.structured_content["upload_url"]

            too_many = await http.post(url, files=[("file", ("a.png", png(8, 8)))] * 3)
            assert too_many.status_code == 400
            # one bad file rejects the whole batch
            mixed = await http.post(url, files=[("file", ("a.png", png(8, 8))),
                                                ("file", ("notes.txt", b"hello"))])
            assert mixed.status_code == 400 and "not a readable image" in mixed.json()["error"]
            assert store.latest() is None
            empty = await http.post(url, files={"file": ("a.png", b"")})
            missing = await http.post(url, files={"other": ("a.png", png(8, 8))})
            big = await http.post(url, files={"file": ("a.png", png(8, 8) + b"\0" * 5000)})
            assert (empty.status_code, missing.status_code, big.status_code) == (400, 400, 413)
            assert (await http.post(url + "x", files={"file": ("a.png", png(8, 8))})).status_code == 404

            server.service.flux.uploads.clock = lambda: float("inf")
            assert (await http.post(url, files={"file": ("a.png", png(8, 8))})).status_code == 404
            gone = await http.get(url)
            assert gone.status_code == 404 and "Ask the assistant for a new one" in gone.text


async def test_upload_link_needs_http(server):
    async with Client(server.create_mcp_app()) as client:
        with pytest.raises(ToolError, match="need the HTTP server"):
            await client.call_tool("create_upload_link", {})
        with pytest.raises(ToolError, match="Unknown or expired"):
            await client.call_tool("check_upload", {"upload_id": "0" * 32})


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


async def test_caption_get_image_and_unload(server, fake_comfy):
    async with Client(server.create_mcp_app()) as client:
        generated = await client.call_tool("generate_image", {"prompt": "a red fox", "seed": 9})
        image_id = generated.structured_content["images"][0]["id"]
        latest = await client.call_tool("get_image", {})
        by_id = await client.call_tool("get_image", {"image": image_id})
        with pytest.raises(ToolError, match="No image"):
            await client.call_tool("get_image", {"image": "0" * 32})
        unloaded = await client.call_tool("unload_models", {})
        status = (await client.call_tool("server_status", {})).structured_content
    assert generated.structured_content["caption"] == "A detailed upsampled caption."
    assert "Caption the model received" in generated.content[0].text
    assert latest.structured_content["id"] == by_id.structured_content["id"] == image_id
    metadata = by_id.structured_content["metadata"]
    assert metadata["prompt"] == "a red fox" and metadata["seed"] == 9
    assert metadata["caption"] == "A detailed upsampled caption."
    assert metadata["model"] == "flux3-image-klein-nvfp4.safetensors" and metadata["steps"] == 16
    assert by_id.content[1].type == "image"
    assert unloaded.structured_content["models_loaded"] is False and fake_comfy.freed == 1
    assert status["memory"]["models_loaded"] is False
