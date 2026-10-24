"""
MCP service for flux-mcp

Builds the ComfyUI client, workflows and image store, registers the tools
from features/*/tool.py, serves stored images at /images/<id>.png and, over
HTTP, accepts uploads through short-lived links at /upload/<token> (POST, or
GET for a browser form).
"""

import importlib
import io
import logging
import pkgutil
from pathlib import Path

import httpx
from fastmcp import FastMCP
from mcp_service_kit import BaseService
from PIL import Image, UnidentifiedImageError
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from .comfy_client import ComfyClient
from .config import AppConfig
from .flux_service import FluxService
from .images import ImageStore
from .uploads import UploadLinks, safe_filename
from .workflows import load_workflows

logger = logging.getLogger(__name__)

LINK_GONE_PAGE = (
    "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
    "<meta name=referrer content=no-referrer>"
    "<title>Upload link expired</title><body style='font:15px system-ui;max-width:640px;"
    "margin:2rem auto;padding:0 16px'><h1>This upload link is invalid or expired</h1>"
    "<p>Links work for a few minutes and a limited number of images. "
    "Ask the assistant for a new one.</p>"
)


class FluxMCPService(BaseService):
    """
    MCP service wrapper for FLUX image generation

    Each package in features/ with a tool.py exposing register_tool(mcp, flux)
    contributes one tool.
    """

    def __init__(self, config: AppConfig, transport: httpx.AsyncBaseTransport | None = None):
        """
        Args:
            config: Application configuration
            transport: HTTP transport for ComfyUI and image downloads (tests pass a mock)
        """
        self.config = config
        self.transport = transport
        self.flux: FluxService | None = None

    def initialize(self) -> None:
        flux_config = self.config.flux
        workflows = load_workflows(flux_config.workflows_dir)
        store = ImageStore(
            flux_config.output_dir, flux_config.public_base_url, flux_config.preview_max_side
        )
        store.ensure_directory()

        # Live for the whole process; cleanup() runs after the event loop has stopped,
        # so they are not closed explicitly.
        comfy_http = httpx.AsyncClient(
            base_url=flux_config.comfyui_url, timeout=60.0, transport=self.transport
        )
        downloader = httpx.AsyncClient(timeout=60.0, transport=self.transport)
        self.flux = FluxService(flux_config, ComfyClient(comfy_http), workflows, store, downloader)
        if self.config.server.transport == "http":  # links need the HTTP routes
            self.flux.uploads = UploadLinks(flux_config.upload_link_ttl_s,
                                            flux_config.upload_link_max_active)
        logger.info(
            f"Using ComfyUI at {flux_config.comfyui_url}; workflows: {', '.join(workflows)}"
        )

    def get_service_name(self) -> str:
        return "flux"

    def register_mcp_tools(self, mcp: FastMCP) -> None:
        features_package = f"{__package__}.features"
        features_path = Path(__file__).parent / "features"

        for _, feature_name, is_pkg in pkgutil.iter_modules([str(features_path)]):
            if not is_pkg:
                continue
            tool_module = importlib.import_module(f"{features_package}.{feature_name}.tool")
            tool_module.register_tool(mcp, self.flux)
            logger.info(f"Registered tool from feature: {feature_name}")

        store = self.flux.store

        @mcp.custom_route("/images/{name}", methods=["GET"], include_in_schema=False)
        async def get_image(request: Request) -> Response:
            path = store.path_for(request.path_params["name"].removesuffix(".png"))
            if path is None:
                return PlainTextResponse("Not found", status_code=404)
            return FileResponse(path, media_type="image/png")

        if self.flux.uploads is not None:
            self._register_upload_routes(mcp)

    def _register_upload_routes(self, mcp: FastMCP) -> None:
        flux = self.flux
        max_bytes = self.config.flux.max_input_bytes
        upload_page = (Path(__file__).parent / "upload_page.html").read_text()

        @mcp.custom_route("/upload/{token}", methods=["GET"], include_in_schema=False)
        async def upload_form(request: Request) -> Response:
            if flux.uploads.for_token(request.path_params["token"]) is None:
                return HTMLResponse(LINK_GONE_PAGE, status_code=404)
            return HTMLResponse(upload_page)

        @mcp.custom_route("/upload/{token}", methods=["POST"], include_in_schema=False)
        async def upload(request: Request) -> Response:
            # The link itself is the credential: no Authorization header. FastMCP's auth
            # covers /mcp only, so this route must not accept anything broader.
            grant = flux.uploads.for_token(request.path_params["token"])
            if grant is None:
                return JSONResponse({"error": "this upload link is invalid, expired or used up"},
                                    status_code=404)
            too_large = JSONResponse({"error": f"each file must be at most {max_bytes} bytes"},
                                     status_code=413)
            if int(request.headers.get("content-length") or 0) > \
                    grant.remaining * max_bytes + 64 * 1024:
                return too_large
            files: list[tuple[str, bytes]] = []
            try:
                async with request.form(max_files=grant.remaining, max_fields=0) as form:
                    for file in form.getlist("file"):
                        if isinstance(file, str):
                            continue
                        if file.size is not None and file.size > max_bytes:
                            return too_large
                        files.append((safe_filename(file.filename or ""), await file.read()))
            except MultiPartException as e:
                return JSONResponse({"error": e.message}, status_code=400)
            if not files:
                return JSONResponse({"error": "send the images as multipart field 'file'"},
                                    status_code=400)
            if any(not data for _, data in files):
                return JSONResponse({"error": "empty files are not accepted"}, status_code=400)
            if any(len(data) > max_bytes for _, data in files):
                return too_large
            # Check every file before storing any, so a bad one leaves nothing half-stored
            for filename, data in files:
                if not _readable_image(data):
                    return JSONResponse({"error": f"{filename} is not a readable image"},
                                        status_code=400)

            stored = []
            for filename, data in files:
                image = flux.store.save(data, {"tool": "upload", "upload_id": grant.upload_id,
                                               "filename": filename, "size": len(data)})
                flux.uploads.record(grant, image.id)
                stored.append({"filename": filename, "size": len(data)})
                logger.info(f"Stored upload {image.id} ({filename}, {image.width}x{image.height}) "
                            f"for {grant.upload_id}")
            return JSONResponse({"files": stored, "remaining": grant.remaining})


def _readable_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
        return True
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        return False
