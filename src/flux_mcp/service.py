"""
MCP service for flux-mcp

Builds the ComfyUI client, workflows and image store, registers the tools
from features/*/tool.py, serves stored images at /images/<id>.png and accepts
uploads at /upload (POST, or GET for a browser form).
"""

import importlib
import logging
import pkgutil
from pathlib import Path

import httpx
from fastmcp import FastMCP
from mcp_service_kit import BaseService
from PIL import Image, UnidentifiedImageError
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from .auth import bearer_token, token_matches
from .comfy_client import ComfyClient
from .config import AppConfig
from .flux_service import FluxService
from .images import ImageStore
from .workflows import load_workflows

logger = logging.getLogger(__name__)


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
        max_bytes = self.config.flux.max_input_bytes
        upload_page = (Path(__file__).parent / "upload_page.html").read_text()

        @mcp.custom_route("/images/{name}", methods=["GET"], include_in_schema=False)
        async def get_image(request: Request) -> Response:
            path = store.path_for(request.path_params["name"].removesuffix(".png"))
            if path is None:
                return PlainTextResponse("Not found", status_code=404)
            return FileResponse(path, media_type="image/png")

        @mcp.custom_route("/upload", methods=["GET"], include_in_schema=False)
        async def upload_form(request: Request) -> Response:
            return HTMLResponse(upload_page)

        @mcp.custom_route("/upload", methods=["POST"], include_in_schema=False)
        async def upload(request: Request) -> Response:
            # FastMCP's auth covers /mcp only, so custom routes check the token themselves
            if not self._authorized(request):
                return JSONResponse({"error": "missing or invalid bearer token"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            too_large = JSONResponse({"error": f"file is larger than {max_bytes} bytes"},
                                     status_code=413)
            if int(request.headers.get("content-length") or 0) > max_bytes + 64 * 1024:
                return too_large
            async with request.form(max_files=1, max_fields=0) as form:
                file = form.get("file")
                if file is None or isinstance(file, str):
                    return JSONResponse({"error": "send the image as multipart field 'file'"},
                                        status_code=400)
                if file.size is not None and file.size > max_bytes:
                    return too_large
                data = await file.read()
            try:
                stored = store.save(data)
            except (UnidentifiedImageError, Image.DecompressionBombError):
                return JSONResponse({"error": "the file is not a readable image"}, status_code=400)
            logger.info(f"Stored upload {stored.id} ({stored.width}x{stored.height})")
            return JSONResponse({"id": stored.id, "url": stored.url,
                                 "width": stored.width, "height": stored.height})

    def _authorized(self, request: Request) -> bool:
        server, token = self.config.server, self.config.flux.mcp_token
        if not (server.transport == "http" and server.auth_enabled):
            return True
        given = bearer_token(request.headers.get("authorization"))
        return token is not None and given is not None and token_matches(token, given)
