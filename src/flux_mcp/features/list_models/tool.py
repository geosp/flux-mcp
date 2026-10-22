"""
MCP Tool: List Models
"""

from typing import Any

from fastmcp import FastMCP

from ...flux_service import FluxService
from ...results import tool_errors

FOLDERS = ("diffusion_models", "text_encoders", "vae")


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def list_models() -> dict[str, Any]:
        """
        List the workflows this server runs and the model files ComfyUI can see
        """
        with tool_errors():
            files = {folder: await flux.comfy.models(folder) for folder in FOLDERS}
        return {"workflows": sorted(flux.workflows), "model_files": files}
