"""
MCP Tool: Upscale Image
"""

from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService
from ...results import image_result, progress_reporter, tool_errors


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def upscale_image(
        image: Annotated[str, Field(description="Image id from this server, or an http(s) URL")],
        ctx: Context,
        factor: Annotated[Literal[2, 4], Field(description="Enlargement factor")] = 2,
    ) -> ToolResult:
        """
        Enlarge an image 2x or 4x with an upscaling model (Real-ESRGAN)

        Adds sharpness rather than new content; use it for prints and wallpapers. Takes a few
        seconds. Returns a preview plus the URL and id of the full-resolution PNG.
        """
        with tool_errors():
            result = await flux.upscale(
                image, factor, on_progress=progress_reporter(ctx, flux.config.timeout_s)
            )
        return image_result(flux, result, "Upscaled")
