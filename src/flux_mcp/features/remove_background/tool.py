"""
MCP Tool: Remove Background
"""

from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService
from ...results import image_result, progress_reporter, tool_errors


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def remove_background(
        image: Annotated[str, Field(description="Image id from this server, or an http(s) URL")],
        ctx: Context,
    ) -> ToolResult:
        """
        Cut out the main subject and make the background transparent (BiRefNet)

        Returns a transparent PNG (the preview shows it on white) plus its URL and id.
        """
        with tool_errors():
            result = await flux.remove_background(
                image, on_progress=progress_reporter(ctx, flux.config.timeout_s)
            )
        return image_result(flux, result, "Cut out")
