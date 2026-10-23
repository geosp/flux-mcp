"""
MCP Tool: Get Image
"""

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService
from ...results import stored_image_result, tool_errors


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def get_image(
        image: Annotated[
            str | None,
            Field(description="Image id or URL from this server; leave empty for the latest image"),
        ] = None,
    ) -> ToolResult:
        """
        Show a stored image again, with how it was made

        Returns the preview, URL and stored details (prompt, seed, upsampled caption, model).
        Without an id it returns the most recent image, which also recovers the result of a
        call that timed out on the client side.
        """
        with tool_errors():
            stored, metadata = await flux.get_image(image)
        return stored_image_result(flux, stored, metadata)
