"""
MCP Tool: Unload Models
"""

from typing import Any

from fastmcp import FastMCP

from ...flux_service import FluxService
from ...results import tool_errors


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def unload_models() -> dict[str, Any]:
        """
        Free the GPU memory held by the image models now

        The models reload automatically on the next request, which then takes longer. The
        server also does this by itself after a period without requests.
        """
        with tool_errors():
            await flux.unload_models()
        return {"unloaded": True, **flux.memory_status()}
