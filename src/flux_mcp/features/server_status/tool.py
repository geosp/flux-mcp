"""
MCP Tool: Server Status
"""

from typing import Any

from fastmcp import FastMCP

from ...flux_service import FluxService
from ...results import tool_errors

GIB = 1024**3


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def server_status() -> dict[str, Any]:
        """
        Report whether ComfyUI is up, its queue, free memory and whether models are loaded
        """
        with tool_errors():
            stats = await flux.comfy.system_stats()
            queue = await flux.comfy.queue_length()
        devices = [
            {
                "name": d.get("name"),
                "vram_total_gib": round(d.get("vram_total", 0) / GIB, 1),
                "vram_free_gib": round(d.get("vram_free", 0) / GIB, 1),
            }
            for d in stats.get("devices", [])
        ]
        system = stats.get("system", {})
        return {
            "comfyui_version": system.get("comfyui_version"),
            "ram_free_gib": round(system.get("ram_free", 0) / GIB, 1),
            "queue": queue,
            "devices": devices,
            "memory": flux.memory_status(),
        }
