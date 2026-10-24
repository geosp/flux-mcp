"""
MCP Tool: Check Upload
"""

import asyncio
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService

POLL_INTERVAL_S = 1.0


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def check_upload(
        upload_id: Annotated[str, Field(description="upload_id from create_upload_link")],
        wait_s: Annotated[
            float,
            Field(description="Wait up to this many seconds for a first image to arrive",
                  ge=0, le=120),
        ] = 0,
    ) -> ToolResult:
        """
        List the images uploaded through a link from create_upload_link

        Returns each image's id, file name, size and URL. Pass the ids in `images` to
        edit_image, or as `image` to upscale_image, remove_background or get_image.
        """
        uploads = flux.uploads
        grant = uploads.get(upload_id) if uploads else None
        if grant is None:
            raise ToolError(f"Unknown or expired upload_id '{upload_id}'")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while not grant.files and not uploads.expired(grant) and loop.time() < deadline:
            await asyncio.sleep(min(POLL_INTERVAL_S, max(deadline - loop.time(), 0)))

        files = []
        for image_id in grant.files:
            stored = flux.store.get(image_id)
            if stored:
                metadata = flux.store.metadata(stored)
                files.append({"id": stored.id, "filename": metadata.get("filename"),
                              "size": metadata.get("size"),
                              "width": stored.width, "height": stored.height,
                              "url": stored.url})
        expired = uploads.expired(grant)
        if files:
            lines = [f"{len(files)} image(s) uploaded:",
                     *(f"- {f['filename']} ({f['width']}x{f['height']}), id {f['id']}"
                       for f in files)]
        else:
            lines = ["No images uploaded yet."]
        lines.append("The link has expired." if expired
                     else f"The link accepts {grant.remaining} more image(s).")
        return ToolResult(
            content="\n".join(lines),
            structured_content={"files": files, "remaining": 0 if expired else grant.remaining,
                                "expired": expired},
        )
