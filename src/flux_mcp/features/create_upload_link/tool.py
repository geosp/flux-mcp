"""
MCP Tool: Create Upload Link
"""

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def create_upload_link(
        max_files: Annotated[
            int, Field(description="How many images the link accepts", ge=1)
        ] = 5,
    ) -> ToolResult:
        """
        Get a short-lived link for uploading images to this server

        Use it when you need an image you cannot pass as a URL: one the user has, or one on
        your shell's filesystem. The link works for a few minutes and a limited number of
        images, and only for uploading. Give it to the user to open in a browser, or upload
        with curl. Then call check_upload with the returned upload_id to get the image ids.
        """
        if flux.uploads is None:
            raise ToolError("Upload links need the HTTP server. On this local server, pass "
                            "images as http(s) URLs instead.")
        limit = flux.config.upload_link_max_files
        if max_files > limit:
            raise ToolError(f"max_files can be at most {limit}")
        try:
            token, grant = flux.uploads.create(max_files)
        except ValueError as e:
            raise ToolError(str(e)) from e

        url = f"{flux.store.public_base_url}/upload/{token}"
        minutes = flux.uploads.ttl_s / 60
        text = "\n".join([
            f"Upload link (valid {minutes:g} min, up to {max_files} image(s), upload only):",
            url,
            "",
            "Browser: open the link and drop the images.",
            f"Shell: curl -F file=@a.png -F file=@b.png {url}",
            "",
            f"Then call check_upload with upload_id {grant.upload_id} "
            "(wait_s waits for the images to arrive).",
        ])
        return ToolResult(
            content=text,
            structured_content={
                "upload_url": url,
                "upload_id": grant.upload_id,
                "expires_in_s": flux.uploads.ttl_s,
                "max_files": max_files,
            },
        )
