"""
MCP Tool: Edit Image
"""

from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from ...flux_service import FluxService
from ...results import image_result, progress_reporter, tool_errors


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def edit_image(
        prompt: Annotated[str, Field(description="The change to make, or the result you want")],
        image: Annotated[
            str, Field(description="Image id from generate_image/edit_image, or an http(s) URL")
        ],
        ctx: Context,
        width: Annotated[
            int | None, Field(description="Output width; default keeps the input's aspect ratio")
        ] = None,
        height: Annotated[
            int | None, Field(description="Output height; default keeps the input's aspect ratio")
        ] = None,
        count: Annotated[int, Field(description="Number of variations")] = 1,
        seed: Annotated[int | None, Field(description="Seed for a reproducible result")] = None,
        steps: Annotated[
            int | None, Field(description="Sampling steps; leave unset for the model default")
        ] = None,
        enhance_prompt: Annotated[
            bool,
            Field(description="Expand the request into a full edit instruction first (recommended)"),
        ] = True,
    ) -> ToolResult:
        """
        Edit an image with FLUX 3 Image Klein

        Describe the change ("make the sky stormy", "add a red scarf to the dog"). Returns a JPEG
        preview of each result plus the URL and id of the full-resolution PNG.
        """
        with tool_errors():
            result = await flux.edit(
                prompt,
                image,
                width=width,
                height=height,
                count=count,
                seed=seed,
                steps=steps,
                enhance_prompt=enhance_prompt,
                on_progress=progress_reporter(ctx, flux.config.timeout_s),
            )
        return image_result(flux, result, "Edited")
