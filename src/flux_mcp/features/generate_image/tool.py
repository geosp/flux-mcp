"""
MCP Tool: Generate Image
"""

from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from ...boxes import MAX_BOXES, Box
from ...flux_service import FluxService
from ...results import image_result, progress_reporter, tool_errors

BOXES_HELP = (
    'Optional layout: where to place elements. Each box is {"bbox": [y_min, x_min, y_max, '
    'x_max] on a 0-1000 grid with (0, 0) at the top left, "desc": what goes there, "text": '
    "exact text to render (optional)}. Boxes guide placement and scale, not exact pixels."
)


def register_tool(mcp: FastMCP, flux: FluxService) -> None:
    @mcp.tool()
    async def generate_image(
        prompt: Annotated[str, Field(description="What to draw. A short request is fine.")],
        ctx: Context,
        width: Annotated[int, Field(description="Width in pixels (rounded to 16)")] = 1024,
        height: Annotated[int, Field(description="Height in pixels (rounded to 16)")] = 1024,
        count: Annotated[int, Field(description="Number of variations")] = 1,
        seed: Annotated[int | None, Field(description="Seed for a reproducible result")] = None,
        steps: Annotated[
            int | None, Field(description="Sampling steps; leave unset for the model default")
        ] = None,
        enhance_prompt: Annotated[
            bool,
            Field(description="Expand the prompt into a full scene caption first (recommended)"),
        ] = True,
        quality: Annotated[
            Literal["fast", "high"],
            Field(description="'high' uses the undistilled base model: more detail and variety, "
                              "about 3 times slower (~45 s). Use only when the user asks for it."),
        ] = "fast",
        boxes: Annotated[
            list[Box] | None,
            Field(description=BOXES_HELP, max_length=MAX_BOXES),
        ] = None,
    ) -> ToolResult:
        """
        Generate images from a text description with FLUX 3 Image Klein

        Returns a JPEG preview of each image plus the URL and id of the full-resolution PNG.
        Use the id with edit_image to refine a result.
        """
        with tool_errors():
            result = await flux.generate(
                prompt,
                width=width,
                height=height,
                count=count,
                seed=seed,
                steps=steps,
                enhance_prompt=enhance_prompt,
                quality=quality,
                boxes=boxes,
                on_progress=progress_reporter(ctx, flux.config.timeout_s),
            )
        return image_result(flux, result, "Generated")
