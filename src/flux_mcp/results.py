"""
Shared helpers for the image tools: progress reporting, error mapping and
turning a GenerationResult into MCP content
"""

from contextlib import contextmanager
from typing import Iterator

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image

from .comfy_client import ComfyError, ProgressCallback
from .flux_service import FluxService, GenerationResult


def progress_reporter(ctx: Context, timeout: float) -> ProgressCallback:
    async def report(elapsed: float, status: str) -> None:
        await ctx.report_progress(elapsed, total=timeout, message=status)
    return report


@contextmanager
def tool_errors() -> Iterator[None]:
    """Report bad input and backend failures to the client as tool errors"""
    try:
        yield
    except (ValueError, ComfyError) as e:
        raise ToolError(str(e)) from e
    except TimeoutError as e:
        raise ToolError(f"{e}. ComfyUI may be busy; try again or lower the size.") from e


def image_result(flux: FluxService, result: GenerationResult, verb: str) -> ToolResult:
    lines = [
        f"{verb} {len(result.images)} image(s) at {result.width}x{result.height}, "
        f"seed {result.seed}. Full-resolution PNG:",
        *(f"- {img.url} (id {img.id})" for img in result.images),
    ]
    previews = [Image(data=flux.store.preview(img), format="jpeg") for img in result.images]
    return ToolResult(
        content=["\n".join(lines), *previews],
        structured_content={
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "images": [{"id": img.id, "url": img.url} for img in result.images],
        },
    )
