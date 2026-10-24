"""
Image generation and editing on top of ComfyUI

Business logic shared by the MCP tools: validates requests, fills workflows,
runs them and stores the results. No MCP imports.
"""

import asyncio
import io
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from typing import Any, Literal

import httpx
from PIL import Image, UnidentifiedImageError

from .boxes import Box, layout_rows
from .comfy_client import ComfyClient, ComfyError, ProgressCallback
from .config import FluxConfig
from .images import ID_PATTERN, ImageStore, StoredImage
from .uploads import UploadLinks
from .workflows import Workflow

logger = logging.getLogger(__name__)

# Latent sizes must be multiples of this
SIZE_STEP = 16
MAX_SEED = 2**53 - 1
# Default edit output size, in pixels, when the caller gives none
EDIT_DEFAULT_PIXELS = 1024 * 1024

Quality = Literal["fast", "high"]
# Workflow name suffix per quality
QUALITY_SUFFIX = {"fast": "", "high": "_hq"}


@dataclass(frozen=True)
class GenerationResult:
    images: list[StoredImage]
    seed: int | None
    width: int
    height: int
    # The full caption the prompt upsampler wrote, when it ran
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FluxService:
    def __init__(
        self,
        config: FluxConfig,
        comfy: ComfyClient,
        workflows: dict[str, Workflow],
        store: ImageStore,
        downloader: httpx.AsyncClient,
    ):
        """
        Args:
            config: Limits and timeouts
            comfy: ComfyUI client
            workflows: Loaded workflows; needs "t2i" and "edit"
            store: Where results are saved
            downloader: HTTP client used to fetch edit_image inputs given as URLs
        """
        self.config = config
        self.comfy = comfy
        self.workflows = workflows
        self.store = store
        self.downloader = downloader
        # Upload links need the HTTP routes, so the service sets this only over HTTP
        self.uploads: UploadLinks | None = None
        # Idle unloading: models count as loaded after a job until unload_models() runs
        self._active_jobs = 0
        self._models_loaded = False
        self._unload_task: asyncio.Task | None = None
        self._unload_at: float | None = None

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        count: int = 1,
        seed: int | None = None,
        steps: int | None = None,
        enhance_prompt: bool = True,
        quality: Quality = "fast",
        boxes: list[Box] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Text to image"""
        self._check_prompt(prompt)
        self._check_count(count)
        width, height = self._check_size(width, height)
        seed = self._seed(seed)
        graph = self._workflow("t2i", quality).fill(
            prompt,
            enhance_prompt,
            width=width,
            height=height,
            batch_size=count,
            seed=seed,
            steps=self._check_steps(steps),
            boxes=layout_rows(boxes) if boxes else None,
        )
        return await self._run(graph, on_progress, {
            "tool": "generate_image", "prompt": prompt, "seed": seed, "quality": quality,
            "width": width, "height": height, "boxes": layout_rows(boxes) if boxes else None,
        })

    async def edit(
        self,
        prompt: str,
        images: list[str],
        width: int | None = None,
        height: int | None = None,
        count: int = 1,
        seed: int | None = None,
        steps: int | None = None,
        enhance_prompt: bool = True,
        quality: Quality = "fast",
        boxes: list[Box] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """
        Edit an image, or compose a new one from several reference images

        Args:
            images: Image ids from this server or http(s) URLs, in the order the
                prompt refers to them ("the first image", "the second image", ...)
            width, height: Output size; by default the first image's aspect ratio
                at about one megapixel
            boxes: Regions of the first image the prompt refers to as obj_1, obj_2, ...
        """
        self._check_prompt(prompt)
        self._check_count(count)
        workflow = self._workflow("edit", quality)
        limit = workflow.references.max
        if not 1 <= len(images) <= limit:
            raise ValueError(f"images must hold 1 to {limit} image ids or URLs")
        data = [await self.load_input(image) for image in images]
        if width is None or height is None:
            width, height = _fit_pixels(*_image_size(data[0]), EDIT_DEFAULT_PIXELS)
        width, height = self._check_size(width, height)
        names = [
            await self.comfy.upload(d, f"flux-mcp-{random.getrandbits(64):016x}.png") for d in data
        ]
        seed = self._seed(seed)
        graph = workflow.fill(
            prompt,
            enhance_prompt,
            images=names,
            width=width,
            height=height,
            batch_size=count,
            seed=seed,
            steps=self._check_steps(steps),
            boxes=layout_rows(boxes) if boxes else None,
        )
        return await self._run(graph, on_progress, {
            "tool": "edit_image", "prompt": prompt, "seed": seed, "quality": quality,
            "width": width, "height": height, "inputs": images,
            "boxes": layout_rows(boxes) if boxes else None,
        })

    async def upscale(
        self, image: str, factor: int = 2, on_progress: ProgressCallback | None = None
    ) -> GenerationResult:
        """Enlarge an image 2x or 4x with an upscaling model"""
        if factor not in (2, 4):
            raise ValueError("factor must be 2 or 4")
        data = await self.load_input(image)
        width, height = _image_size(data)
        limit = self.config.max_upscale_side
        if max(width, height) * factor > limit:
            raise ValueError(
                f"A {width}x{height} image upscaled {factor}x would exceed {limit} px; "
                f"use factor 2 or a smaller image"
            )
        return await self._process_one(
            "upscale", data, on_progress, {"tool": "upscale_image", "inputs": [image],
                                           "factor": factor},
            scale=factor / 4,
        )

    async def remove_background(
        self, image: str, on_progress: ProgressCallback | None = None
    ) -> GenerationResult:
        """Cut out the main subject onto a transparent background"""
        return await self._process_one(
            "remove_background", await self.load_input(image), on_progress,
            {"tool": "remove_background", "inputs": [image]},
        )

    async def get_image(self, image: str | None = None) -> tuple[StoredImage, dict[str, Any]]:
        """
        A stored image and its metadata; the most recent one when ``image`` is None

        Raises:
            ValueError: Unknown or malformed id, or no images yet
        """
        if image is None:
            stored = self.store.latest()
            if stored is None:
                raise ValueError("No images have been stored yet")
        else:
            image_id = self._own_image_id(image)
            stored = self.store.get(image_id) if image_id else None
            if stored is None:
                raise ValueError(f"No image {image!r} on this server")
        return stored, self.store.metadata(stored)

    async def unload_models(self) -> None:
        """Ask ComfyUI to free all model memory now (models reload on the next request)"""
        self._cancel_unload()
        await self.comfy.free()
        self._models_loaded = False
        logger.info("Asked ComfyUI to unload models")

    def memory_status(self) -> dict[str, Any]:
        """Whether flux-mcp has left models loaded, and when the idle unload fires"""
        remaining = None
        if self._unload_at is not None:
            remaining = max(0, round(self._unload_at - time.monotonic()))
        return {
            "models_loaded": self._models_loaded,
            "idle_unload_after_s": self.config.idle_unload_s or None,
            "idle_unload_in_s": remaining,
            "jobs_running": self._active_jobs,
        }

    async def _process_one(
        self,
        workflow: str,
        data: bytes,
        on_progress: ProgressCallback | None,
        metadata: dict[str, Any],
        **values: Any,
    ) -> GenerationResult:
        """Run a single-image workflow (no prompt, no seed)"""
        name = await self.comfy.upload(data, f"flux-mcp-{random.getrandbits(64):016x}.png")
        graph = self.workflows[workflow].fill("", enhance_prompt=False, image=name, **values)
        return await self._run(graph, on_progress, metadata)

    async def load_input(self, image: str) -> bytes:
        """
        Read an edit input given as an image id or an http(s) URL

        Raises:
            ValueError: Unknown id, unsupported URL, too large, or not an image
        """
        image_id = self._own_image_id(image)
        if image_id:
            path = self.store.path_for(image_id)
            if path is None:
                raise ValueError(f"No image with id {image_id} on this server")
            return path.read_bytes()

        image = image.strip()

        if not image.startswith(("http://", "https://")):
            raise ValueError("image must be an image id returned by this server or an http(s) URL")
        data = await self._download(image)
        _image_size(data)  # validates that it is an image
        return data

    async def _download(self, url: str) -> bytes:
        limit = self.config.max_input_bytes
        try:
            async with self.downloader.stream("GET", url, follow_redirects=True) as response:
                if response.status_code >= 400:
                    raise ValueError(f"Fetching {url} returned HTTP {response.status_code}")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f"Image at {url} is larger than {limit} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPError as e:
            raise ValueError(f"Could not fetch {url}: {e}") from e

    def _own_image_id(self, image: str) -> str | None:
        """The id in an image id or one of this server's image URLs, else None"""
        image = image.strip()
        own_prefix = f"{self.store.public_base_url}/images/"
        if image.startswith(own_prefix):
            image = image[len(own_prefix):].removesuffix(".png")
        return image if ID_PATTERN.match(image) else None

    async def _run(
        self,
        graph: dict[str, Any],
        on_progress: ProgressCallback | None,
        metadata: dict[str, Any],
    ) -> GenerationResult:
        """Submit a graph, wait for it, and store its images with metadata"""
        self._active_jobs += 1
        self._cancel_unload()
        try:
            prompt_id = await self.comfy.submit(graph)
            output = await self.comfy.wait(prompt_id, self.config.timeout_s, on_progress)
            self._models_loaded = True
            caption = output.texts[0].strip() if output.texts else None
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadata.update(_graph_settings(graph))
            if caption:
                metadata["caption"] = caption
            metadata["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            images = [self.store.save(await self.comfy.fetch(o), metadata) for o in output.images]
        finally:
            self._active_jobs -= 1
            if self._active_jobs == 0:
                self._schedule_unload()
        return GenerationResult(
            images, metadata.get("seed"), images[0].width, images[0].height, caption, metadata
        )

    def _schedule_unload(self) -> None:
        delay = self.config.idle_unload_s
        if delay <= 0 or not self._models_loaded:
            return
        self._unload_at = time.monotonic() + delay
        self._unload_task = asyncio.create_task(self._unload_when_idle(delay))

    def _cancel_unload(self) -> None:
        if self._unload_task is not None:
            self._unload_task.cancel()
        self._unload_task = None
        self._unload_at = None

    async def _unload_when_idle(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._unload_task = None  # so unload_models() doesn't cancel this task
        try:
            await self.unload_models()
            logger.info(f"Unloaded models after {delay:.0f} s idle")
        except ComfyError as e:
            logger.warning(f"Idle unload failed: {e}")

    def _workflow(self, name: str, quality: Quality):
        if quality not in QUALITY_SUFFIX:
            raise ValueError("quality must be 'fast' or 'high'")
        return self.workflows[name + QUALITY_SUFFIX[quality]]

    def _check_prompt(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(prompt) > self.config.max_prompt_chars:
            raise ValueError(f"prompt is longer than {self.config.max_prompt_chars} characters")

    def _check_count(self, count: int) -> None:
        if not 1 <= count <= self.config.max_count:
            raise ValueError(f"count must be between 1 and {self.config.max_count}")

    def _check_size(self, width: int, height: int) -> tuple[int, int]:
        lo, hi = self.config.min_side, self.config.max_side
        for name, value in (("width", width), ("height", height)):
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
        return _round(width), _round(height)

    @staticmethod
    def _check_steps(steps: int | None) -> int | None:
        if steps is not None and not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        return steps

    @staticmethod
    def _seed(seed: int | None) -> int:
        if seed is None:
            return random.randint(0, MAX_SEED)
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}")
        return seed


def _graph_settings(graph: dict[str, Any]) -> dict[str, Any]:
    """Model and sampler settings worth recording, read from the filled graph"""
    settings = {}
    for node in graph.values():
        inputs = node.get("inputs", {})
        if node.get("class_type") == "UNETLoader":
            settings["model"] = inputs.get("unet_name")
        elif node.get("class_type") == "KSampler":
            settings["steps"] = inputs.get("steps")
            settings["cfg"] = inputs.get("cfg")
    return settings


def _round(value: int) -> int:
    return max(SIZE_STEP, round(value / SIZE_STEP) * SIZE_STEP)


def _image_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height
    except UnidentifiedImageError as e:
        raise ValueError("The input is not a readable image") from e


def _fit_pixels(width: int, height: int, pixels: int) -> tuple[int, int]:
    scale = sqrt(pixels / (width * height))
    return _round(int(width * scale)), _round(int(height * scale))
