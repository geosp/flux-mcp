"""
Image generation and editing on top of ComfyUI

Business logic shared by the MCP tools: validates requests, fills workflows,
runs them and stores the results. No MCP imports.
"""

import io
import logging
import random
from dataclasses import dataclass
from math import sqrt
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from .comfy_client import ComfyClient, ProgressCallback
from .config import FluxConfig
from .images import ID_PATTERN, ImageStore, StoredImage
from .workflows import Workflow

logger = logging.getLogger(__name__)

# Latent sizes must be multiples of this
SIZE_STEP = 16
MAX_SEED = 2**53 - 1
# Default edit output size, in pixels, when the caller gives none
EDIT_DEFAULT_PIXELS = 1024 * 1024


@dataclass(frozen=True)
class GenerationResult:
    images: list[StoredImage]
    seed: int
    width: int
    height: int


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

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        count: int = 1,
        seed: int | None = None,
        steps: int | None = None,
        enhance_prompt: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Text to image"""
        self._check_prompt(prompt)
        self._check_count(count)
        width, height = self._check_size(width, height)
        seed = self._seed(seed)
        graph = self.workflows["t2i"].fill(
            prompt,
            enhance_prompt,
            width=width,
            height=height,
            batch_size=count,
            seed=seed,
            steps=self._check_steps(steps),
        )
        images = await self._run(graph, on_progress)
        return GenerationResult(images, seed, width, height)

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
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """
        Edit an image, or compose a new one from several reference images

        Args:
            images: Image ids from this server or http(s) URLs, in the order the
                prompt refers to them ("the first image", "the second image", ...)
            width, height: Output size; by default the first image's aspect ratio
                at about one megapixel
        """
        self._check_prompt(prompt)
        self._check_count(count)
        workflow = self.workflows["edit"]
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
        )
        images = await self._run(graph, on_progress)
        return GenerationResult(images, seed, width, height)

    async def load_input(self, image: str) -> bytes:
        """
        Read an edit input given as an image id or an http(s) URL

        Raises:
            ValueError: Unknown id, unsupported URL, too large, or not an image
        """
        image = image.strip()
        own_prefix = f"{self.store.public_base_url}/images/"
        if image.startswith(own_prefix):
            image = image[len(own_prefix):].removesuffix(".png")

        if ID_PATTERN.match(image):
            path = self.store.path_for(image)
            if path is None:
                raise ValueError(f"No image with id {image} on this server")
            return path.read_bytes()

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

    async def _run(self, graph: dict[str, Any], on_progress: ProgressCallback | None) -> list[StoredImage]:
        prompt_id = await self.comfy.submit(graph)
        outputs = await self.comfy.wait(prompt_id, self.config.timeout_s, on_progress)
        return [self.store.save(await self.comfy.fetch(o)) for o in outputs]

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
