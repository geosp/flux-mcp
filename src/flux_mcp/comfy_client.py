"""
Async client for the ComfyUI HTTP API

Plain business logic: no MCP imports. Submits API-format workflows, waits for
them to finish and fetches the resulting images.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Called while waiting: (elapsed seconds, status message)
ProgressCallback = Callable[[float, str], Awaitable[None]]


class ComfyError(RuntimeError):
    """ComfyUI rejected a workflow, failed while running it, or is unreachable"""


@dataclass(frozen=True)
class OutputImage:
    """An image ComfyUI wrote, as listed in /history"""
    filename: str
    subfolder: str
    type: str


@dataclass(frozen=True)
class JobOutput:
    """What a finished prompt produced: saved images and text from text-output nodes"""
    images: list[OutputImage]
    texts: list[str]


class ComfyClient:
    """
    Thin wrapper over ComfyUI's /prompt, /history, /view, /upload/image, /queue
    and /system_stats endpoints
    """

    def __init__(self, http: httpx.AsyncClient, poll_interval: float = 0.5):
        """
        Args:
            http: Client with base_url set to the ComfyUI server
            poll_interval: Seconds between /history checks
        """
        self.http = http
        self.poll_interval = poll_interval
        self.client_id = uuid.uuid4().hex

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.http.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise ComfyError(f"ComfyUI is unreachable: {e}") from e
        if response.status_code >= 400:
            raise ComfyError(f"ComfyUI {method} {url} returned {response.status_code}: "
                             f"{_error_text(response)}")
        return response

    async def submit(self, workflow: dict[str, Any]) -> str:
        """Queue an API-format workflow and return its prompt id"""
        response = await self._request(
            "POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id}
        )
        data = response.json()
        if data.get("node_errors"):
            raise ComfyError(f"ComfyUI rejected the workflow: {data['node_errors']}")
        return data["prompt_id"]

    async def wait(
        self,
        prompt_id: str,
        timeout: float,
        on_progress: ProgressCallback | None = None,
    ) -> JobOutput:
        """
        Wait for a queued prompt to finish and return what it produced

        Raises:
            ComfyError: The prompt failed in ComfyUI
            TimeoutError: It did not finish within ``timeout`` seconds (it is
                removed from the queue or interrupted)
        """
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            entry = (await self._request("GET", f"/history/{prompt_id}")).json().get(prompt_id)
            if entry is not None:
                return _output_from_history(prompt_id, entry)
            if elapsed > timeout:
                await self.cancel(prompt_id)
                raise TimeoutError(f"Image generation did not finish within {timeout:.0f} s")
            if on_progress is not None:
                await on_progress(elapsed, await self._queue_status(prompt_id))
            await asyncio.sleep(self.poll_interval)

    async def _queue_status(self, prompt_id: str) -> str:
        queue = (await self._request("GET", "/queue")).json()
        if any(item[1] == prompt_id for item in queue.get("queue_running", [])):
            return "generating"
        pending = sorted(item[0] for item in queue.get("queue_pending", []))
        mine = [item[0] for item in queue.get("queue_pending", []) if item[1] == prompt_id]
        if mine:
            return f"queued, {pending.index(mine[0]) + 1} of {len(pending)}"
        return "finishing"

    async def cancel(self, prompt_id: str) -> None:
        """Remove a prompt from the queue, or interrupt it if it is running"""
        try:
            queue = (await self._request("GET", "/queue")).json()
            if any(item[1] == prompt_id for item in queue.get("queue_running", [])):
                await self._request("POST", "/interrupt", json={"prompt_id": prompt_id})
            else:
                await self._request("POST", "/queue", json={"delete": [prompt_id]})
        except ComfyError as e:
            logger.warning(f"Could not cancel prompt {prompt_id}: {e}")

    async def fetch(self, image: OutputImage) -> bytes:
        """Download an output image"""
        response = await self._request(
            "GET",
            "/view",
            params={"filename": image.filename, "subfolder": image.subfolder, "type": image.type},
        )
        return response.content

    async def upload(self, data: bytes, filename: str) -> str:
        """Upload an input image and return the name to put in a LoadImage node"""
        response = await self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, data)},
            data={"type": "input", "overwrite": "true"},
        )
        result = response.json()
        return f"{result['subfolder']}/{result['name']}" if result.get("subfolder") else result["name"]

    async def system_stats(self) -> dict[str, Any]:
        """ComfyUI version, devices and memory"""
        return (await self._request("GET", "/system_stats")).json()

    async def queue_length(self) -> dict[str, int]:
        """Number of running and pending prompts"""
        queue = (await self._request("GET", "/queue")).json()
        return {
            "running": len(queue.get("queue_running", [])),
            "pending": len(queue.get("queue_pending", [])),
        }

    async def free(self) -> None:
        """Ask ComfyUI to unload all models and release cached memory (applied when idle)"""
        await self._request("POST", "/free", json={"unload_models": True, "free_memory": True})

    async def models(self, folder: str) -> list[str]:
        """Model files ComfyUI sees in a folder such as ``diffusion_models``"""
        return (await self._request("GET", f"/models/{folder}")).json()


def _output_from_history(prompt_id: str, entry: dict[str, Any]) -> JobOutput:
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        raise ComfyError(f"ComfyUI failed on prompt {prompt_id}: {_execution_error(status)}")
    images = [
        OutputImage(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
        for node_output in entry.get("outputs", {}).values()
        for img in node_output.get("images", [])
        if img.get("type", "output") == "output"
    ]
    if not images:
        raise ComfyError(f"ComfyUI finished prompt {prompt_id} without saving an image")
    texts = [
        text
        for node_output in entry.get("outputs", {}).values()
        for text in node_output.get("text", [])
        if isinstance(text, str)
    ]
    return JobOutput(images, texts)


def _execution_error(status: dict[str, Any]) -> str:
    for name, details in status.get("messages", []):
        if name == "execution_error":
            node = details.get("node_type", "?")
            return f"{node}: {details.get('exception_message', '').strip()}"
    return "unknown error"


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        text = error.get("message", "")
        if data.get("node_errors"):
            text += f" {data['node_errors']}"
        return text
    return str(data)[:500]
