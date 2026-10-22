"""
A fake ComfyUI served through httpx.MockTransport

It accepts /prompt, reports the prompt as queued once and running once, then
finished, and serves a solid-color PNG of the requested latent size.
"""

import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from flux_mcp.config import AppConfig, FluxConfig

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"


def png(width: int, height: int, color: str = "red") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


class FakeComfy:
    def __init__(self):
        self.prompts: dict[str, dict] = {}
        self.polls: dict[str, int] = {}
        self.uploads: list[str] = []
        self.fail_with: str | None = None
        self.never_finish = False
        self.deleted: list[str] = []
        self.remote_images: dict[str, bytes] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "comfy":
            data = self.remote_images.get(str(request.url))
            return httpx.Response(200, content=data) if data else httpx.Response(404)

        path = request.url.path
        if path == "/prompt":
            graph = json.loads(request.content)["prompt"]
            prompt_id = f"p{len(self.prompts) + 1}"
            self.prompts[prompt_id] = graph
            self.polls[prompt_id] = 0
            return httpx.Response(200, json={"prompt_id": prompt_id, "number": 1, "node_errors": {}})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[1]
            self.polls[prompt_id] += 1
            if self.never_finish or self.polls[prompt_id] < 3:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={prompt_id: self._history(prompt_id)})
        if path == "/queue" and request.method == "GET":
            running = [[1, pid, {}, {}, []] for pid, n in self.polls.items() if n == 2]
            pending = [[0, pid, {}, {}, []] for pid, n in self.polls.items() if n == 1]
            return httpx.Response(200, json={"queue_running": running, "queue_pending": pending})
        if path == "/queue" and request.method == "POST":
            self.deleted += json.loads(request.content).get("delete", [])
            return httpx.Response(200, json={})
        if path == "/view":
            prompt_id, index = request.url.params["filename"].removesuffix(".png").split("_")
            latent = self._latent(prompt_id)
            return httpx.Response(200, content=png(latent["width"], latent["height"]))
        if path == "/upload/image":
            name = f"upload{len(self.uploads) + 1}.png"
            self.uploads.append(name)
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/system_stats":
            return httpx.Response(200, json={
                "system": {"comfyui_version": "0.42.1", "ram_free": 40 * 1024**3},
                "devices": [{"name": "cuda:0 NVIDIA GB10", "vram_total": 121 * 1024**3,
                             "vram_free": 42 * 1024**3}],
            })
        if path.startswith("/models/"):
            return httpx.Response(200, json=[f"{path.rsplit('/', 1)[1]}-file.safetensors"])
        return httpx.Response(404)

    def _latent(self, prompt_id: str) -> dict:
        return self.prompts[prompt_id]["7"]["inputs"]

    def _history(self, prompt_id: str) -> dict:
        if self.fail_with:
            return {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "KSampler", "exception_message": self.fail_with}],
            ]}}
        count = self._latent(prompt_id)["batch_size"]
        images = [{"filename": f"{prompt_id}_{i}.png", "subfolder": "flux-mcp", "type": "output"}
                  for i in range(count)]
        return {"outputs": {"10": {"images": images}},
                "status": {"status_str": "success", "completed": True, "messages": []}}


@pytest.fixture
def fake_comfy() -> FakeComfy:
    return FakeComfy()


@pytest.fixture
def transport(fake_comfy: FakeComfy) -> httpx.MockTransport:
    return httpx.MockTransport(fake_comfy.handler)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(flux=FluxConfig(
        comfyui_url="http://comfy",
        public_base_url="http://flux.test:8300",
        output_dir=tmp_path / "images",
        workflows_dir=WORKFLOWS_DIR,
        timeout_s=5,
    ))
