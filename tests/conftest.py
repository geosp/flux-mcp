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


def png(width: int, height: int, color: str = "red", mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), color).save(out, format="PNG")
    return out.getvalue()


class FakeComfy:
    def __init__(self):
        self.prompts: dict[str, dict] = {}
        self.polls: dict[str, int] = {}
        self.uploads: list[str] = []
        self.upload_sizes: dict[str, tuple[int, int]] = {}
        self.fail_with: str | None = None
        self.never_finish = False
        self.deleted: list[str] = []
        self.remote_images: dict[str, bytes] = {}
        self.freed = 0

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
            graph = self.prompts[prompt_id]
            if graph.get("5", {}).get("class_type") == "JoinImageWithAlpha":
                width, height = self.upload_sizes[graph["1"]["inputs"]["image"]]
                return httpx.Response(200, content=png(width, height, (0, 0, 0, 0), "RGBA"))
            width, height = self._output_size(graph)
            return httpx.Response(200, content=png(width, height))
        if path == "/upload/image":
            name = f"upload{len(self.uploads) + 1}.png"
            self.uploads.append(name)
            data = _multipart_file(request)
            with Image.open(io.BytesIO(data)) as img:
                self.upload_sizes[name] = img.size
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/system_stats":
            return httpx.Response(200, json={
                "system": {"comfyui_version": "0.42.1", "ram_free": 40 * 1024**3},
                "devices": [{"name": "cuda:0 NVIDIA GB10", "vram_total": 121 * 1024**3,
                             "vram_free": 42 * 1024**3}],
            })
        if path == "/free":
            assert json.loads(request.content) == {"unload_models": True, "free_memory": True}
            self.freed += 1
            return httpx.Response(200, json={})
        if path.startswith("/models/"):
            return httpx.Response(200, json=[f"{path.rsplit('/', 1)[1]}-file.safetensors"])
        return httpx.Response(404)

    def _output_size(self, graph: dict) -> tuple[int, int]:
        if "7" in graph and "width" in graph["7"]["inputs"]:
            return graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]
        width, height = self.upload_sizes[graph["1"]["inputs"]["image"]]
        scale = 4 * graph["4"]["inputs"].get("resize_type.multiplier", 0.25)
        return int(width * scale), int(height * scale)

    def _history(self, prompt_id: str) -> dict:
        if self.fail_with:
            return {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "KSampler", "exception_message": self.fail_with}],
            ]}}
        latent = self.prompts[prompt_id].get("7", {}).get("inputs", {})
        count = latent.get("batch_size", 1)
        images = [{"filename": f"{prompt_id}_{i}.png", "subfolder": "flux-mcp", "type": "output"}
                  for i in range(count)]
        outputs = {"10": {"images": images}}
        if "22" in self.prompts[prompt_id]:
            outputs["22"] = {"text": ["A detailed upsampled caption."]}
        return {"outputs": outputs,
                "status": {"status_str": "success", "completed": True, "messages": []}}


def _multipart_file(request: httpx.Request) -> bytes:
    body = request.read()
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    for part in body.split(b"--" + boundary):
        if b'name="image"' in part:
            return part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    raise AssertionError("no image part")


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
