# flux-mcp

An MCP server that lets any MCP client (Claude Code, Claude Desktop, Open WebUI, your own
agents) generate and edit images with **FLUX 3 Image Klein**, running locally in ComfyUI.

```
MCP client ──streamable HTTP + bearer token──▶ flux-mcp :8300 ──HTTP──▶ ComfyUI :8188 (GPU)
```

flux-mcp has no GPU or model code: it fills a ComfyUI workflow, submits it, waits, and returns
a JPEG preview plus a link to the full-resolution PNG. It is built on
[mcp-service-kit](https://github.com/geosp/mcp-service-kit), which provides the transports,
CLI and configuration.

## Tools

| Tool | What it does |
|---|---|
| `generate_image(prompt, width=1024, height=1024, count=1, seed?, steps?, enhance_prompt=true, quality="fast")` | Text to image. `quality="high"` uses the undistilled Klein Base model (50 steps, CFG 4): more detail and variety, about 3× slower |
| `edit_image(prompt, images, width?, height?, count=1, seed?, steps?, enhance_prompt=true, quality="fast")` | Edit an image, or compose a new one from up to 10 reference images. `images` lists ids (from previous results) or http(s) URLs; the prompt refers to them by order ("the fox from the first image on the porch from the second image"). Output keeps the first image's aspect ratio at about 1 MP unless a size is given |
| `upscale_image(image, factor=2)` | Enlarge 2× or 4× with Real-ESRGAN (output ≤ 8192 px per side) |
| `remove_background(image)` | Cut out the subject onto a transparent PNG (BiRefNet) |
| `list_models()` | Workflows and the model files ComfyUI sees |
| `server_status()` | ComfyUI version, queue length, free memory |

Each image result contains a text block with the seed and full-resolution URL
(`<FLUX_PUBLIC_BASE_URL>/images/<id>.png`), one JPEG preview (longest side ≤ 1024 px) per image,
and structured content `{seed, width, height, images: [{id, url}]}`.

**Prompt upsampler.** FLUX 3 Klein expects full scene captions; a short prompt as typed gives
generic results. By default (`enhance_prompt=true`) the request first goes through Black Forest
Labs' Qwen3.5-9B prompt upsampler. Turn it off only when the client writes a complete caption.

Limits (configurable): sides 256–4096 px (rounded to multiples of 16), up to 4 images per call,
prompts up to 4000 characters, edit inputs up to 20 MB, 300 s per call.

## Layout with bounding boxes

`generate_image` and `edit_image` take optional `boxes`. Each box is
`{"bbox": [y_min, x_min, y_max, x_max], "desc": "...", "text": "..."}` on a 0–1000 grid with
(0, 0) at the top left, stretched over the canvas; `text` is optional and rendered verbatim.
Boxes guide placement and scale; they are not hard masks.

```json
{"prompt": "a snowy forest clearing", "width": 1536, "height": 1024, "boxes": [
  {"bbox": [350, 60, 950, 420], "desc": "a red fox sitting, facing right"},
  {"bbox": [250, 600, 800, 950], "desc": "a weathered wooden trail sign", "text": "FOX TRAIL"}]}
```

For `edit_image`, boxes mark regions of the **first** image and are named `obj_1`, `obj_2`, …
in order, so the prompt can say "remove obj_1" or "replace obj_2 with a bicycle". Behind the
scenes the prompt upsampler switches to BFL's box-aware system prompts
(`workflows/prompts/*_bbox.txt`); with `enhance_prompt=false` the element table is appended to
your caption, as ComfyUI's `Flux3LayoutPrompt` does.

## Uploading your own images

Tools take image ids, so upload an image first and use the returned id:

```bash
curl -H "Authorization: Bearer $FLUX_MCP_TOKEN" -F file=@photo.jpg http://10.0.0.96:8300/upload
# {"id": "1329da32…", "url": "http://10.0.0.96:8300/images/1329da32….png", "width": 1776, "height": 1184}
```

Agents with a shell (Claude Code, Cline, opencode) can run this themselves; the server's
instructions tell them how. In a browser, open `http://10.0.0.96:8300/upload`, paste the token,
and drop images to get their ids. Images pasted into a chat cannot be forwarded to a tool: the
model only receives them, it cannot hand the bytes on. Uploads are limited to
`FLUX_MAX_INPUT_BYTES` (20 MB) and stored as PNG.

## Models

The weights are in the gated repo
[black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
(FLUX Kommunity License; accept it on that page with the account your `HF_TOKEN` belongs to).

```bash
cd /mnt/shared/huggingface/comfyui     # the MODELS_DIR mounted into ComfyUI
for f in diffusion_models/flux3-image-klein-nvfp4.safetensors \
         diffusion_models/flux3-image-klein-base-nvfp4.safetensors \
         text_encoders/qwen3vl_4b_bf16.safetensors \
         text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors \
         vae/flux2-vae.safetensors; do
  uvx --from 'huggingface_hub[cli]' hf download black-forest-labs/flux-3-image-klein-comfy "$f" --local-dir .
done
uvx --from 'huggingface_hub[cli]' hf download Comfy-Org/BiRefNet background_removal/birefnet.safetensors --local-dir .
mkdir -p upscale_models && curl -L -o upscale_models/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

| File | Size | Role |
|---|---|---|
| `flux3-image-klein-nvfp4.safetensors` | 5.3 GB | distilled transformer, NVFP4 (Blackwell) |
| `qwen3vl_4b_bf16.safetensors` | 8.9 GB | text encoder |
| `qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors` | 6.7 GB | prompt upsampler |
| `flux2-vae.safetensors` | 0.3 GB | VAE |
| `flux3-image-klein-base-nvfp4.safetensors` | 5.3 GB | undistilled base model, for `quality="high"` |
| `upscale_models/RealESRGAN_x4plus.pth` | 64 MB | upscaler (BSD-3-Clause) |
| `background_removal/birefnet.safetensors` | 0.4 GB | background removal (MIT) |

To use other files (int8, bf16, the base checkpoint), edit the loader nodes in
`workflows/*.json`.

## Performance (DGX Spark / GB10, next to a running vLLM)

Measured through the MCP endpoint, 1024×1024, one image, 16 steps:

| Call | Time |
|---|---|
| First call after ComfyUI starts (loads ~21 GB of weights from NFS) | ~260 s |
| `generate_image`, upsampler on | ~17 s |
| `generate_image`, `enhance_prompt=false` | ~8.5 s |
| `edit_image`, one reference, upsampler on | ~29 s |
| `edit_image`, two references, 1536×1024 | ~45 s |
| `generate_image` with two boxes, 1536×1024 | ~27 s |
| `generate_image`, `quality="high"` | ~46 s (~58 s when the base model loads) |
| `upscale_image` 2× / 4× from 1024² | ~10 s / ~11 s |
| `remove_background`, 1024² | ~2 s |

Memory: with the fast-mode models loaded, the host's available memory dropped by about 10 GB
(the rest came out of page cache); with the base model resident as well, about 27 GB stayed
available next to vLLM. Clients with a short tool-call timeout should allow at least 60 s
(120 s for `quality="high"`), or more for the first call; flux-mcp sends progress notifications
while it waits.

## Run

```bash
cp .env.example .env
sed -i "s/^FLUX_MCP_TOKEN=.*/FLUX_MCP_TOKEN=$(openssl rand -hex 32)/" .env
docker compose up -d --build
docker compose logs -f flux-mcp
```

The ComfyUI web UI is at `http://<host>:8188`, the MCP endpoint at `http://<host>:8300/mcp`.
Both are meant for the LAN only: ComfyUI has no authentication, and image URLs are protected
only by their random ids.

Configuration (environment or `.env`):

| Variable | Default | |
|---|---|---|
| `FLUX_MCP_TOKEN` | — | bearer token; required for HTTP unless `--no-auth` |
| `FLUX_COMFYUI_URL` | `http://localhost:8188` | compose sets `http://comfyui:8188` |
| `FLUX_PUBLIC_BASE_URL` | `http://localhost:8300` | base of the image URLs returned to clients |
| `FLUX_OUTPUT_DIR` | `data/images` | |
| `FLUX_WORKFLOWS_DIR` | `workflows` | |
| `FLUX_MAX_SIDE`, `FLUX_MIN_SIDE`, `FLUX_MAX_COUNT`, `FLUX_MAX_PROMPT_CHARS`, `FLUX_MAX_INPUT_BYTES`, `FLUX_TIMEOUT_S`, `FLUX_PREVIEW_MAX_SIDE` | see `config.py` | limits |
| `MCP_PORT`, `MCP_HOST` | `3000`, `0.0.0.0` | from mcp-service-kit (the image sets `MCP_PORT=8300`) |

### Without Docker (server only)

```bash
uv sync
FLUX_MCP_TOKEN=... FLUX_COMFYUI_URL=http://10.0.0.96:8188 uv run flux-mcp --mode http --port 8300
uv run flux-mcp --mode stdio      # for local clients that launch the server themselves
```

## Connect a client

Any MCP client that supports streamable HTTP: URL `http://10.0.0.96:8300/mcp`, header
`Authorization: Bearer <FLUX_MCP_TOKEN>`.

**Claude Code**

```bash
claude mcp add --transport http flux http://10.0.0.96:8300/mcp \
  --header "Authorization: Bearer $FLUX_MCP_TOKEN"
```

**Claude Desktop** (`claude_desktop_config.json`, via stdio on the same machine as ComfyUI or
with `FLUX_COMFYUI_URL` pointing at it):

```json
{
  "mcpServers": {
    "flux": {
      "command": "uv",
      "args": ["--directory", "/path/to/flux-mcp", "run", "flux-mcp", "--mode", "stdio"],
      "env": {"FLUX_COMFYUI_URL": "http://10.0.0.96:8188"}
    }
  }
}
```

In stdio mode, image URLs point at `FLUX_PUBLIC_BASE_URL`, which only works while an HTTP
instance serves the same `FLUX_OUTPUT_DIR`; the inline previews always work.

**Open WebUI**: Settings → External Tools → add an MCP (Streamable HTTP) server with the URL
above and the bearer token.

**MCP Inspector**: `npx @modelcontextprotocol/inspector`, transport "Streamable HTTP", same URL
and header.

## Workflows

`workflows/flux3_klein_t2i.json` and `flux3_klein_edit.json` are ComfyUI API-format graphs,
flattened by hand from Black Forest Labs' Klein templates (MIT, in the model repo's
`workflows/`). `workflows/nodes.yaml` maps tool parameters to node inputs and describes the
upsampler nodes and the reference-image inputs (extra references get copies of the image
loader node, wired into the reference and upsampler nodes), so you can swap models or add a workflow without code changes. To start from
a workflow built in the web UI, use *Export (API)* and map its node ids in `nodes.yaml`.

## Development

```bash
uv sync
uv run pytest        # fake ComfyUI over httpx.MockTransport; no GPU needed
uv run ruff check
```

Layout follows mcp-service-kit's example: `server.py` (server, auth, `main`), `service.py`
(tool discovery, image route), `features/*/tool.py` (one tool each), `flux_service.py` and
`comfy_client.py` (business logic, no MCP imports), `workflows.py`, `images.py`, `config.py`.

## License

Code: Apache License 2.0, see [LICENSE](LICENSE). FLUX 3 Image Klein weights: FLUX Kommunity License. Text encoder and
prompt upsampler: Apache-2.0. Workflow templates: MIT (Comfy Org).
