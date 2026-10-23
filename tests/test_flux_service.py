import asyncio
import io

import pytest
from PIL import Image

from flux_mcp.service import FluxMCPService

from .conftest import png


@pytest.fixture
def flux(app_config, transport):
    service = FluxMCPService(app_config, transport)
    service.initialize()
    service.flux.comfy.poll_interval = 0
    return service.flux


async def test_generate_rounds_size_and_stores(flux, fake_comfy):
    result = await flux.generate("a fox", width=1000, height=600, count=2, seed=42)
    assert (result.width, result.height, result.seed) == (992, 608, 42)
    assert len(result.images) == 2
    image = result.images[0]
    assert image.url == f"http://flux.test:8300/images/{image.id}.png"
    assert image.path.read_bytes().startswith(b"\x89PNG")
    assert (image.width, image.height) == (992, 608)


@pytest.mark.parametrize("kwargs, message", [
    ({"prompt": ""}, "empty"),
    ({"prompt": "x" * 5000}, "longer than"),
    ({"width": 8192}, "width must be"),
    ({"count": 9}, "count must be"),
    ({"steps": 0}, "steps must be"),
    ({"seed": -1}, "seed must be"),
])
async def test_generate_validation(flux, kwargs, message):
    args = {"prompt": "a fox", **kwargs}
    with pytest.raises(ValueError, match=message):
        await flux.generate(**args)


async def test_edit_by_id_keeps_aspect_ratio(flux, fake_comfy):
    source = flux.store.save(png(2000, 1000))
    result = await flux.edit("make it blue", [source.id])
    assert (result.width, result.height) == (1440, 720)
    graph = fake_comfy.prompts["p1"]
    assert graph["11"]["inputs"]["image"] == "upload1.png"
    assert graph["21"]["inputs"]["prompt"].endswith("User request: make it blue\nOutput the caption.")


async def test_edit_with_several_references(flux, fake_comfy):
    first = flux.store.save(png(1024, 768))
    fake_comfy.remote_images["https://example.com/forest.jpg"] = png(800, 800)
    result = await flux.edit("the person from the first image in the forest from the second "
                             "image", [first.id, "https://example.com/forest.jpg"])
    assert (result.width, result.height) == (1184, 880)
    graph = fake_comfy.prompts["p1"]
    assert fake_comfy.uploads == ["upload1.png", "upload2.png"]
    assert graph["24"]["inputs"]["image"] == "upload2.png"
    assert graph["12"]["inputs"]["images.image1"] == ["24", 0]


@pytest.mark.parametrize("count", [0, 11])
async def test_edit_reference_count(flux, fake_comfy, count):
    with pytest.raises(ValueError, match="1 to 10"):
        await flux.edit("x", [flux.store.save(png(64, 64)).id] * count)
    assert fake_comfy.uploads == []


async def test_edit_accepts_own_url_and_remote_url(flux, fake_comfy):
    source = flux.store.save(png(512, 512))
    assert await flux.load_input(source.url) == source.path.read_bytes()
    fake_comfy.remote_images["https://example.com/cat.png"] = png(64, 64)
    assert await flux.load_input("https://example.com/cat.png") == png(64, 64)


@pytest.mark.parametrize("image, message", [
    ("0" * 32, "No image with id"),
    ("/etc/passwd", "image id returned by this server or an http"),
    ("https://example.com/missing.png", "HTTP 404"),
])
async def test_edit_rejects_bad_input(flux, image, message):
    with pytest.raises(ValueError, match=message):
        await flux.load_input(image)


async def test_edit_rejects_oversized_and_non_images(flux, fake_comfy):
    flux.config.max_input_bytes = 100
    fake_comfy.remote_images["https://example.com/big.png"] = png(512, 512)
    with pytest.raises(ValueError, match="larger than"):
        await flux.load_input("https://example.com/big.png")
    flux.config.max_input_bytes = 10_000_000
    fake_comfy.remote_images["https://example.com/text"] = b"hello"
    with pytest.raises(ValueError, match="not a readable image"):
        await flux.load_input("https://example.com/text")


async def test_quality_selects_workflow(flux, fake_comfy):
    await flux.generate("a fox", quality="high")
    await flux.edit("make it blue", [flux.store.save(png(64, 64)).id], quality="high")
    assert fake_comfy.prompts["p1"]["8"]["inputs"]["steps"] == 50
    assert fake_comfy.prompts["p2"]["1"]["inputs"]["unet_name"].startswith("flux3-image-klein-base")
    with pytest.raises(ValueError, match="quality"):
        await flux.generate("a fox", quality="ultra")


@pytest.mark.parametrize("factor, size", [(2, (200, 100)), (4, (400, 200))])
async def test_upscale(flux, fake_comfy, factor, size):
    source = flux.store.save(png(100, 50))
    result = await flux.upscale(source.id, factor)
    assert (result.width, result.height) == size and result.seed is None
    assert fake_comfy.prompts["p1"]["4"]["inputs"]["resize_type.multiplier"] == factor / 4


async def test_upscale_limits(flux):
    big = flux.store.save(png(3000, 1000))
    with pytest.raises(ValueError, match="would exceed 8192"):
        await flux.upscale(big.id, 4)
    with pytest.raises(ValueError, match="factor must be 2 or 4"):
        await flux.upscale(big.id, 3)


async def test_remove_background_keeps_alpha(flux):
    result = await flux.remove_background(flux.store.save(png(80, 60)).id)
    with Image.open(result.images[0].path) as img:
        assert img.mode == "RGBA" and img.size == (80, 60)
    with Image.open(io.BytesIO(flux.store.preview(result.images[0]))) as preview:
        assert preview.getpixel((0, 0)) == (255, 255, 255)



async def test_idle_unload_after_jobs(flux, fake_comfy):
    flux.config.idle_unload_s = 0.05
    await flux.generate("a fox")
    assert flux.memory_status()["models_loaded"] and flux.memory_status()["idle_unload_in_s"] == 0
    await asyncio.sleep(0.15)
    assert fake_comfy.freed == 1 and not flux.memory_status()["models_loaded"]


async def test_new_job_postpones_idle_unload(flux, fake_comfy):
    flux.config.idle_unload_s = 0.2
    await flux.generate("a fox")
    await asyncio.sleep(0.1)
    await flux.generate("another fox")
    await asyncio.sleep(0.15)
    assert fake_comfy.freed == 0
    await asyncio.sleep(0.15)
    assert fake_comfy.freed == 1


async def test_idle_unload_disabled(flux, fake_comfy):
    flux.config.idle_unload_s = 0
    await flux.generate("a fox")
    assert flux.memory_status()["idle_unload_in_s"] is None


async def test_metadata_without_upsampler_has_no_caption(flux):
    result = await flux.generate("A complete caption.", enhance_prompt=False, seed=4)
    assert result.caption is None
    stored, metadata = await flux.get_image(result.images[0].id)
    assert metadata["prompt"] == "A complete caption." and "caption" not in metadata
    assert (await flux.get_image())[0].id == stored.id


async def test_upload_has_no_metadata(flux):
    stored = flux.store.save(png(8, 8))
    assert (await flux.get_image(stored.url))[1] == {}
