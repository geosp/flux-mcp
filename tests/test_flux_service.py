import pytest

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
