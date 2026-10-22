import httpx
import pytest

from flux_mcp.comfy_client import ComfyClient, ComfyError


def client(transport) -> ComfyClient:
    return ComfyClient(httpx.AsyncClient(base_url="http://comfy", transport=transport),
                       poll_interval=0)


def graph(batch_size=1):
    return {"7": {"inputs": {"width": 64, "height": 32, "batch_size": batch_size}}}


async def test_submit_wait_fetch(transport):
    comfy = client(transport)
    progress = []

    async def on_progress(elapsed, status):
        progress.append(status)

    prompt_id = await comfy.submit(graph(batch_size=2))
    images = await comfy.wait(prompt_id, timeout=5, on_progress=on_progress)
    assert [i.filename for i in images] == ["p1_0.png", "p1_1.png"]
    assert progress == ["queued, 1 of 1", "generating"]
    assert (await comfy.fetch(images[0])).startswith(b"\x89PNG")


async def test_execution_error(fake_comfy, transport):
    fake_comfy.fail_with = "CUDA out of memory"
    comfy = client(transport)
    with pytest.raises(ComfyError, match="KSampler: CUDA out of memory"):
        await comfy.wait(await comfy.submit(graph()), timeout=5)


async def test_timeout_cancels(fake_comfy, transport):
    fake_comfy.never_finish = True
    comfy = client(transport)
    prompt_id = await comfy.submit(graph())
    with pytest.raises(TimeoutError):
        await comfy.wait(prompt_id, timeout=0)
    assert fake_comfy.deleted == [prompt_id]


async def test_rejected_workflow():
    def handler(request):
        return httpx.Response(400, json={
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {"1": {"errors": [{"message": "Value not in list"}]}},
        })
    comfy = client(httpx.MockTransport(handler))
    with pytest.raises(ComfyError, match="failed validation.*Value not in list"):
        await comfy.submit(graph())


async def test_unreachable():
    def handler(request):
        raise httpx.ConnectError("connection refused")
    with pytest.raises(ComfyError, match="unreachable"):
        await client(httpx.MockTransport(handler)).queue_length()
