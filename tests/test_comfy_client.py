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
    output = await comfy.wait(prompt_id, timeout=5, on_progress=on_progress)
    assert [i.filename for i in output.images] == ["p1_0.png", "p1_1.png"]
    assert output.texts == []
    assert progress == ["queued, 1 of 1", "generating"]
    assert (await comfy.fetch(output.images[0])).startswith(b"\x89PNG")


async def test_text_outputs_and_free(fake_comfy, transport):
    comfy = client(transport)
    graph_with_caption = {**graph(), "22": {"inputs": {}}}
    output = await comfy.wait(await comfy.submit(graph_with_caption), timeout=5)
    assert output.texts == ["A detailed upsampled caption."]
    await comfy.free()
    assert fake_comfy.freed == 1


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
