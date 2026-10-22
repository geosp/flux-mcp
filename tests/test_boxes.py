import pytest
from pydantic import ValidationError

from flux_mcp.boxes import MAX_BOXES, Box, layout_rows
from flux_mcp.workflows import load_workflows

from .conftest import WORKFLOWS_DIR

ROWS = [{"id": "obj_1", "bbox": [100, 50, 900, 450], "desc": "a red fox"}]


@pytest.fixture
def workflows():
    return load_workflows(WORKFLOWS_DIR)


def test_rows_like_flux3_layout_prompt():
    rows = layout_rows([Box(bbox=[0, 0, 500, 500], desc="a sign", text="Open"),
                        Box(bbox=[500, 500, 1000, 1000], desc="a cat")])
    assert rows == [{"id": "obj_1", "bbox": [0, 0, 500, 500], "desc": 'a sign "Open"'},
                    {"id": "obj_2", "bbox": [500, 500, 1000, 1000], "desc": "a cat"}]


@pytest.mark.parametrize("bbox", [[0, 0, 500], [500, 0, 100, 900], [0, 0, 500, 1001], [0, 9, 9, 9]])
def test_bad_boxes_rejected(bbox):
    with pytest.raises(ValidationError):
        Box(bbox=bbox, desc="x")


def test_too_many_boxes():
    with pytest.raises(ValueError, match=f"at most {MAX_BOXES}"):
        layout_rows([Box(bbox=[0, 0, 10, 10], desc="x")] * (MAX_BOXES + 1))


@pytest.mark.parametrize("name, system_start, request_start", [
    ("t2i", "You are the prompt upsampler of a text-to-image model. Plan",
     "TARGET CANVAS: 3:2\nORIGINAL PROMPT:\na fox\n\nLOCKED BOUNDING-BOX ANNOTATIONS"),
    ("t2i_hq", "You are the prompt upsampler of a text-to-image model. Plan",
     "TARGET CANVAS: 3:2\nORIGINAL PROMPT:\na fox"),
    ("edit", "You are the annotation system an image-editing model was trained on",
     "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
     "User editing instruction: ORIGINAL PROMPT:\na fox\n\nLOCKED"),
])
def test_boxes_switch_upsampler(workflows, name, system_start, request_start):
    extra = {"images": ["a.png"]} if name == "edit" else {}
    graph = workflows[name].fill("a fox ", boxes=ROWS, width=1536, height=1024, **extra)
    request = graph["21"]["inputs"]["prompt"]
    assert graph["21"]["inputs"]["system_prompt"].startswith(system_start)
    assert request.startswith(request_start)
    assert '[{"id": "obj_1", "bbox": [100, 50, 900, 450], "desc": "a red fox"}]' in request


def test_without_boxes_plain_system_prompt(workflows):
    graph = workflows["t2i"].fill("a fox", width=1024, height=1024)
    assert "Plan the full image" not in graph["21"]["inputs"]["system_prompt"]


def test_boxes_without_upsampler_append_table(workflows):
    graph = workflows["t2i"].fill("A caption.", False, boxes=ROWS)
    assert graph["5"]["inputs"]["text"].startswith('A caption. [{"id": "obj_1"')


def test_boxes_unsupported_workflow(workflows):
    with pytest.raises(ValueError, match="does not support bounding boxes"):
        workflows["upscale"].fill("", False, image="a.png", boxes=ROWS)
