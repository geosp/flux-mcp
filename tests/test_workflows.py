import json
from pathlib import Path

import pytest

from flux_mcp.workflows import load_workflows

from .conftest import WORKFLOWS_DIR


@pytest.fixture
def workflows():
    return load_workflows(WORKFLOWS_DIR)


def test_shipped_workflows_load(workflows):
    assert set(workflows) == {"t2i", "edit"}


def test_fill_sets_params_without_changing_template(workflows):
    t2i = workflows["t2i"]
    graph = t2i.fill("a fox", width=1536, height=1024, batch_size=2, seed=7, steps=8)
    assert graph["7"]["inputs"] == {"width": 1536, "height": 1024, "batch_size": 2}
    assert graph["8"]["inputs"]["seed"] == 7
    assert graph["21"]["inputs"]["sampling_mode.seed"] == 7
    assert graph["8"]["inputs"]["steps"] == 8
    assert t2i.graph["7"]["inputs"]["width"] == 1024


def test_none_keeps_default(workflows):
    graph = workflows["t2i"].fill("a fox", steps=None)
    assert graph["8"]["inputs"]["steps"] == 16


def test_upsampler_on_formats_request(workflows):
    graph = workflows["t2i"].fill("a fox", True, width=1536, height=1024)
    assert graph["21"]["inputs"]["prompt"] == "TARGET CANVAS: 3:2\na fox"
    assert graph["5"]["inputs"]["text"] == ["21", 0]


def test_upsampler_off_removes_nodes(workflows):
    graph = workflows["edit"].fill("make it blue", False, images=["a.png", "b.png"])
    assert not {"20", "21", "23"} & set(graph)
    assert graph["5"]["inputs"]["text"] == "make it blue"
    assert graph["12"]["inputs"]["images.image1"] == ["24", 0]


def test_one_reference(workflows):
    graph = workflows["edit"].fill("make it blue", images=["a.png"])
    assert graph["11"]["inputs"]["image"] == "a.png"
    assert "images.image1" not in graph["12"]["inputs"]
    assert graph["23"]["inputs"] == {"images.image0": ["11", 0]}


def test_several_references_clone_the_loader(workflows):
    graph = workflows["edit"].fill("combine them", images=["a.png", "b.png", "c.png"])
    assert graph["24"]["class_type"] == "LoadImage" and graph["24"]["inputs"]["image"] == "b.png"
    assert graph["25"]["inputs"]["image"] == "c.png"
    for node in ("12", "23"):
        inputs = graph[node]["inputs"]
        assert [inputs[f"images.image{i}"] for i in range(3)] == [["11", 0], ["24", 0], ["25", 0]]
    assert "24" not in workflows["edit"].graph


@pytest.mark.parametrize("images", [[], [f"{i}.png" for i in range(11)]])
def test_reference_count_checked(workflows, images):
    with pytest.raises(ValueError, match="1 to 10 reference images"):
        workflows["edit"].fill("x", images=images)


def test_t2i_takes_no_references(workflows):
    with pytest.raises(ValueError, match="no reference images"):
        workflows["t2i"].fill("a fox", images=["a.png"])


def test_unknown_param_rejected(workflows):
    with pytest.raises(KeyError):
        workflows["t2i"].fill("a fox", denoise=0.5)


def test_bad_mapping_rejected(tmp_path: Path):
    (tmp_path / "w.json").write_text(json.dumps({"1": {"class_type": "X", "inputs": {"a": 1}}}))
    (tmp_path / "nodes.yaml").write_text("w:\n  file: w.json\n  params:\n    b: [['1', 'b']]\n")
    with pytest.raises(ValueError, match="no input 'b'"):
        load_workflows(tmp_path)


def test_missing_file_rejected(tmp_path: Path):
    (tmp_path / "nodes.yaml").write_text("w:\n  file: nope.json\n  params: {}\n")
    with pytest.raises(ValueError, match="file not found"):
        load_workflows(tmp_path)
