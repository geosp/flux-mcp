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
    graph = workflows["edit"].fill("make it blue", False, image="x.png")
    assert "20" not in graph and "21" not in graph
    assert graph["5"]["inputs"]["text"] == "make it blue"
    assert graph["11"]["inputs"]["image"] == "x.png"


def test_unknown_param_rejected(workflows):
    with pytest.raises(KeyError):
        workflows["t2i"].fill("a fox", image="x.png")


def test_bad_mapping_rejected(tmp_path: Path):
    (tmp_path / "w.json").write_text(json.dumps({"1": {"class_type": "X", "inputs": {"a": 1}}}))
    (tmp_path / "nodes.yaml").write_text("w:\n  file: w.json\n  params:\n    b: [['1', 'b']]\n")
    with pytest.raises(ValueError, match="no input 'b'"):
        load_workflows(tmp_path)


def test_missing_file_rejected(tmp_path: Path):
    (tmp_path / "nodes.yaml").write_text("w:\n  file: nope.json\n  params: {}\n")
    with pytest.raises(ValueError, match="file not found"):
        load_workflows(tmp_path)
