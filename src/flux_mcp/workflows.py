"""
API-format ComfyUI workflows and how to fill them

Each workflow is a JSON file exported in ComfyUI's API format. ``nodes.yaml``
maps parameter names (prompt, width, seed, ...) to the node inputs they set,
so a new model or workflow needs no code change. See workflows/nodes.yaml.
"""

import copy
import json
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import yaml

# (node id, input name)
Target = tuple[str, str]


@dataclass(frozen=True)
class Upsampler:
    """
    Optional prompt-upsampler nodes

    When enabled, the user's request is formatted with ``prompt_template`` and
    written to ``prompt_target``; the graph already routes the upsampler's
    output to the text encoder. When disabled, ``nodes`` are removed and each
    of ``consumers`` gets the raw prompt instead.
    """
    nodes: list[str]
    prompt_target: Target
    consumers: list[Target]
    prompt_template: str = "{prompt}"


@dataclass(frozen=True)
class References:
    """
    Reference-image inputs

    The first image name goes into ``loader``'s ``input``. Each further image
    gets a copy of the loader node, and every consumer input gets a link to
    it: ``{i}`` in a consumer's input name is the image's 0-based index.
    """
    loader: str
    input: str
    consumers: list[Target]
    max: int = 10


@dataclass(frozen=True)
class Workflow:
    """A workflow graph plus the inputs each parameter sets"""
    name: str
    graph: dict[str, Any]
    params: dict[str, list[Target]]
    upsampler: Upsampler | None = None
    references: References | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    def fill(
        self,
        prompt: str,
        enhance_prompt: bool = True,
        images: list[str] | None = None,
        **values: Any,
    ) -> dict[str, Any]:
        """
        Return a copy of the graph with the prompt and parameters set

        Args:
            prompt: The user's request
            enhance_prompt: Run the prompt upsampler (if the workflow has one)
            images: Uploaded reference image names, in order (workflows with references)
            **values: Parameters from nodes.yaml; None means keep the default

        Raises:
            KeyError: A parameter is not defined for this workflow
            ValueError: Wrong number of reference images
        """
        graph = copy.deepcopy(self.graph)
        merged = {**self.defaults, **{k: v for k, v in values.items() if v is not None}}

        for name, value in merged.items():
            if name not in self.params:
                raise KeyError(f"Workflow '{self.name}' has no parameter '{name}'")
            for node, input_name in self.params[name]:
                graph[node]["inputs"][input_name] = value

        if self.references is not None:
            self._add_references(graph, images or [])
        elif images:
            raise ValueError(f"Workflow '{self.name}' takes no reference images")

        up = self.upsampler
        if up is not None and enhance_prompt:
            text = up.prompt_template.format(
                prompt=prompt, aspect=_aspect(merged.get("width"), merged.get("height"))
            )
            node, input_name = up.prompt_target
            graph[node]["inputs"][input_name] = text
        else:
            consumers = up.consumers if up is not None else self.params.get("prompt", [])
            for node, input_name in consumers:
                graph[node]["inputs"][input_name] = prompt
            for node in up.nodes if up is not None else []:
                graph.pop(node, None)

        return graph

    def _add_references(self, graph: dict[str, Any], images: list[str]) -> None:
        refs = self.references
        if not 1 <= len(images) <= refs.max:
            raise ValueError(f"Workflow '{self.name}' takes 1 to {refs.max} reference images")
        graph[refs.loader]["inputs"][refs.input] = images[0]
        next_id = max(int(n) for n in graph if n.isdigit()) + 1
        for i, name in enumerate(images[1:], start=1):
            loader = copy.deepcopy(graph[refs.loader])
            loader["inputs"][refs.input] = name
            loader.setdefault("_meta", {})["title"] = f"Reference image {i + 1}"
            node_id = str(next_id)
            next_id += 1
            graph[node_id] = loader
            for node, input_name in refs.consumers:
                graph[node]["inputs"][input_name.format(i=i)] = [node_id, 0]


def _aspect(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "1:1"
    d = gcd(width, height)
    return f"{width // d}:{height // d}"


def load_workflows(directory: Path) -> dict[str, Workflow]:
    """
    Load every workflow listed in ``directory/nodes.yaml``

    Raises:
        ValueError: nodes.yaml refers to a missing file, node or input
    """
    spec = yaml.safe_load((directory / "nodes.yaml").read_text())
    workflows = {}
    for name, entry in spec.items():
        path = directory / entry["file"]
        if not path.is_file():
            raise ValueError(f"Workflow '{name}': file not found: {path}")
        graph = json.loads(path.read_text())

        params = {p: [_target(t) for t in targets] for p, targets in entry["params"].items()}
        upsampler = None
        if "upsampler" in entry:
            u = entry["upsampler"]
            upsampler = Upsampler(
                nodes=[str(n) for n in u["nodes"]],
                prompt_target=_target(u["prompt_target"]),
                consumers=[_target(t) for t in u["consumers"]],
                prompt_template=u.get("prompt_template", "{prompt}"),
            )

        references = None
        if "references" in entry:
            r = entry["references"]
            references = References(
                loader=str(r["loader"]),
                input=str(r["input"]),
                consumers=[_target(t) for t in r["consumers"]],
                max=int(r.get("max", 10)),
            )

        workflow = Workflow(
            name, graph, params, upsampler, references, entry.get("defaults", {})
        )
        _check(workflow)
        workflows[name] = workflow
    return workflows


def _target(value: list[Any]) -> Target:
    node, input_name = value
    return str(node), str(input_name)


def _check(workflow: Workflow) -> None:
    targets = [t for ts in workflow.params.values() for t in ts]
    if workflow.upsampler is not None:
        targets += [workflow.upsampler.prompt_target, *workflow.upsampler.consumers]
        targets += [(n, None) for n in workflow.upsampler.nodes]
    if workflow.references is not None:
        refs = workflow.references
        targets.append((refs.loader, refs.input))
        targets += [(node, name.format(i=0)) for node, name in refs.consumers]
    for node, input_name in targets:
        if node not in workflow.graph:
            raise ValueError(f"Workflow '{workflow.name}': node {node} not in graph")
        if input_name is not None and input_name not in workflow.graph[node]["inputs"]:
            raise ValueError(
                f"Workflow '{workflow.name}': node {node} has no input '{input_name}'"
            )
