"""
Bounding boxes for FLUX 3 Klein layout prompts

Boxes are [y_min, x_min, y_max, x_max] on a 0-1000 grid with (0, 0) at the top
left, stretched onto the canvas. Rows are built the way ComfyUI's
Flux3LayoutPrompt builds them: ids obj_1, obj_2, ... and the text (if any)
quoted at the end of the description.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

MAX_BOXES = 20


class Box(BaseModel):
    bbox: Annotated[
        list[Annotated[int, Field(ge=0, le=1000)]],
        Field(
            min_length=4,
            max_length=4,
            description="[y_min, x_min, y_max, x_max] on a 0-1000 grid, (0, 0) = top left",
        ),
    ]
    desc: Annotated[str, Field(min_length=1, max_length=1000,
                               description="What goes in the box: identity, look, colors")]
    text: Annotated[str | None, Field(max_length=500,
                                      description="Exact text to render in the box, if any")] = None

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        y_min, x_min, y_max, x_max = self.bbox
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("bbox must be [y_min, x_min, y_max, x_max] with min < max")
        return self


def layout_rows(boxes: list[Box]) -> list[dict[str, Any]]:
    """Rows for the prompt's element table"""
    if len(boxes) > MAX_BOXES:
        raise ValueError(f"at most {MAX_BOXES} boxes")
    rows = []
    for i, box in enumerate(boxes, start=1):
        desc = f'{box.desc} "{box.text}"'.strip() if box.text else box.desc
        rows.append({"id": f"obj_{i}", "bbox": box.bbox, "desc": desc})
    return rows
