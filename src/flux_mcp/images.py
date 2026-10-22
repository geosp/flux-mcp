"""
Storage for generated images

Each image is saved as <id>.png under the output directory, where <id> is a
random uuid4 hex string. The id doubles as the unguessable part of its URL.
"""

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class StoredImage:
    id: str
    path: Path
    url: str
    width: int
    height: int


class ImageStore:
    def __init__(self, directory: Path, public_base_url: str, preview_max_side: int = 1024):
        self.directory = directory
        self.public_base_url = public_base_url.rstrip("/")
        self.preview_max_side = preview_max_side

    def ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes) -> StoredImage:
        """Store an image (any format Pillow reads) as PNG"""
        with Image.open(io.BytesIO(data)) as img:
            image_id = uuid.uuid4().hex
            path = self.directory / f"{image_id}.png"
            if img.format == "PNG":
                path.write_bytes(data)
            else:
                img.save(path, format="PNG")
            return StoredImage(image_id, path, self.url_for(image_id), img.width, img.height)

    def url_for(self, image_id: str) -> str:
        return f"{self.public_base_url}/images/{image_id}.png"

    def path_for(self, image_id: str) -> Path | None:
        """Path of a stored image, or None if the id is malformed or unknown"""
        if not ID_PATTERN.match(image_id):
            return None
        path = self.directory / f"{image_id}.png"
        return path if path.is_file() else None

    def preview(self, image: StoredImage) -> bytes:
        """JPEG copy no larger than preview_max_side on its longest side"""
        with Image.open(image.path) as img:
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                # Show transparent areas as white instead of whatever color they hide
                rgba = img.convert("RGBA")
                img = Image.new("RGB", rgba.size, "white")
                img.paste(rgba, mask=rgba.getchannel("A"))
            else:
                img = img.convert("RGB")
            img.thumbnail((self.preview_max_side, self.preview_max_side))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()
