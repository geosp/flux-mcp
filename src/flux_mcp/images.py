"""
Storage for generated images

Each image is saved as <id>.png under the output directory, where <id> is a
random uuid4 hex string. The id doubles as the unguessable part of its URL.

What produced an image (tool, prompt, upsampled caption, seed, ...) is kept in
the PNG itself, as JSON in an iTXt chunk named "flux-mcp", so the file stays
self-describing wherever it is copied. ComfyUI's own chunks are left in place.
"""

import io
import json
import re
import struct
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
METADATA_KEY = "flux-mcp"


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

    def save(self, data: bytes, metadata: dict[str, Any] | None = None) -> StoredImage:
        """Store an image (any format Pillow reads) as PNG, with optional metadata"""
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            if img.format != "PNG":
                out = io.BytesIO()
                img.save(out, format="PNG")
                data = out.getvalue()
        if metadata:
            data = _with_text_chunk(data, METADATA_KEY, json.dumps(metadata, ensure_ascii=False))
        image_id = uuid.uuid4().hex
        path = self.directory / f"{image_id}.png"
        path.write_bytes(data)
        return StoredImage(image_id, path, self.url_for(image_id), width, height)

    def get(self, image_id: str) -> StoredImage | None:
        path = self.path_for(image_id)
        if path is None:
            return None
        with Image.open(path) as img:
            return StoredImage(image_id, path, self.url_for(image_id), img.width, img.height)

    def latest(self) -> StoredImage | None:
        """The most recently stored image"""
        paths = [p for p in self.directory.glob("*.png") if ID_PATTERN.match(p.stem)]
        if not paths:
            return None
        return self.get(max(paths, key=lambda p: p.stat().st_mtime).stem)

    def metadata(self, image: StoredImage) -> dict[str, Any]:
        """The flux-mcp metadata stored in an image, or {} if it has none"""
        with Image.open(image.path) as img:
            raw = getattr(img, "text", {}).get(METADATA_KEY)
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {}

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


def _with_text_chunk(png: bytes, key: str, text: str) -> bytes:
    """Insert an uncompressed UTF-8 iTXt chunk before IEND, without re-encoding the image"""
    body = key.encode("latin-1") + b"\0\0\0\0\0" + text.encode("utf-8")
    chunk = (struct.pack(">I", len(body)) + b"iTXt" + body
             + struct.pack(">I", zlib.crc32(b"iTXt" + body) & 0xFFFFFFFF))
    iend = png.rindex(b"IEND") - 4  # chunk length field precedes the type
    return png[:iend] + chunk + png[iend:]
