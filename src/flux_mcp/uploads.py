"""
Short-lived upload links

An authenticated MCP caller asks for a link with create_upload_link. The link
carries a random token that allows uploading a few images for a few minutes
and nothing else. The caller also gets a non-secret upload_id, the only
handle that lists what was uploaded (check_upload), so whoever uploads never
learns the stored image ids.

Tokens are kept only as sha256 hashes, in memory: a restart invalidates every
link. Grants stay readable by upload_id for a while after the link closes.
"""

import hashlib
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable

GRANT_RETENTION_S = 24 * 3600
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass
class Grant:
    upload_id: str
    expires_at: float
    max_files: int
    files: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(self.max_files - len(self.files), 0)


def safe_filename(name: str, default: str = "upload") -> str:
    """A plain file name: no directories, no unusual characters"""
    name = UNSAFE_CHARS.sub("_", PurePosixPath(name.replace("\\", "/")).name).strip(" .")
    return name[:120] or default


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class UploadLinks:
    def __init__(self, ttl_s: float, max_links: int,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_links = max_links
        self.clock = clock
        self._by_token: dict[str, Grant] = {}
        self._by_id: dict[str, Grant] = {}

    def create(self, max_files: int) -> tuple[str, Grant]:
        self._purge()
        if len(self._by_token) >= self.max_links:
            raise ValueError("Too many open upload links; wait for some to expire")
        token = secrets.token_urlsafe(32)
        grant = Grant(uuid.uuid4().hex, self.clock() + self.ttl_s, max_files)
        self._by_token[_hash(token)] = grant
        self._by_id[grant.upload_id] = grant
        return token, grant

    def for_token(self, token: str) -> Grant | None:
        """The grant for a usable token: unknown, expired and used-up all give None"""
        self._purge()
        grant = self._by_token.get(_hash(token))
        return grant if grant is not None and grant.remaining > 0 else None

    def record(self, grant: Grant, image_id: str) -> None:
        grant.files.append(image_id)

    def get(self, upload_id: str) -> Grant | None:
        self._purge()
        return self._by_id.get(upload_id)

    def expired(self, grant: Grant) -> bool:
        return self.clock() >= grant.expires_at

    def _purge(self) -> None:
        now = self.clock()
        for key, grant in list(self._by_token.items()):
            if now >= grant.expires_at:
                del self._by_token[key]
        for key, grant in list(self._by_id.items()):
            if now >= grant.expires_at + GRANT_RETENTION_S:
                del self._by_id[key]
