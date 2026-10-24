import pytest

from flux_mcp.uploads import GRANT_RETENTION_S, UploadLinks, safe_filename


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_token_is_stored_only_as_hash(clock):
    links = UploadLinks(ttl_s=600, max_links=10, clock=clock)
    token, grant = links.create(max_files=2)
    assert token not in links._by_token and token not in repr(vars(links))
    assert links.for_token(token) is grant
    assert links.for_token(token + "x") is None and links.for_token("") is None


def test_max_files_closes_link(clock):
    links = UploadLinks(ttl_s=600, max_links=10, clock=clock)
    token, grant = links.create(max_files=2)
    links.record(grant, "a" * 32)
    assert links.for_token(token) is grant and grant.remaining == 1
    links.record(grant, "b" * 32)
    assert links.for_token(token) is None
    assert links.get(grant.upload_id).files == ["a" * 32, "b" * 32]


def test_max_links(clock):
    links = UploadLinks(ttl_s=600, max_links=2, clock=clock)
    links.create(1)
    links.create(1)
    with pytest.raises(ValueError, match="Too many open upload links"):
        links.create(1)
    clock.now += 600
    links.create(1)


def test_expiry_and_retention(clock):
    links = UploadLinks(ttl_s=600, max_links=10, clock=clock)
    token, grant = links.create(max_files=3)
    clock.now += 599
    assert links.for_token(token) is grant and not links.expired(grant)
    clock.now += 1
    assert links.for_token(token) is None and links.expired(grant)
    assert links.get(grant.upload_id) is grant
    clock.now += GRANT_RETENTION_S
    assert links.get(grant.upload_id) is None


@pytest.mark.parametrize("name, expected", [
    ("photo.jpg", "photo.jpg"),
    ("../../etc/passwd", "passwd"),
    ("C:\\Users\\me\\ref image.png", "ref image.png"),
    ("<script>.png", "_script_.png"),
    ("..", "upload"),
    ("", "upload"),
])
def test_safe_filename(name, expected):
    assert safe_filename(name) == expected
