"""
Phase J — generative security-pattern art tests.

Contract under test (backend/security_pattern):
  - pattern_png(seed) returns valid PNG bytes (magic header)
  - deterministic: same (seed, size) → byte-identical output
  - distinct seeds → distinct output (it actually varies)
  - accepts int / str / weird seeds without raising
  - size is clamped to the safe [MIN_SIZE, MAX_SIZE] window
These run on numpy + PIL only — no model load — so they are fast.
"""

import io

import pytest
from PIL import Image

from backend import security_pattern as sp

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_returns_valid_png():
    data = sp.pattern_png(1, size=256)
    assert isinstance(data, (bytes, bytearray))
    assert data[:8] == _PNG_MAGIC
    # Decodes to an image of the requested size.
    img = Image.open(io.BytesIO(data))
    assert img.size == (256, 256)


def test_deterministic_same_seed():
    a = sp.pattern_png("ABC123", size=300)
    b = sp.pattern_png("ABC123", size=300)
    assert a == b


def test_different_seeds_differ():
    a = sp.pattern_png(1, size=300)
    b = sp.pattern_png(2, size=300)
    assert a != b


def test_string_and_int_seed_supported():
    # Numeric string and the equivalent int should map to the same pattern
    # (intuitive ?seed=42 behaviour).
    assert sp.pattern_png("42", size=200) == sp.pattern_png(42, size=200)
    # A non-numeric string is hashed and still produces a valid PNG.
    assert sp.pattern_png("serial-7AB91", size=200)[:8] == _PNG_MAGIC


def test_size_clamped():
    assert sp.generate_pattern(1, size=10).size == (sp.MIN_SIZE, sp.MIN_SIZE)
    assert sp.generate_pattern(1, size=99999).size == (sp.MAX_SIZE, sp.MAX_SIZE)
    # Garbage size falls back to the default rather than raising.
    assert sp.generate_pattern(1, size="huge").size == (sp.DEFAULT_SIZE, sp.DEFAULT_SIZE)


@pytest.mark.parametrize("seed", [None, "", 0, -5, True, "x" * 500, 2 ** 40])
def test_never_raises_on_weird_seeds(seed):
    data = sp.pattern_png(seed, size=160)
    assert data[:8] == _PNG_MAGIC


def test_seed_to_int_is_stable_and_nonnegative():
    # Stable across calls (SHA-256 based, not process-salted hash()).
    assert sp._seed_to_int("hello") == sp._seed_to_int("hello")
    assert sp._seed_to_int("hello") >= 0
    assert sp._seed_to_int(None) == 0


def test_endpoint_returns_png():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r1 = client.get("/security-pattern", params={"seed": "demo", "size": 256})
    assert r1.status_code == 200
    assert r1.headers["content-type"] == "image/png"
    assert r1.content[:8] == _PNG_MAGIC

    # Determinism + variation through the HTTP layer.
    r1b = client.get("/security-pattern", params={"seed": "demo", "size": 256})
    r2 = client.get("/security-pattern", params={"seed": "other", "size": 256})
    assert r1.content == r1b.content
    assert r1.content != r2.content


def test_endpoint_rejects_out_of_range_size():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    # FastAPI Query(ge=128, le=1200) validation → 422, never a 500.
    r = client.get("/security-pattern", params={"seed": "x", "size": 999999})
    assert r.status_code == 422
