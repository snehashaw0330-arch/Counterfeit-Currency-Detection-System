"""
Phase R.1 — secure-note token (serial -> guilloché binding).

Tested at the module level (backend.security_pattern) so these stay fast and do
NOT load TensorFlow/Keras via backend.main. They assert the core contract:
  - deterministic: same serial -> byte-identical token
  - canonical: formatting variants of the same serial -> identical token
  - unique: different serials -> different tokens
  - valid PNG, size-clamped, never raises (incl. weird/empty input)
  - the verification CODE is stable + serial-derived
"""

import io

from PIL import Image

from backend.security_pattern import (
    normalize_serial,
    serial_token,
    secure_note_token,
    secure_note_png,
    MIN_SIZE,
    MAX_SIZE,
)


# --------------------------------------------------------------------------
# Serial normalization + verification code
# --------------------------------------------------------------------------

def test_normalize_serial_canonicalises_formatting():
    assert normalize_serial("abc 123") == "ABC123"
    assert normalize_serial("ABC-123") == "ABC123"
    assert normalize_serial("  a b c 1 2 3 ") == "ABC123"
    assert normalize_serial("7MP 979885") == "7MP979885"


def test_serial_token_is_deterministic_and_serial_derived():
    code1 = serial_token("ABC 123")
    code2 = serial_token("abc-123")  # same canonical serial
    assert code1 == code2
    assert len(code1) == 8
    assert all(c in "0123456789ABCDEF" for c in code1)
    # different serial -> different code (overwhelmingly likely)
    assert serial_token("ABC123") != serial_token("ABC124")


# --------------------------------------------------------------------------
# Token image determinism / uniqueness
# --------------------------------------------------------------------------

def test_same_serial_byte_identical_token():
    a = secure_note_png("KKL 7MP 979885")
    b = secure_note_png("KKL 7MP 979885")
    assert a == b


def test_formatting_variants_produce_identical_token():
    base = secure_note_png("ABC123")
    assert secure_note_png("abc 123") == base
    assert secure_note_png("ABC-123") == base


def test_different_serials_produce_different_tokens():
    assert secure_note_png("ABC123") != secure_note_png("ABC124")
    assert secure_note_png("AAAA1111") != secure_note_png("BBBB2222")


def test_token_is_valid_png_and_header_taller_than_square():
    png = secure_note_png("ABC123", size=300)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    # square guilloché (300) + a header band => taller than wide
    assert img.width == 300
    assert img.height > img.width


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_size_is_clamped():
    small = secure_note_token("ABC123", size=1)       # below MIN
    big = secure_note_token("ABC123", size=999999)    # above MAX
    assert small.width == MIN_SIZE
    assert big.width == MAX_SIZE


def test_never_raises_on_weird_serials():
    for s in ["", "   ", "!!!", "你好", "0", "x" * 500, "AB\tCD\n12"]:
        png = secure_note_png(s, size=200)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_token_image_is_rgb_encodable_and_nonblank():
    img = secure_note_token("ABC123", size=256)
    assert img.mode == "RGBA"
    # not a flat single-colour image (guilloché + text were drawn)
    extrema = img.convert("L").getextrema()
    assert extrema[0] != extrema[1]
