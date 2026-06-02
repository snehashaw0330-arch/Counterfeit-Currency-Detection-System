"""
Phase J — Generative security-pattern art.

Procedurally generates the kind of intricate, mathematically-defined ornament
that real security printing uses — *guilloché* (woven sinusoidal lattices) and
*rosettes* (layered hypotrochoid / spirograph curves) — plus a micro-text ring.

WHY THIS, AND WHAT IT IS NOT
----------------------------
The user repeatedly asked for "a GenAI that makes notes that can't be cloned."
A model that emits realistic banknote imagery is illegal and is explicitly NOT
built here. What genuinely resists cloning in real currency is the *physics*
(intaglio, optically variable ink, paper) and, on the design side, the
deterministic mathematical complexity of guilloché work — fine interwoven
curves that are trivial to regenerate from a seed but very hard to reproduce by
hand or to copy cleanly through a scan/print cycle.

So this module is a legitimate generative feature:
  - It produces ABSTRACT ornamental art only. No portraits, no denominations,
    no "RBI"/"Reserve Bank" wording, no note-like layout — nothing that imitates
    currency.
  - It is fully DETERMINISTIC: the same seed always yields the same pattern.
    A seed can be derived from a serial number, so a pattern can be regenerated
    and visually compared for verification ("does the printed guilloché match
    the one this serial should produce?").
  - It never raises: on any internal failure it falls back to a simple but still
    deterministic pattern, so the endpoint always returns a valid PNG.

Pure numpy + PIL. No model weights, no network.
"""

import io
import math
import hashlib
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bounds keep the endpoint safe (no giant-canvas DoS) and the output legible.
MIN_SIZE = 128
MAX_SIZE = 1200
DEFAULT_SIZE = 600

# Micro-text drawn around the ring. Deliberately generic — security-themed,
# never currency wording.
_MICROTEXT_BASE = "SECUREPATTERN"


# ----------------------------------------------------------------------------
# Seed handling
# ----------------------------------------------------------------------------

def _seed_to_int(seed) -> int:
    """Map any seed (int / str / None) to a stable non-negative 32-bit int.

    Strings (e.g. a serial number) are hashed with SHA-256 so the mapping is
    deterministic across machines and Python runs (Python's built-in hash() is
    salted per-process and would NOT be reproducible)."""
    if seed is None:
        return 0
    if isinstance(seed, bool):  # bool is an int subclass — normalise first
        seed = int(seed)
    if isinstance(seed, int):
        return abs(seed) % (2 ** 32)
    text = str(seed).strip()
    if text == "":
        return 0
    # Plain integer string → use it directly so ?seed=42 behaves intuitively.
    try:
        return abs(int(text)) % (2 ** 32)
    except ValueError:
        pass
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _clamp_size(size) -> int:
    try:
        size = int(size)
    except (TypeError, ValueError):
        return DEFAULT_SIZE
    return max(MIN_SIZE, min(MAX_SIZE, size))


# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------

def _palette(rng):
    """Two/three muted 'security ink' colours derived from the seed.

    Hues are kept in the teal→green→indigo→maroon range and de-saturated a
    little so the result reads as engraved ink on paper, not neon."""
    base_h = rng.random()
    inks = []
    for offset in (0.0, 0.5, 0.13):  # base, complementary, near-analogous
        h = (base_h + offset) % 1.0
        s = 0.45 + 0.25 * rng.random()
        v = 0.35 + 0.25 * rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        inks.append((int(r * 255), int(g * 255), int(b * 255)))
    return inks


# ----------------------------------------------------------------------------
# Curve primitives
# ----------------------------------------------------------------------------

def _hypotrochoid(R, r, d, samples=4000):
    """Spirograph curve. R/r controls the number of petals; d controls reach.

    n_turns is chosen so the curve closes (r / gcd(R, r) revolutions)."""
    g = math.gcd(int(R), int(r)) or 1
    n_turns = int(r) // g
    t = np.linspace(0.0, 2.0 * np.pi * n_turns, samples)
    k = (R - r) / float(r)
    x = (R - r) * np.cos(t) + d * np.cos(k * t)
    y = (R - r) * np.sin(t) - d * np.sin(k * t)
    return x, y


def _fit_points(x, y, cx, cy, target_radius):
    """Scale a centred curve so its max extent matches target_radius, then
    translate to (cx, cy). Returns a flat list of (px, py) tuples for PIL."""
    extent = float(max(np.max(np.abs(x)), np.max(np.abs(y)))) or 1.0
    scale = target_radius / extent
    px = cx + x * scale
    py = cy + y * scale
    return list(zip(px.tolist(), py.tolist()))


def _guilloche_ring(cx, cy, base_r, amp, freq, phase, samples=2000):
    """A sinusoidally-modulated circle — the basic woven-band element."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    rad = base_r + amp * np.sin(freq * theta + phase)
    x = cx + rad * np.cos(theta)
    y = cy + rad * np.sin(theta)
    return list(zip(x.tolist(), y.tolist()))


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------

def _draw_woven_band(draw, cx, cy, base_r, rng, color, copies=14):
    """Phase-shifted modulated rings → an interwoven guilloché band."""
    amp = base_r * (0.04 + 0.05 * rng.random())
    freq = rng.integers(14, 34)
    for i in range(copies):
        phase = 2.0 * np.pi * i / copies
        pts = _guilloche_ring(cx, cy, base_r, amp, freq, phase)
        draw.line(pts, fill=color + (90,), width=1, joint="curve")


def _draw_rosette(draw, cx, cy, radius, rng, inks):
    """Layered hypotrochoids, rotated and phase-shifted, in cycling inks."""
    layers = int(rng.integers(3, 6))
    for layer in range(layers):
        R = int(rng.integers(90, 150))
        r = int(rng.integers(25, R - 10))
        d = float(rng.integers(20, 80))
        x, y = _hypotrochoid(R, r, d)

        # Rotate the whole curve for this layer so layers interleave.
        ang = 2.0 * np.pi * layer / layers + rng.random()
        xr = x * math.cos(ang) - y * math.sin(ang)
        yr = x * math.sin(ang) + y * math.cos(ang)

        pts = _fit_points(xr, yr, cx, cy, radius * (0.95 - 0.12 * layer))
        color = inks[layer % len(inks)]
        draw.line(pts, fill=color + (140,), width=1, joint="curve")


def _draw_microtext_ring(img, cx, cy, radius, text, color):
    """Tiny repeated text wrapped around a circle (security micro-print motif).

    Each glyph is rendered to a small tile, rotated to be tangent to the
    circle, and pasted. Wrapped so a missing font never breaks the image."""
    try:
        font = ImageFont.load_default()
    except Exception:
        return
    if not text:
        return

    n = max(len(text), 1)
    count = max(n * 3, 48)  # repeat the word a few times around the ring
    for i in range(count):
        ch = text[i % n]
        if ch == " ":
            continue
        angle = 2.0 * np.pi * i / count
        tile = Image.new("RGBA", (14, 14), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text((2, 1), ch, font=font, fill=color + (255,))
        # Rotate so text sits tangent to the circle, reading outward-up.
        deg = -math.degrees(angle) - 90.0
        tile = tile.rotate(deg, expand=True, resample=Image.BICUBIC)
        px = int(cx + radius * math.cos(angle) - tile.width / 2)
        py = int(cy + radius * math.sin(angle) - tile.height / 2)
        img.alpha_composite(tile, (px, py))


def _fallback_pattern(size, seed_int):
    """Dead-simple but deterministic pattern. Used only if the rich generator
    raises, so /security-pattern always returns a valid, seed-stable image."""
    rng = np.random.default_rng(seed_int)
    img = Image.new("RGBA", (size, size), (244, 240, 230, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    cx = cy = size / 2.0
    inks = _palette(rng)
    for i in range(20):
        base_r = size * (0.1 + 0.018 * i)
        pts = _guilloche_ring(cx, cy, base_r,
                              amp=base_r * 0.06,
                              freq=int(rng.integers(10, 26)),
                              phase=rng.random() * 2 * np.pi)
        draw.line(pts, fill=inks[i % len(inks)] + (110,), width=1, joint="curve")
    return img


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def generate_pattern(seed, size=DEFAULT_SIZE) -> Image.Image:
    """Return a deterministic RGBA guilloché/rosette artwork for `seed`.

    Same (seed, size) → byte-identical pixels. Never raises."""
    seed_int = _seed_to_int(seed)
    size = _clamp_size(size)

    try:
        rng = np.random.default_rng(seed_int)
        inks = _palette(rng)

        img = Image.new("RGBA", (size, size), (245, 241, 232, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        cx = cy = size / 2.0

        # Outer woven guilloché frame (a few concentric bands).
        for band, frac in enumerate((0.46, 0.42, 0.385)):
            _draw_woven_band(draw, cx, cy, size * frac, rng,
                             inks[band % len(inks)])

        # Micro-text ring between the frame and the central rosette.
        token = f"{_MICROTEXT_BASE}{seed_int:08X}"
        _draw_microtext_ring(img, cx, cy, size * 0.345, token, inks[0])

        # Central rosette.
        _draw_rosette(draw, cx, cy, size * 0.3, rng, inks)

        # Crisp outer rim.
        m = size * 0.02
        draw.ellipse([m, m, size - m, size - m],
                     outline=inks[0] + (200,), width=2)

        return img
    except Exception:
        return _fallback_pattern(size, seed_int)


def pattern_png(seed, size=DEFAULT_SIZE) -> bytes:
    """Generate the pattern and encode it as PNG bytes. Never raises."""
    img = generate_pattern(seed, size)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
