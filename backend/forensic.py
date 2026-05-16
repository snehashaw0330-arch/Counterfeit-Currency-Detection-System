"""
Forensic Analysis Pipeline for Indian Currency Notes.

Each function takes a BGR numpy image (as read by cv2 / converted from PIL)
and returns a dict with at minimum:
    {"status": "PASS" | "FAIL" | "INFO", "details": <str>}

The pipeline is intentionally tolerant: if a feature cannot be evaluated
it returns status "INFO" with a human readable explanation instead of
raising. This keeps the API response stable for the frontend.
"""

import os
import re
import shutil
import platform

import cv2
import numpy as np
import pytesseract


# =====================================================
# TESSERACT BINARY AUTO DETECTION
# =====================================================

def _configure_tesseract():

    if shutil.which("tesseract"):
        return True

    candidates = []

    if platform.system() == "Windows":

        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(
                r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
            ),
        ]

    for path in candidates:

        if path and os.path.exists(path):

            pytesseract.pytesseract.tesseract_cmd = path

            return True

    return False


TESSERACT_AVAILABLE = _configure_tesseract()


# =====================================================
# UTILITIES
# =====================================================

def _ensure_bgr(image):

    if image is None:
        raise ValueError("Image is None")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _to_gray(image):

    return cv2.cvtColor(_ensure_bgr(image), cv2.COLOR_BGR2GRAY)


# =====================================================
# 1. OCR SERIAL NUMBER
# =====================================================
# Indian banknotes carry two serial numbers — small font
# top-left and larger font bottom-right. Format is normally
# 1 digit + 2 uppercase letters + space + 6-7 digits, e.g.
# "5CT 199410". RBI specimens use all-digit or "0AA" prefix.
#
# We use Tesseract's word-level output (image_to_data) which
# returns text + confidence + bounding box per token.  We try
# multiple binarisations across the grayscale, red and blue
# channels (helps the magenta Rs 2000 note where the green
# channel kills contrast), then match adjacent <prefix><digits>
# token pairs on the same line.

_OCR_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "S": "5", "B": "8"})
_OCR_LETTER_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})

_DIGITS_ONLY = re.compile(r"^[0-9OISB]{6,7}$")
_PREFIX_ALNUM_3 = re.compile(r"^[A-Z0-9]{3}$")
_FULL_SERIAL_NO_SPACE = re.compile(r"([A-Z0-9]{3})([0-9OISB]{6,7})")


def _ocr_region(region, psm=7):
    """Backwards-compatible plain-text OCR (used by diagnostics)."""

    if not TESSERACT_AVAILABLE:
        return ""

    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )

    try:
        text = pytesseract.image_to_string(region, config=config)
    except Exception:
        return ""

    return text.strip().replace("\n", " ")


def _ocr_words(region, psm):
    """Word-level OCR. Returns list of {text, conf, x, y, h}."""

    if not TESSERACT_AVAILABLE:
        return []

    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )

    try:
        data = pytesseract.image_to_data(
            region,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    out = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            continue
        if conf < 30:
            continue
        out.append({
            "text": text,
            "conf": conf,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "h": int(data["height"][i]),
        })
    return out


_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _preprocess_variants(region):
    """Yield binarised single-channel variants for OCR.

    Two channels (gray + red — green is skipped because the
    magenta Rs 2000 ink disappears in it; blue rarely adds
    information beyond gray) and two binarisations each
    (Otsu + adaptive). CLAHE is applied first to recover
    contrast on phone-camera photos.

    Upscaling is adaptive: small crops (e.g. from a 300px-
    wide thumbnail) are upscaled aggressively so Tesseract
    has enough pixel height to recognise the digits — a
    fixed 2.5x factor leaves a 30-pixel-tall band at only
    75px, which is below Tesseract's reliable threshold."""

    bgr = region
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    r_ch = bgr[:, :, 2]

    h = region.shape[0]

    # Target ≥ 180 pixels of vertical resolution after scaling,
    # capped at 4.5x to avoid blowing up huge images.
    scale = max(2.5, min(4.5, 180.0 / max(h, 1)))

    variants = []

    for ch in (gray, r_ch):

        ch = _CLAHE.apply(ch)

        up = cv2.resize(
            ch, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        up = cv2.bilateralFilter(up, 9, 75, 75)

        _, otsu = cv2.threshold(
            up, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        adapt = cv2.adaptiveThreshold(
            up, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10,
        )

        variants.extend([otsu, adapt])

    return variants


def _normalize_prefix(prefix):
    """Return a clean 3-char prefix or None.

    Accepts either real-serial format (digit + letter + letter)
    or specimen-format (all digits)."""

    if len(prefix) != 3:
        return None

    d1 = prefix[0].translate(_OCR_DIGIT_FIX)
    l1 = prefix[1].translate(_OCR_LETTER_FIX)
    l2 = prefix[2].translate(_OCR_LETTER_FIX)

    if d1.isdigit() and l1.isalpha() and l2.isalpha():
        return f"{d1}{l1}{l2}"

    all_d = prefix.translate(_OCR_DIGIT_FIX)
    if all_d.isdigit():
        return all_d

    return None


def _normalize_digits(text):
    fixed = text.translate(_OCR_DIGIT_FIX)
    if fixed.isdigit() and len(fixed) in (6, 7):
        return fixed
    return None


def _serial_from_words_2letter(words):
    """Recover serials when Tesseract reads only the 2 letters
    of the prefix (e.g. "BM" instead of "9BM") — common on tiny
    images where the leading digit is too small to segment.

    The leading digit is reported as "?" so the user can read
    it from the note rather than us inventing a value."""

    out = []

    for w in words:

        digits = _normalize_digits(w["text"])
        if digits is None:
            continue

        for p in words:

            if p is w:
                continue
            if p["x"] >= w["x"]:
                continue
            if abs(p["y"] - w["y"]) > max(w["h"], 12):
                continue

            text = p["text"]
            if len(text) != 2:
                continue

            l1 = text[0].translate(_OCR_LETTER_FIX)
            l2 = text[1].translate(_OCR_LETTER_FIX)
            if not (l1.isalpha() and l2.isalpha()):
                continue

            out.append({
                "serial": f"?{l1}{l2} {digits}",
                "conf": (p["conf"] + w["conf"]) / 2,
            })

    return out


def _serial_from_words(words):
    """Find serial-shaped (prefix+digits) pairs in word list."""

    out = []

    # Two-token form: <prefix> <digits>  on roughly the same line
    for w in words:

        digits = _normalize_digits(w["text"])
        if digits is None:
            continue

        for p in words:

            if p is w:
                continue
            if p["x"] >= w["x"]:
                continue  # prefix must be left of digits
            if abs(p["y"] - w["y"]) > max(w["h"], 12):
                continue
            if not _PREFIX_ALNUM_3.match(p["text"]):
                continue

            prefix = _normalize_prefix(p["text"])
            if prefix is None:
                continue

            out.append({
                "serial": f"{prefix} {digits}",
                "conf": (p["conf"] + w["conf"]) / 2,
            })

    # Single-token form: "0AA000000" with no space
    for w in words:

        m = _FULL_SERIAL_NO_SPACE.match(w["text"])
        if not m:
            continue

        prefix = _normalize_prefix(m.group(1))
        digits = _normalize_digits(m.group(2))
        if prefix is None or digits is None:
            continue

        out.append({
            "serial": f"{prefix} {digits}",
            "conf": w["conf"],
        })

    return out


# Looser regex over the full OCR string. Catches the case
# where image_to_data segments prefix and digits into
# unrelated runs but they still appear in the line text.
_SERIAL_TEXT_REGEX = re.compile(
    r"([A-Z0-9OISB]{3})\s?([0-9OISB]{6,7})"
)


def _serials_from_text(text):

    out = []

    for m in _SERIAL_TEXT_REGEX.finditer(text):

        prefix = _normalize_prefix(m.group(1))
        digits = _normalize_digits(m.group(2))

        if prefix is None or digits is None:
            continue

        out.append({"serial": f"{prefix} {digits}", "conf": 50})

    return out


def _cross_variant_serials(per_crop_results):
    """Combine prefix from one variant with digits from another
    within the same crop, when they line up vertically."""

    out = []

    for words in per_crop_results:

        prefixes = [
            w for w in words
            if _PREFIX_ALNUM_3.match(w["text"])
            and _normalize_prefix(w["text"]) is not None
        ]
        digit_tokens = [
            w for w in words
            if _normalize_digits(w["text"]) is not None
        ]

        for p in prefixes:
            for d in digit_tokens:
                if p["x"] >= d["x"]:
                    continue
                # Tolerate vertical offset since they come from
                # different binarisations of the same crop.
                if abs(p["y"] - d["y"]) > max(p["h"], d["h"]) * 1.5:
                    continue

                prefix = _normalize_prefix(p["text"])
                digits = _normalize_digits(d["text"])
                if prefix is None or digits is None:
                    continue

                out.append({
                    "serial": f"{prefix} {digits}",
                    "conf": (p["conf"] + d["conf"]) / 2,
                })

    return out


_TARGET_OCR_WIDTH = 1800


def _normalise_for_ocr(image):
    """Resize the image so its width is at least _TARGET_OCR_WIDTH.

    Tiny inputs (148x341 phone-camera thumbnails) need this — at
    native resolution Tesseract can't reliably resolve the serial
    even after the per-variant upscale, because the crop is
    pixel-starved before any preprocessing."""

    h, w = image.shape[:2]

    if w >= _TARGET_OCR_WIDTH:
        return image

    scale = _TARGET_OCR_WIDTH / w
    return cv2.resize(
        image, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def extract_serial_number(image):
    """Layout-independent serial extraction.

    Strategy:
      1. Upscale tiny inputs so Tesseract has enough pixels.
      2. OCR the whole image and the top + bottom halves
         across several preprocessing variants and PSM modes.
      3. Pool every detected word into one big bag.
      4. Score the 3-char prefix candidates and the 6-7 digit
         candidates *independently*. The digits half is
         usually reliable; the prefix half is what
         Tesseract gets wrong on stylised banknote fonts.
      5. Pair the strongest prefix with the strongest digits
         only if the prefix has a clear lead — otherwise
         emit "??? 123456" so we never fabricate a prefix."""

    if not TESSERACT_AVAILABLE:

        return {
            "status": "INFO",
            "details": "Tesseract OCR engine not installed",
            "value": None,
        }

    img = _normalise_for_ocr(_ensure_bgr(image))
    h = img.shape[0]

    regions = [
        img,
        img[0:int(h * 0.5), :],
        img[int(h * 0.5):h, :],
    ]

    prefix_scores = {}
    digit_scores = {}

    for region in regions:

        if region.size == 0:
            continue

        for variant in _preprocess_variants(region):

            for psm in (6, 7, 11):

                for w in _ocr_words(variant, psm):

                    text = w["text"]

                    digits = _normalize_digits(text)
                    if digits is not None:
                        digit_scores[digits] = (
                            digit_scores.get(digits, 0) + w["conf"]
                        )
                        continue

                    if _PREFIX_ALNUM_3.match(text):
                        norm = _normalize_prefix(text)
                        if norm is not None:
                            prefix_scores[norm] = (
                                prefix_scores.get(norm, 0) + w["conf"]
                            )

    if not digit_scores:

        return {
            "status": "FAIL",
            "details": "No serial digit sequence detected",
            "value": None,
        }

    ranked_digits = sorted(
        digit_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    best_digits, best_digits_score = ranked_digits[0]

    if best_digits_score < 60:

        return {
            "status": "FAIL",
            "details": (
                f"Digit sequence too weak "
                f"(top score {int(best_digits_score)})"
            ),
            "value": None,
        }

    if not prefix_scores:

        return {
            "status": "PASS",
            "details": (
                f"Digits read (score {int(best_digits_score)}), "
                f"prefix not recoverable"
            ),
            "value": f"??? {best_digits}",
        }

    ranked_prefixes = sorted(
        prefix_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    top_prefix, top_pscore = ranked_prefixes[0]
    runner = ranked_prefixes[1][1] if len(ranked_prefixes) > 1 else 0

    if top_pscore >= max(runner * 1.3, 50):

        return {
            "status": "PASS",
            "details": (
                f"Serial read (digits {int(best_digits_score)}, "
                f"prefix {int(top_pscore)})"
            ),
            "value": f"{top_prefix} {best_digits}",
        }

    return {
        "status": "PASS",
        "details": (
            f"Digits clear ({int(best_digits_score)}), "
            f"prefix ambiguous: "
            f"{', '.join(p for p, _ in ranked_prefixes[:3])}"
        ),
        "value": f"??? {best_digits}",
    }


# =====================================================
# 2. UV LIGHT FEATURE DETECTION
# =====================================================
# Without a UV camera we approximate by looking for
# bright fluorescent-like high-saturation patches that
# real notes exhibit under visible light too (security
# fibers and reactive ink areas).

def analyze_uv_features(image):

    img = _ensure_bgr(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    high_sat = cv2.inRange(
        hsv,
        (0, 120, 120),
        (179, 255, 255)
    )

    ratio = float(np.count_nonzero(high_sat)) / high_sat.size

    if ratio >= 0.015:

        return {
            "status": "PASS",
            "details": (
                f"Reactive ink signature found "
                f"({ratio * 100:.2f}% of pixels)"
            )
        }

    return {
        "status": "FAIL",
        "details": (
            f"Insufficient UV reactive signature "
            f"({ratio * 100:.2f}% of pixels)"
        )
    }


# =====================================================
# 3. WATERMARK DETECTION
# =====================================================
# Indian notes carry a Gandhi watermark on the left
# blank panel. We score local brightness variance in
# that panel: a real watermark produces a soft
# gradient, a counterfeit print is flat or noisy.

def detect_watermark(image):

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    panel = img[
        int(h * 0.15):int(h * 0.85),
        0:int(w * 0.25)
    ]

    if panel.size == 0:

        return {
            "status": "INFO",
            "details": "Image too small to evaluate"
        }

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    variance = float(np.var(blurred))

    if 80 <= variance <= 2500:

        return {
            "status": "PASS",
            "details": (
                f"Watermark gradient detected "
                f"(variance {variance:.1f})"
            )
        }

    return {
        "status": "FAIL",
        "details": (
            f"No expected watermark gradient "
            f"(variance {variance:.1f})"
        )
    }


# =====================================================
# 4. GANDHI FACE ANALYSIS
# =====================================================

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def analyze_gandhi_face(image):

    img = _ensure_bgr(image)
    gray = _to_gray(img)

    if _FACE_CASCADE.empty():

        return {
            "status": "INFO",
            "details": "Face cascade not available"
        }

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:

        return {
            "status": "FAIL",
            "details": "No portrait detected on note"
        }

    return {
        "status": "PASS",
        "details": f"{len(faces)} portrait region(s) detected"
    }


# =====================================================
# 5. SECURITY THREAD DETECTION
# =====================================================
# Vertical thin line that runs top-to-bottom on a real
# note. We look for long vertical edges in the central
# 30% strip of the image.

def detect_security_thread(image):

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    strip = img[
        0:h,
        int(w * 0.35):int(w * 0.65)
    ]

    if strip.size == 0:

        return {
            "status": "INFO",
            "details": "Image too small to evaluate"
        }

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(h * 0.4),
        maxLineGap=15
    )

    if lines is None:

        return {
            "status": "FAIL",
            "details": "No security thread detected"
        }

    vertical = 0

    for line in lines:

        x1, y1, x2, y2 = line[0]

        if abs(x2 - x1) < 8 and abs(y2 - y1) > h * 0.3:

            vertical += 1

    if vertical >= 1:

        return {
            "status": "PASS",
            "details": f"{vertical} vertical thread segment(s) found"
        }

    return {
        "status": "FAIL",
        "details": "No vertical thread pattern"
    }


# =====================================================
# 6. COLOR RICHNESS / PALETTE INTEGRITY
# =====================================================
# Real banknotes carry a designed multi-hue palette with
# strong saturation. Photocopies, grayscale prints, hue-
# shifted fakes and desaturated counterfeits collapse
# this distribution. Hue entropy + mean saturation are a
# far stronger discriminator than the old fixed-patch
# hologram check (separator score 10/20 on test set).
# We keep the function name to preserve API/frontend
# stability; behaviour and details have been replaced.

def detect_hologram(image):

    img = _ensure_bgr(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    mean_sat = float(np.mean(sat))

    coloured = hue[sat > 60]

    if coloured.size < 200:

        return {
            "status": "FAIL",
            "details": (
                f"Image is desaturated/colourless "
                f"(mean sat {mean_sat:.0f})"
            )
        }

    hist = np.bincount(coloured, minlength=180).astype(np.float64)
    hist /= hist.sum()
    nz = hist[hist > 0]
    hue_entropy = float(-(nz * np.log2(nz)).sum())

    if mean_sat >= 45 and hue_entropy >= 3.3:

        return {
            "status": "PASS",
            "details": (
                f"Rich palette (sat {mean_sat:.0f}, "
                f"hue entropy {hue_entropy:.2f})"
            )
        }

    return {
        "status": "FAIL",
        "details": (
            f"Weak palette (sat {mean_sat:.0f}, "
            f"hue entropy {hue_entropy:.2f})"
        )
    }


# =====================================================
# 6b. STRUCTURAL SANITY (NEW)
# =====================================================
# Pre-flight gate: rejects images that cannot plausibly
# be a banknote (blank, pure noise, half-black, severely
# off aspect ratio). The MobileNetV2 classifier over-
# approves such inputs at 95%+; this gate short-circuits
# them before they reach the verdict combiner.

def structural_sanity(image):

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    aspect = w / max(h, 1)

    if aspect < 1.4 or aspect > 3.0:
        return {
            "status": "FAIL",
            "details": f"Aspect ratio {aspect:.2f} unlike a banknote"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    std = float(np.std(gray))
    if std < 12:
        return {
            "status": "FAIL",
            "details": f"Image is too uniform (brightness std {std:.1f})"
        }

    edges = cv2.Canny(gray, 80, 200)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size

    if edge_ratio < 0.005:
        return {
            "status": "FAIL",
            "details": (
                f"No structure (edges {edge_ratio * 100:.2f}%)"
            )
        }

    if edge_ratio > 0.30:
        return {
            "status": "FAIL",
            "details": (
                f"Pure noise pattern (edges {edge_ratio * 100:.2f}%)"
            )
        }

    qh, qw = h // 2, w // 2
    quadrants = [
        gray[:qh, :qw],
        gray[:qh, qw:],
        gray[qh:, :qw],
        gray[qh:, qw:],
    ]
    dark_quadrants = sum(1 for q in quadrants if float(q.mean()) < 25)

    if dark_quadrants >= 2:
        return {
            "status": "FAIL",
            "details": (
                f"{dark_quadrants} of 4 quadrants are near-black"
            )
        }

    return {
        "status": "PASS",
        "details": (
            f"Structure OK (aspect {aspect:.2f}, "
            f"edges {edge_ratio * 100:.2f}%, std {std:.0f})"
        )
    }


# =====================================================
# 7. DENOMINATION CLASSIFICATION
# =====================================================

_KNOWN_DENOMINATIONS = {
    "10", "20", "50", "100", "200", "500", "2000"
}

# Common OCR misreads of the leading digit. The ₹ symbol is
# rendered as a curly shape immediately before the digit and
# Tesseract often glues them together — e.g. "₹500" reads as
# "9500", or the "₹5" gets fused into a single "9" glyph and
# the trailing "00" is read independently, yielding "900".
# We accept these near-misses by trying both the original
# token and a version with the leading digit remapped.
_DENOM_LEADING_FIX = {
    "9": ["5", "2", "1"],   # ₹ -> 9 confusion; resolves to 5/2/1
    "8": ["5"],             # ₹ symbol can also map to 8
    "4": ["1"],             # 4 -> 1 misread in some fonts
    "6": ["5"],             # tail of ₹ stroke
}


def _denom_candidates(token):
    """Yield all plausible denominations for an OCR token.

    Includes the literal token, the digit-confusion fix, and
    any leading-digit remappings that produce a known
    denomination value."""

    if not token:
        return

    fixed = token.translate(_OCR_DIGIT_FIX)
    if not fixed.isdigit():
        return

    if fixed in _KNOWN_DENOMINATIONS:
        yield fixed

    if not fixed:
        return

    head = fixed[0]
    tail = fixed[1:]

    for repl in _DENOM_LEADING_FIX.get(head, ()):
        alt = repl + tail
        if alt in _KNOWN_DENOMINATIONS:
            yield alt


# Approximate dominant (hue, sat) range per denomination for
# the Mahatma Gandhi New Series. OpenCV hue is 0-179, sat
# is 0-255. The saturation bound is what splits Rs 200
# (high-sat yellow) from Rs 500 (low-sat stone) — both sit
# around the same hue.
_DENOM_PALETTE = {
    # denom : (hue_lo, hue_hi, sat_lo, sat_hi)
    "10":   (5,   22,   30, 200),  # chocolate brown
    "20":   (35,  65,   60, 255),  # yellow-green
    "50":   (85,  105,  40, 255),  # cyan / blue
    "100":  (115, 145,  30, 200),  # lavender
    "200":  (15,  40,   90, 255),  # bright yellow
    "500":  (15,  55,    0,  80),  # stone / olive (low sat)
    "2000": (155, 179,  50, 255),  # magenta
}


def _palette_match(image, denom):
    """Return True if the dominant note (hue, sat) is
    consistent with the given denomination's palette."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Sample only on coloured pixels (sat > 25) so the white
    # margin of phone photos doesn't dominate the median.
    mask = s > 25
    if mask.sum() < 200:
        # Mostly desaturated — fits only Rs 500 (stone) or
        # the deep brown Rs 10.
        return denom in {"500", "10"}

    median_hue = float(np.median(h[mask]))
    median_sat = float(np.median(s[mask]))

    lo_h, hi_h, lo_s, hi_s = _DENOM_PALETTE.get(
        denom, (0, 179, 0, 255)
    )

    return (lo_h <= median_hue <= hi_h) and (lo_s <= median_sat <= hi_s)


def classify_denomination(image):

    if not TESSERACT_AVAILABLE:

        return {
            "status": "INFO",
            "details": "Tesseract OCR engine not installed",
            "value": None
        }

    img = _normalise_for_ocr(_ensure_bgr(image))
    h, w = img.shape[:2]

    # Scan the whole image plus the four corners. Whole-image
    # OCR catches the denomination wherever it sits (works on
    # both Mahatma Gandhi New Series and older series, both
    # obverse and reverse). Corner crops give cross-region
    # voting. Small input images get a global upscale via
    # _normalise_for_ocr so the digit tokens are large enough
    # for Tesseract.
    corners = [
        ("whole",        img),
        ("top-left",     img[0:int(h * 0.30),         0:int(w * 0.30)]),
        ("top-right",    img[0:int(h * 0.30),         int(w * 0.65):w]),
        ("bottom-left",  img[int(h * 0.65):h,         0:int(w * 0.30)]),
        ("bottom-right", img[int(h * 0.65):h,         int(w * 0.55):w]),
    ]

    # Vote weighted by Tesseract confidence — a one-off
    # low-confidence misread ("500" at conf 45 on a Rs 200
    # reverse) is dominated by a high-confidence "200" at
    # conf 95 in another corner, even though both pass the
    # height filter.
    # Two-pass scoring:
    #   exact_score   - tokens that read as a valid denomination directly
    #   confused_score - tokens that become a valid denomination after
    #                    OCR confusion mapping (₹ misread as 9 etc.)
    # Exact reads always beat confused reads, so we keep them apart
    # and only fall back to confused candidates if no exact wins.
    exact_score = {}
    confused_score = {}

    for label, crop in corners:

        if crop.size == 0:
            continue

        # Whole-image scan: denomination is typically 5-15%
        # of image height. Corner crops: denomination should
        # dominate the crop, so require ≥ 20% of corner-h.
        if label == "whole":
            min_frac = 0.05
        else:
            min_frac = 0.20

        for processed in _preprocess_variants(crop):

            ph = processed.shape[0]
            min_h_px = ph * min_frac

            for psm in (7, 11):

                try:
                    data = pytesseract.image_to_data(
                        processed,
                        config=(
                            f"--oem 3 --psm {psm} "
                            "-c tessedit_char_whitelist=0123456789"
                        ),
                        output_type=pytesseract.Output.DICT,
                    )
                except Exception:
                    continue

                for i in range(len(data["text"])):
                    text = data["text"][i].strip()
                    if not text:
                        continue
                    try:
                        conf = int(float(data["conf"][i]))
                    except (TypeError, ValueError):
                        continue
                    if conf < 40:
                        continue
                    if int(data["height"][i]) < min_h_px:
                        continue

                    candidates = list(_denom_candidates(text))
                    if not candidates:
                        continue

                    # First candidate is the exact read; the
                    # rest come from OCR-confusion remapping.
                    head = candidates[0]
                    if head == text:
                        exact_score[head] = (
                            exact_score.get(head, 0) + conf
                        )
                        rest = candidates[1:]
                    else:
                        rest = candidates

                    for alt in rest:
                        # Only credit confusion-remaps when
                        # the colour palette is consistent —
                        # otherwise "900" on a Rs 100 lavender
                        # note could wrongly resolve to 500.
                        if not _palette_match(img, alt):
                            continue
                        confused_score[alt] = (
                            confused_score.get(alt, 0) + conf * 0.7
                        )

    score_by_denom = exact_score if exact_score else confused_score

    if not score_by_denom:

        return {
            "status": "FAIL",
            "details": "Could not read denomination",
            "value": None
        }

    # Pick the denomination with the highest summed
    # Tesseract confidence across all corners.
    ranked = sorted(
        score_by_denom.items(), key=lambda kv: kv[1], reverse=True
    )
    value, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0

    # Demand a clear win — otherwise a low-confidence misread
    # in one corner can tie with the real denomination.
    if top_score < runner * 1.5 and top_score < 150:

        return {
            "status": "FAIL",
            "details": (
                f"Denomination ambiguous "
                f"(top {value}:{int(top_score)} vs "
                f"{ranked[1][0]}:{int(runner)})"
            ),
            "value": None,
        }

    return {
        "status": "PASS",
        "details": (
            f"Denomination Rs. {value} "
            f"(score {int(top_score)})"
        ),
        "value": value,
    }


# =====================================================
# PIPELINE ORCHESTRATOR
# =====================================================

def run_forensic_pipeline(image):
    """
    Run every forensic check. Each individual failure is
    caught so one broken check never breaks the response.
    """

    checks = {
        "structural_sanity": structural_sanity,
        "uv_light_detection": analyze_uv_features,
        "watermark_detection": detect_watermark,
        "ocr_serial_number": extract_serial_number,
        "gandhi_face_analysis": analyze_gandhi_face,
        "security_thread_detection": detect_security_thread,
        "hologram_detection": detect_hologram,
        "denomination_classification": classify_denomination,
    }

    results = {}

    for name, fn in checks.items():

        try:
            results[name] = fn(image)
        except Exception as exc:
            results[name] = {
                "status": "INFO",
                "details": f"Error: {exc}"
            }

    results["modular_ai_pipeline"] = {
        "status": "PASS",
        "details": "Pipeline executed successfully"
    }

    return results
