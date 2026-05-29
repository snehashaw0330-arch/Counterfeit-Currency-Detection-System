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
from collections import Counter

import cv2
import numpy as np
import pytesseract

# EasyOCR is heavy (pulls PyTorch) but it's *significantly*
# more accurate than Tesseract on Indian banknote fonts. We
# load it lazily so test code that doesn't need OCR (forensic
# unit tests, structural checks) doesn't pay the import cost.
try:
    import easyocr  # noqa: F401
    _EASYOCR_IMPORT_OK = True
except Exception:
    _EASYOCR_IMPORT_OK = False

_EASYOCR_READER = None

# Reused by both call sites. EasyOCR ignores characters
# outside this set, so the ₹ glyph / Hindi script can't
# pollute the candidate token bag.
_ALNUM_SPACE = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 *"
)
# Asterisk is included because RBI marks replacement notes
# (substituted for damaged ones in a sheet) with a * between
# the prefix and the digits, e.g. "2DA*012720". Without it
# in the allowlist EasyOCR substitutes 'X' for the glyph and
# the downstream regex fails to recognise the serial.
_DIGITS_ALLOWLIST = "0123456789"


def _get_easyocr_reader():

    global _EASYOCR_READER

    if not _EASYOCR_IMPORT_OK:
        return None

    if _EASYOCR_READER is None:
        try:
            _EASYOCR_READER = easyocr.Reader(
                ["en"], gpu=False, verbose=False
            )
        except Exception:
            return None

    return _EASYOCR_READER


def warmup_ocr():
    """Force the EasyOCR reader to initialise.

    First-call latency is ~3 s (model load) plus a one-time
    ~64 MB download on the very first ever run. Calling this
    at FastAPI startup pre-pays that cost so the first user
    request doesn't hang. Safe to call repeatedly.

    Returns True on success, False if EasyOCR is unavailable
    or initialisation failed."""

    reader = _get_easyocr_reader()
    if reader is None:
        return False

    # Touch the model with a small noisy image so the lazy
    # PyTorch graph compiles now rather than on the first
    # real call. Failure here is non-fatal — the Reader()
    # constructor is what costs ~3 s, and that already
    # succeeded above.
    try:
        dummy = np.random.randint(
            0, 255, (64, 128, 3), dtype=np.uint8
        )
        reader.readtext(dummy, allowlist=_DIGITS_ALLOWLIST, detail=0)
    except Exception:
        pass

    return True


EASYOCR_AVAILABLE = _EASYOCR_IMPORT_OK


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
# AUTO ORIENT (Phase A polish)
# =====================================================
# Indian banknotes are landscape (~2.2:1). When users hold
# their phone vertically and shoot a note, the resulting
# image is portrait (H > W). The structural_sanity gate
# then hard-fails on aspect ratio < 1.4 and the verdict
# combiner force-marks the input as FAKE — even though
# the note inside is real.
#
# We rotate the whole image 90° clockwise when it's
# portrait so every downstream geometry check sees a
# banknote-shaped frame.

def _auto_orient(image):

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img


# =====================================================
# AUTO-CROP THE NOTE FROM BACKGROUND (Phase B)
# =====================================================
# Phone photos often include desk, hand, table edges, sky.
# The note can occupy as little as 30% of the frame. We
# detect the largest banknote-shaped quad in the image and
# perspective-rectify it to a canonical landscape crop so
# every forensic check operates on the note alone.

_BANKNOTE_ASPECT = 2.2  # Indian notes are roughly 2.2:1
_BANKNOTE_ASPECT_LO = 1.6
_BANKNOTE_ASPECT_HI = 3.2


# Canonical RBI banknote dimensions (mm) for the Mahatma
# Gandhi New Series. Source: RBI banknote specifications.
# Used by analyze_proportions to detect digital stretching
# or wrong-size paper counterfeits.
_RBI_DIMENSIONS_MM = {
    "10":   (123, 63),
    "20":   (129, 63),
    "50":   (135, 66),
    "100":  (142, 66),
    "200":  (146, 66),
    "500":  (150, 66),
    "2000": (166, 66),
}

_RBI_EXPECTED_ASPECT = {
    denom: w / h for denom, (w, h) in _RBI_DIMENSIONS_MM.items()
}

# Tolerance for the proportion check. Phone perspective,
# auto-crop rounding, Wikipedia scan-aspect drift, and frame-
# padding on specimen images all introduce up to ~12% of
# honest error on genuine notes. Tolerance set at 15% — flags
# clear digital stretching (typically 25%+) without false-
# positiving real uploads.
_PROPORTION_TOLERANCE_PCT = 15.0


def _order_quad(pts):
    """Return 4 corner points ordered [top-left, top-right, bottom-right, bottom-left]."""

    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_note_quad(image):
    """Find the largest banknote-shaped 4-sided contour in the
    image. Returns a dict with `quad` (4×2 ordered TL/TR/BR/BL),
    `aspect` (long/short edge ratio), `avg_w`, `avg_h`, or
    None if no plausible quad is found.

    Pure detection — does not warp. Used by both `_locate_note`
    (which then applies the perspective transform) and
    `analyze_proportions` (which compares aspect to the canonical
    RBI dimensions for the OCR'd denomination)."""

    try:
        img = _ensure_bgr(image)
        h, w = img.shape[:2]
        img_area = h * w

        if img_area < 10_000:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(
            edges, np.ones((3, 3), np.uint8), iterations=2
        )

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contours = sorted(
            contours, key=cv2.contourArea, reverse=True
        )

        for contour in contours[:6]:

            area = cv2.contourArea(contour)
            if area < img_area * 0.15:
                break
            if area > img_area * 0.97:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            quad = _order_quad(approx.reshape(4, 2))
            tl, tr, br, bl = quad

            width_top = float(np.linalg.norm(tr - tl))
            width_bot = float(np.linalg.norm(br - bl))
            height_left = float(np.linalg.norm(bl - tl))
            height_right = float(np.linalg.norm(br - tr))

            avg_w = (width_top + width_bot) / 2.0
            avg_h = (height_left + height_right) / 2.0

            long_edge = max(avg_w, avg_h)
            short_edge = min(avg_w, avg_h)
            if short_edge < 1:
                continue

            aspect = long_edge / short_edge
            if (
                aspect < _BANKNOTE_ASPECT_LO
                or aspect > _BANKNOTE_ASPECT_HI
            ):
                continue

            return {
                "quad": quad,
                "aspect": aspect,
                "avg_w": avg_w,
                "avg_h": avg_h,
            }

        return None

    except Exception:
        return None


def _locate_note(image):
    """Find the banknote region and return a rectified landscape crop.

    Uses `_detect_note_quad` for detection then perspective-
    transforms to a canonical landscape rectangle.
    Returns the original image unchanged when no plausible quad
    is found. Never raises."""

    try:
        img = _ensure_bgr(image)
        detection = _detect_note_quad(img)
        if detection is None:
            return img

        quad = detection["quad"]
        aspect = detection["aspect"]
        avg_w = detection["avg_w"]
        avg_h = detection["avg_h"]
        tl, tr, br, bl = quad

        long_edge = max(avg_w, avg_h)

        # Build the canonical landscape destination. Keep the
        # long edge as the new width and derive height from the
        # actual quad aspect (don't force exactly 2.2 — older /
        # damaged notes can be ±10%).
        target_w = int(long_edge)
        target_h = int(long_edge / aspect)
        if target_w < 200 or target_h < 80:
            return img

        # If the quad is portrait (avg_h > avg_w), reorder the
        # source corners so the warp produces a landscape
        # rectangle. Rotating the source ordering by 90° CW
        # maps [TL,TR,BR,BL] → [BL,TL,TR,BR].
        if avg_h > avg_w:
            src = np.array([bl, tl, tr, br], dtype=np.float32)
        else:
            src = np.array([tl, tr, br, bl], dtype=np.float32)

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            img, transform, (target_w, target_h)
        )

    except Exception:
        return _ensure_bgr(image)


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

_OCR_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "J": "1", "S": "5", "B": "8"})
_OCR_LETTER_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})

_DIGITS_ONLY = re.compile(r"^[0-9OISBJ]{6,7}$")
_PREFIX_ALNUM_3 = re.compile(r"^[A-Z0-9]{3}$")
# The optional middle group captures the RBI replacement-note
# asterisk (* between prefix and digits). Whether it matched
# is preserved through to the formatted output so the user
# sees exactly what is printed on the note.
_FULL_SERIAL_NO_SPACE = re.compile(
    r"([A-Z0-9]{3})(\*?)([0-9OISBJ]{6,7})"
)


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

    # Single-token form: "0AA000000" with no space, or
    # "2DA*012720" for RBI replacement notes.
    for w in words:

        m = _FULL_SERIAL_NO_SPACE.match(w["text"])
        if not m:
            continue

        prefix = _normalize_prefix(m.group(1))
        digits = _normalize_digits(m.group(3))
        if prefix is None or digits is None:
            continue

        sep = "*" if m.group(2) else " "
        out.append({
            "serial": f"{prefix}{sep}{digits}",
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


def _easyocr_words(image):
    """Run EasyOCR and return Tesseract-shaped word dicts.

    Each entry has the same {text, conf, x, y, h} shape the
    existing `_serial_from_words` helper expects. Confidence
    is rescaled from EasyOCR's 0..1 float to Tesseract's
    0..100 int range so downstream scoring code is unchanged.

    Returns None if EasyOCR is unavailable (caller falls back
    to Tesseract). Returns [] if EasyOCR ran but found nothing."""

    reader = _get_easyocr_reader()
    if reader is None:
        return None

    img = _ensure_bgr(image)

    try:
        results = reader.readtext(
            img, allowlist=_ALNUM_SPACE, detail=1
        )
    except Exception:
        return None

    words = []
    for bbox, text, conf in results:

        text = text.strip().upper()
        if not text:
            continue
        # Permissive floor: specimen notes with "0AA 000000"
        # serials and tilted phone photos can hand back the
        # right serial token at conf 0.20-0.30. The strict
        # regex in extract_serial_number (3 alnum + optional
        # asterisk + 6-7 digit-like chars) does the real
        # filtering — keeping low-conf tokens just gives it
        # more raw material to match against.
        if conf < 0.20:
            continue

        # bbox is [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_left = int(min(xs))
        y_top = int(min(ys))
        y_bot = int(max(ys))

        words.append({
            "text": text,
            "conf": int(conf * 100),
            "x": x_left,
            "y": (y_top + y_bot) // 2,
            "h": max(y_bot - y_top, 1),
        })

    return words


def extract_serial_number(image):
    """Extract an Indian banknote serial via EasyOCR.

    EasyOCR handles the stylised banknote fonts and the ₹
    glyph that Tesseract misreads on Indian notes.

    Strategy:
      1. Get word-level OCR results from EasyOCR.
      2. Match single-token serials (e.g. "9BM 000793" comes
         back as one bbox with an internal space; "9BM000793"
         as one bbox without). Both forms handled.
      3. Match two-token <prefix> <digits> pairs on the same
         horizontal line for cases where the recogniser
         split prefix and digits into separate bboxes.
      4. Vote by occurrence count first (real banknotes
         carry the serial twice — top-left small and
         bottom-right large), tiebreak by summed confidence.

    Falls through to `_extract_serial_number_tesseract` only
    when EasyOCR is unavailable (import or model-load
    failure)."""

    words = _easyocr_words(image)
    if words is None:
        return _extract_serial_number_tesseract(image)

    if not words:
        return {
            "status": "FAIL",
            "details": "No readable text on note",
            "value": None,
        }

    candidates = []  # list of (serial_string, conf)

    # Single-bbox form: try the full serial regex against the
    # space-stripped text. Catches both "9BM 000793" and
    # "9BM000793" coming back as one EasyOCR token, plus
    # "2DA*012720" for RBI replacement notes.
    for w in words:
        compact = w["text"].replace(" ", "")
        m = _FULL_SERIAL_NO_SPACE.match(compact)
        if not m:
            continue
        prefix = _normalize_prefix(m.group(1))
        digits = _normalize_digits(m.group(3))
        if prefix and digits:
            sep = "*" if m.group(2) else " "
            candidates.append((f"{prefix}{sep}{digits}", w["conf"]))

    # Two-bbox form: prefix and digits in separate tuples
    # on the same horizontal band. Existing helper does the
    # spatial pairing.
    for s in _serial_from_words(words):
        candidates.append((s["serial"], int(s["conf"])))

    if not candidates:
        return {
            "status": "FAIL",
            "details": "No serial-shaped tokens detected",
            "value": None,
        }

    # Vote: occurrence count first, tiebreak by summed conf.
    counts = Counter(s for s, _ in candidates)
    conf_sum = {}
    for s, c in candidates:
        conf_sum[s] = conf_sum.get(s, 0) + c

    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], -conf_sum[kv[0]])
    )
    best_serial, best_count = ranked[0]

    return {
        "status": "PASS",
        "details": (
            f"Serial read via EasyOCR "
            f"({best_count}x, sum conf {conf_sum[best_serial]})"
        ),
        "value": best_serial,
    }


def _extract_serial_number_tesseract(image):
    """Legacy Tesseract path. Kept as silent fallback for
    deployments without PyTorch/EasyOCR available.

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

    # Prefer 6-digit readings (standard Indian banknote serial
    # length); fall back to 7-digit only if nothing 6-digit is
    # available. Avoids picking up "2000000" — a fragment that
    # glued the denomination numeral onto serial digits.
    six = {k: v for k, v in digit_scores.items() if len(k) == 6}
    seven = {k: v for k, v in digit_scores.items() if len(k) == 7}
    active = six if six else seven

    ranked_digits = sorted(
        active.items(), key=lambda kv: kv[1], reverse=True
    )
    best_digits, best_digits_score = ranked_digits[0]

    if best_digits_score < 40:

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
    """Visible-light proxy for UV-reactive ink.

    True UV detection needs a UV lamp; on phone photos we
    approximate by looking for the high-saturation patches
    that real notes carry under visible light. This is a
    weak signal — many real phone photos have 0% qualifying
    pixels (small notes, poor lighting, JPEG compression).
    To avoid penalising real notes for a signal we can't
    reliably measure, we return INFO (not FAIL) when below
    threshold. PASS still credits notes with clear high-
    sat patches."""

    img = _ensure_bgr(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    high_sat = cv2.inRange(
        hsv,
        (0, 120, 120),
        (179, 255, 255)
    )

    ratio = float(np.count_nonzero(high_sat)) / high_sat.size

    if ratio >= 0.001:  # 0.1%

        return {
            "status": "PASS",
            "details": (
                f"Reactive ink signature found "
                f"({ratio * 100:.2f}% of pixels)"
            )
        }

    return {
        "status": "INFO",
        "details": (
            f"UV-proxy signal too weak to assess "
            f"({ratio * 100:.2f}% of pixels) — "
            f"visible-light limitation, not a fake indicator"
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
    """Detect the vertical security thread on Indian banknotes.

    Production-grade approach: vertical Sobel + Otsu binarisation
    + morphological closing along a tall vertical kernel +
    opening to suppress noise + connectedComponentsWithStats to
    count narrow tall components. This is the standard CV
    technique for vertical-feature extraction used in document
    analysis (table-border detection, barcode-bar segmentation).

    Why this replaces the previous HoughLinesP approach:
      - The new Mahatma Gandhi Series uses a WINDOWED thread
        (broken into dashes). HoughLinesP needs continuous
        line segments of `minLineLength` height — it misses
        windowed threads entirely. The user reported FAIL on
        a clean 0AA specimen because of this.
      - Morphological vertical closing connects the dashes into
        a single tall connected component, which the connected-
        components pass then sees as one valid thread.

    Verified on the 33-fixture corpus: 22/22 real notes (clean +
    phone, including the previously-failing 0AA specimen) detect
    a thread component; structural fakes (blank, pure noise)
    correctly reject."""

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    if h < 60 or w < 60:
        return {
            "status": "INFO",
            "details": "Image too small to evaluate"
        }

    # Cover both old-series (~40-50% from left) and new-series
    # (~50-60% from left) thread positions with a wider strip.
    strip = img[0:h, int(w * 0.30):int(w * 0.70)]
    if strip.size == 0:
        return {
            "status": "INFO",
            "details": "Strip empty after crop"
        }

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Vertical Sobel: peaks at vertical edges.
    sobel = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel)

    # Otsu binarisation — adapts to lighting per image.
    _, binary = cv2.threshold(
        sobel_abs, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Vertical morphological CLOSE: connects dashes of a
    # windowed thread into continuous vertical features.
    # Kernel height ~8% of image height (min 15px).
    close_h = max(15, int(h * 0.08))
    vk_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, close_h),
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vk_close)

    # Vertical morphological OPEN: keep only column-aligned
    # features that span at least 20% of the image height.
    open_h = max(20, int(h * 0.20))
    vk_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, open_h),
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vk_open)

    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8,
    )

    # Filter components: tall (>= 25% of h), narrow (<= 15 px),
    # and reasonably dense (bbox fill >= 0.3 to reject sparse
    # noise patterns).
    threads = []
    for i in range(1, nlabels):
        _, _, cw, ch, area = stats[i]
        if ch < h * 0.25:
            continue
        if cw > 15:
            continue
        if area < cw * ch * 0.3:
            continue
        threads.append((ch, cw, area))

    if not threads:
        return {
            "status": "FAIL",
            "details": "No vertical thread structure detected",
        }

    # Report the tallest qualifying component as the canonical
    # signal.
    threads.sort(key=lambda t: t[0], reverse=True)
    tallest_h, tallest_w, _ = threads[0]
    tallest_pct = tallest_h / h * 100.0

    return {
        "status": "PASS",
        "details": (
            f"{len(threads)} vertical thread component(s) "
            f"found; tallest {tallest_pct:.0f}% of image height, "
            f"{tallest_w} px wide"
        ),
    }


# =====================================================
# 5b. ASCENDING SERIAL NUMBER TYPOGRAPHY ANALYSIS
# =====================================================
# RBI's Mahatma Gandhi New Series prints the bottom-right
# serial number with PROGRESSIVELY INCREASING digit sizes
# from left to right — the leftmost digit is smallest, the
# rightmost largest. This is a deliberate anti-counterfeit
# feature; cheap photocopy / inkjet fakes typically print
# every digit at a uniform height. Reference: RBI "Know
# Your Banknote" / press release on banknote security
# features.
#
# Pipeline:
#   1. Crop the bottom-right region of the rectified note
#      where the ascending serial lives (the small top-left
#      serial is uniform-size and NOT what we measure here).
#      When EasyOCR is available we refine the crop onto
#      the OCR'd serial bbox; otherwise we fall back to a
#      heuristic quadrant.
#   2. CLAHE + bilateral filter + adaptive threshold to
#      extract digit blobs from uneven phone-camera lighting.
#      A small morphological close repairs broken strokes.
#   3. External-contour pass with strict geometric filters
#      (area, aspect, fill, baseline alignment) to reject
#      noise, prefix letters, watermark texture and
#      background streaks. Overlapping detections are
#      merged so adaptive-threshold split-tops don't double-
#      count one digit.
#   4. From the left-to-right blob list, isolate the
#      trailing digit run — the 3-char prefix preceding the
#      digits has uniform typography and including it would
#      flatten the slope.
#   5. Linear regression of digit height vs position. Real
#      notes: positive slope, growth ratio > 1.08, decent
#      R². Fakes: flat or shrinking. We use a STATISTICALLY
#      TOLERANT validation (not strict-monotonic) because
#      real notes show ±5–10% jitter per digit from ink
#      wear, perspective and segmentation error.

_SERIAL_TYPO_DIGIT_AREA_FRAC = 0.0015  # min digit area / strip area
_SERIAL_TYPO_MIN_DIGITS = 3            # min aligned digits to call it
_SERIAL_TYPO_GROWTH_LO = 1.08          # last/first height lower bound
_SERIAL_TYPO_GROWTH_HI = 2.50          # implausible-segmentation cap
_SERIAL_TYPO_R2_PASS = 0.45            # R² floor for confident PASS
_SERIAL_TYPO_SLOPE_TOL = 0.005         # normalised slope/digit floor


def _serial_typo_locate_strip(image):
    """Crop the strip of the rectified note that contains the
    ascending bottom-right serial.

    Returns (strip_bgr, offset_xy) or (None, None). When
    EasyOCR is available we refine the crop onto the actual
    serial bbox; otherwise we fall back to a heuristic
    bottom-right quadrant which downstream contour analysis
    can still work with."""

    img = _ensure_bgr(image)
    h, w = img.shape[:2]

    if h < 40 or w < 80:
        return None, None

    # Bottom-right region. The ascending serial sits within
    # roughly the lower 45% × right 55% of the note.
    cx0, cy0 = int(w * 0.45), int(h * 0.55)
    coarse = img[cy0:int(h * 0.97), cx0:w]
    if coarse.size == 0:
        return None, None

    reader = _get_easyocr_reader()
    if reader is not None:

        try:
            results = reader.readtext(
                coarse, allowlist=_ALNUM_SPACE, detail=1
            )
        except Exception:
            results = []

        best = None
        best_area = 0

        for bbox, text, conf in results:

            if conf < 0.25:
                continue

            compact = text.strip().upper().replace(" ", "")
            if not compact:
                continue

            looks_like_serial = (
                _FULL_SERIAL_NO_SPACE.match(compact) is not None
                or _SERIAL_TEXT_REGEX.search(compact) is not None
                or (
                    len(compact) >= 6
                    and sum(c.isdigit() for c in compact) >= 5
                )
            )
            if not looks_like_serial:
                continue

            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            area = bw * bh
            if area > best_area:
                best = (xs, ys)
                best_area = area

        if best is not None:
            xs, ys = best
            pad_x = max(8, (max(xs) - min(xs)) // 20)
            pad_y = max(6, (max(ys) - min(ys)) // 8)
            x0 = max(min(xs) - pad_x, 0)
            x1 = min(max(xs) + pad_x, coarse.shape[1])
            y0 = max(min(ys) - pad_y, 0)
            y1 = min(max(ys) + pad_y, coarse.shape[0])

            crop = coarse[y0:y1, x0:x1]
            if crop.size > 0 and crop.shape[0] >= 12:
                return crop, (cx0 + x0, cy0 + y0)

    # Fallback: hand back the coarse quadrant. Contour
    # segmentation has been tuned to still pull out the
    # serial digits in this larger context.
    return coarse, (cx0, cy0)


def _serial_typo_segment(strip):
    """Adaptive-threshold + contour-extract candidate digit
    blobs from the serial strip. Returns left-to-right list
    of dicts with bbox and measurements."""

    if strip is None or strip.size == 0:
        return []

    gray = cv2.cvtColor(_ensure_bgr(strip), cv2.COLOR_BGR2GRAY)

    # Upscale tiny crops so adaptive thresholding has enough
    # pixels per stroke. Target ~80 px tall.
    h = gray.shape[0]
    scale = max(1.0, 80.0 / max(h, 1))
    if scale > 1.0:
        gray = cv2.resize(
            gray, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = _CLAHE.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Adaptive (gaussian) threshold inverted so digits become
    # white foreground. Block size + C calibrated for the
    # 50–120 px-tall serial strips we see after upscale.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 12,
    )

    # Tiny close to bridge broken strokes without merging
    # neighbouring digits.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    strip_h, strip_w = binary.shape
    strip_area = strip_h * strip_w
    min_digit_area = strip_area * _SERIAL_TYPO_DIGIT_AREA_FRAC

    raw = []
    for c in contours:

        x, y, cw, ch = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))

        if area < min_digit_area:
            continue
        if ch < strip_h * 0.20:        # too short — noise speck
            continue
        if ch > strip_h * 0.95:        # spans whole strip — border
            continue

        aspect = cw / max(ch, 1)
        if aspect < 0.15 or aspect > 1.6:  # not digit-shaped
            continue

        fill = area / max(cw * ch, 1)
        if fill < 0.18:                # sparse blob → background texture
            continue

        raw.append({
            "x": int(x), "y": int(y),
            "w": int(cw), "h": int(ch),
            "area": area,
            "baseline": int(y + ch),
            "fill": float(fill),
        })

    if not raw:
        return []

    # Drop blobs whose heights are nowhere near the median —
    # rejects watermark scraps and partial joins.
    heights = np.array([b["h"] for b in raw], dtype=np.float32)
    median_h = float(np.median(heights))
    raw = [
        b for b in raw
        if 0.35 * median_h <= b["h"] <= 2.5 * median_h
    ]
    if not raw:
        return []

    raw.sort(key=lambda b: b["x"])

    # De-duplicate overlapping detections (adaptive threshold
    # can split a single digit into stacked fragments).
    merged = [raw[0]]
    for b in raw[1:]:
        prev = merged[-1]
        overlap = b["x"] < prev["x"] + prev["w"] * 0.6
        if overlap:
            x0 = min(prev["x"], b["x"])
            y0 = min(prev["y"], b["y"])
            x1 = max(prev["x"] + prev["w"], b["x"] + b["w"])
            y1 = max(prev["y"] + prev["h"], b["y"] + b["h"])
            merged[-1] = {
                "x": x0, "y": y0,
                "w": x1 - x0, "h": y1 - y0,
                "area": prev["area"] + b["area"],
                "baseline": y1,
                "fill": (prev["fill"] + b["fill"]) / 2.0,
            }
        else:
            merged.append(b)

    return merged


def _serial_typo_pick_digits(blobs):
    """Isolate the trailing digit run from the left-to-right
    blob list. The 3-char alphabetic prefix does NOT ascend,
    so including it would flatten the slope and false-
    negative real notes. We also reject blobs whose baseline
    drifts from the median — ascenders, dots and stray noise."""

    if len(blobs) < _SERIAL_TYPO_MIN_DIGITS:
        return []

    # Drop the alphabetic prefix when we clearly have enough
    # trailing blobs. With 7-9 blobs total assume serial is
    # 6 digits + 3-char prefix; with ≤6 assume already digits-
    # only (refined EasyOCR crop).
    if len(blobs) >= 9:
        candidates = blobs[-7:]
    elif len(blobs) >= 7:
        candidates = blobs[-6:]
    else:
        candidates = blobs[:]

    baselines = np.array(
        [b["baseline"] for b in candidates], dtype=np.float32
    )
    heights = np.array(
        [b["h"] for b in candidates], dtype=np.float32
    )
    median_base = float(np.median(baselines))
    median_h = float(np.median(heights))
    tol = max(0.20 * median_h, 4.0)

    aligned = [
        b for b in candidates
        if abs(b["baseline"] - median_base) <= tol
    ]

    return aligned


def _serial_typo_fit(digits):
    """Forensic regression on per-digit measurements.

    Computes the full statistical envelope needed for RBI
    progression validation:

      Trend     — slope / intercept / R² of a linear fit
                  of height vs position
      Growth    — per-step ratios and percentages, mean,
                  median and overall first→last percentage
      Stability — monotonic_score (fraction of positive
                  steps), smoothness_score (1 − norm. std
                  of step ratios), consistency (1 − norm.
                  RMSE of the regression)
      Geometry  — per-digit height / width / area / baseline
                  lists and a baseline_alignment_score

    Returns None if there's not enough data to fit."""

    if len(digits) < _SERIAL_TYPO_MIN_DIGITS:
        return None

    xs = np.arange(len(digits), dtype=np.float64)
    ys = np.array([d["h"] for d in digits], dtype=np.float64)
    widths = np.array([d["w"] for d in digits], dtype=np.float64)
    areas = np.array([d["area"] for d in digits], dtype=np.float64)
    baselines = np.array(
        [d["baseline"] for d in digits], dtype=np.float64
    )

    mean_h = float(ys.mean())
    if mean_h < 1.0:
        return None

    # ---- Linear regression: height ~ position ----
    slope, intercept = np.polyfit(xs, ys, 1)
    fit = slope * xs + intercept
    ss_res = float(((ys - fit) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
    norm_slope = float(slope / mean_h)              # per-digit, normalised

    # ---- Per-step growth: ratio and % increase ----
    growth_ratios = [
        float(ys[i] / max(ys[i - 1], 1.0))
        for i in range(1, len(ys))
    ]
    growth_percentages = [
        round((r - 1.0) * 100.0, 2) for r in growth_ratios
    ]
    avg_growth_pct = (
        float(np.mean(growth_percentages))
        if growth_percentages else 0.0
    )
    median_growth_pct = (
        float(np.median(growth_percentages))
        if growth_percentages else 0.0
    )
    total_growth_pct = float((ys[-1] / max(ys[0], 1.0) - 1.0) * 100.0)
    overall_growth_ratio = float(ys[-1] / max(ys[0], 1.0))

    # ---- Stability scores ----
    # Monotonic score: fraction of inter-digit steps that
    # are non-negative. Tolerant of single-step jitter
    # (a fake reads ~0.50, a real note 0.85+).
    if growth_ratios:
        monotonic_score = float(
            sum(1 for r in growth_ratios if r >= 0.985)
            / len(growth_ratios)
        )
        gr_arr = np.array(growth_ratios, dtype=np.float64)
        gr_std = float(np.std(gr_arr))
        # Smoothness: 1 − coefficient of variation of growth
        # ratios. Real notes have low CV (smooth ramp), fakes
        # show wild noise around 1.0.
        smoothness_score = max(
            0.0, 1.0 - gr_std / max(float(np.mean(gr_arr)), 1e-3)
        )
    else:
        monotonic_score = 0.0
        smoothness_score = 0.0

    rmse = float(np.sqrt(ss_res / len(ys)))
    consistency = max(0.0, 1.0 - rmse / mean_h)

    # ---- Baseline alignment ----
    # std of baselines relative to mean digit height. A real
    # serial sits on a perfect horizontal baseline → score
    # near 1.0; a tilted or noisy crop drops below 0.7.
    baseline_std = float(np.std(baselines))
    baseline_alignment_score = max(
        0.0, 1.0 - baseline_std / mean_h
    )

    return {
        # --- regression ---
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2": round(r2, 4),
        "norm_slope": round(norm_slope, 4),
        # --- growth ---
        "growth_ratio": round(overall_growth_ratio, 4),
        "growth_ratios": [round(r, 4) for r in growth_ratios],
        "growth_percentages": growth_percentages,
        "average_growth_percentage": round(avg_growth_pct, 2),
        "median_growth_percentage": round(median_growth_pct, 2),
        "total_growth_percentage": round(total_growth_pct, 2),
        # --- stability ---
        "monotonic_score": round(monotonic_score, 4),
        "smoothness_score": round(smoothness_score, 4),
        "consistency": round(consistency, 4),
        # --- geometry ---
        "mean_height": round(mean_h, 2),
        "digit_heights": [int(v) for v in ys],
        "digit_widths": [int(v) for v in widths],
        "digit_areas": [round(float(v), 1) for v in areas],
        "digit_baselines": [int(v) for v in baselines],
        "baseline_alignment_score": round(baseline_alignment_score, 4),
        # --- legacy alias (kept so any caller relying on the
        # old key continues to work) ---
        "heights": [int(v) for v in ys],
        "growth_steps": [round(r, 3) for r in growth_ratios],
    }


def analyze_serial_typography(image):
    """RBI ascending-serial typography validation.

    Real banknote serials (bottom-right large serial) print
    progressively larger digits from left to right.
    Counterfeits typically print every digit the same size —
    flat slope, growth ratio near 1.0.

    Returns:
      PASS — positive growth trend with growth ratio in valid
             range and R² above floor → matches RBI spec
      FAIL — flat or shrinking digits + growth ratio ≈ 1.0 →
             uniform-size print, counterfeit signature
      INFO — serial region not isolatable, too few digit blobs
             segmented, baseline-alignment too weak, or signal
             too noisy to call confidently. Absence of
             measurement is NOT proof of fakery — we never
             FAIL on insufficient evidence."""

    try:
        strip, _ = _serial_typo_locate_strip(image)
    except Exception as exc:
        return {
            "status": "INFO",
            "details": f"Serial region not isolatable: {exc}",
            "value": None,
        }

    if strip is None or strip.size == 0:
        return {
            "status": "INFO",
            "details": "Serial region not isolatable in image",
            "value": None,
        }

    try:
        blobs = _serial_typo_segment(strip)
    except Exception as exc:
        return {
            "status": "INFO",
            "details": f"Digit segmentation failed: {exc}",
            "value": None,
        }

    if len(blobs) < _SERIAL_TYPO_MIN_DIGITS:
        return {
            "status": "INFO",
            "details": (
                f"Only {len(blobs)} digit blob(s) segmented in "
                f"serial strip — image may be blurry, low-"
                f"contrast or the serial may be partially cropped"
            ),
            "value": {"blobs": len(blobs)},
        }

    digits = _serial_typo_pick_digits(blobs)
    if len(digits) < _SERIAL_TYPO_MIN_DIGITS:
        return {
            "status": "INFO",
            "details": (
                f"{len(blobs)} blobs found but only {len(digits)} "
                f"share a baseline — typography measurement "
                f"unreliable on this image"
            ),
            "value": {"blobs": len(blobs), "aligned": len(digits)},
        }

    metrics = _serial_typo_fit(digits)
    if metrics is None:
        return {
            "status": "INFO",
            "details": "Could not fit progression model to digits",
            "value": None,
        }

    # Internal metrics drive the verdict; only the
    # human-friendly fields below are exposed to the UI.
    growth = metrics["growth_ratio"]
    norm_slope = metrics["norm_slope"]
    r2 = metrics["r2"]
    monotonic = metrics["monotonic_score"]
    total_pct = metrics["total_growth_percentage"]

    rbi_match = (
        growth >= _SERIAL_TYPO_GROWTH_LO
        and growth <= _SERIAL_TYPO_GROWTH_HI
        and norm_slope > _SERIAL_TYPO_SLOPE_TOL
        and r2 >= _SERIAL_TYPO_R2_PASS
        and monotonic >= 0.60
    )

    # Format growth percentages as plain integers — easier
    # to read than two-decimal floats on the UI chip.
    simple_growth_pcts = [
        int(round(p)) for p in metrics["growth_percentages"]
    ]
    total_growth_str = f"{int(round(total_pct)):+d}%"

    value = {
        "digit_sizes": metrics["digit_heights"],
        "growth_percentages": simple_growth_pcts,
        "total_growth": total_growth_str,
        "rbi_match": rbi_match,
    }

    # ---- FAIL: flat or shrinking — uniform-size print. ----
    if growth < 1.02 and norm_slope <= _SERIAL_TYPO_SLOPE_TOL:
        return {
            "status": "FAIL",
            "details": (
                "Serial numbers are not increasing properly. "
                "Genuine RBI notes usually show gradually "
                "increasing digit sizes."
            ),
            "value": value,
        }

    # ---- Sanity guard: implausibly large growth means we
    # caught non-digit blobs. Don't FAIL the note — say so. ----
    if growth > _SERIAL_TYPO_GROWTH_HI:
        return {
            "status": "INFO",
            "details": (
                "Could not measure the serial reliably "
                "(image too noisy or partially cropped)."
            ),
            "value": value,
        }

    # ---- PASS: smooth ascending progression. ----
    if rbi_match:
        return {
            "status": "PASS",
            "details": (
                "Serial numbers are gradually increasing in size "
                "from left to right, like genuine RBI notes."
            ),
            "value": value,
        }

    # ---- Borderline: don't FAIL on noisy phone photos. ----
    return {
        "status": "INFO",
        "details": (
            "Serial size pattern is unclear — image may be "
            "blurry or tilted. Pattern not clearly matching "
            "RBI notes but also not clearly fake."
        ),
        "value": value,
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

    # Thresholds calibrated against the 33-fixture corpus:
    #   - Reals (clean + phone + edge_case): min sat 21.8,
    #     min entropy 3.49 — all pass at sat>=20 / he>=3.4
    #   - Structural blanks: sat 0 — fail naturally
    #   - Desaturated fake: sat 0 — fails naturally
    # Phone reals have lower mean saturation than clean
    # Wikipedia scans because of camera processing and ambient
    # lighting; the old sat>=45 floor was calibrated against
    # the original 5-fixture sample and excluded most phone
    # uploads.
    if mean_sat >= 20 and hue_entropy >= 3.4:

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

    # Frame aspect is highly variable for phone uploads
    # (landscape shot, portrait shot, square crop, partial
    # crop). The aspect gate exists only to reject extreme
    # geometries (1×1000 strip, square logo). Real fakery
    # detection lives in the std/edge/quadrant checks below.
    if aspect < 0.25 or aspect > 4.0:
        return {
            "status": "FAIL",
            "details": f"Aspect ratio {aspect:.2f} extreme — not a photo of a note"
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

    Considers:
      1. The literal token after digit-confusion fix.
      2. Leading-digit remappings (₹ misread as 9/8/4/6).
      3. Stripping a junk leading character (₹50 read as "950"
         where the ₹ became a phantom "9"). Stripping yields
         the actual denomination directly."""

    if not token:
        return

    fixed = token.translate(_OCR_DIGIT_FIX)
    if not fixed.isdigit():
        return

    seen = set()

    if fixed in _KNOWN_DENOMINATIONS:
        seen.add(fixed)
        yield fixed

    if not fixed:
        return

    head = fixed[0]
    tail = fixed[1:]

    # Remap of just the leading digit (e.g. "900" -> "500")
    for repl in _DENOM_LEADING_FIX.get(head, ()):
        alt = repl + tail
        if alt in _KNOWN_DENOMINATIONS and alt not in seen:
            seen.add(alt)
            yield alt

    # Strip a phantom leading char (e.g. "950" -> "50",
    # "9500" -> "500", "92000" -> "2000")
    if tail in _KNOWN_DENOMINATIONS and tail not in seen:
        seen.add(tail)
        yield tail


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
    """Classify the note's face value via EasyOCR.

    Strategy:
      1. Get word-level OCR results with a digits-only
         allowlist (so the ₹ glyph and Hindi script don't
         pollute the candidate set).
      2. Run `_denom_candidates` on every token to catch
         exact matches and OCR-confusion remaps (e.g.
         "950" → "500" when the ₹ was misread as 9).
      3. Vote by summed confidence with exact reads
         outweighing confusion-remaps.
      4. Tiebreak with the colour-palette check so "200"
         on a stone-grey Rs 500 note resolves to 500.

    Falls through to `_classify_denomination_tesseract` only
    when EasyOCR is unavailable."""

    reader = _get_easyocr_reader()
    if reader is None:
        return _classify_denomination_tesseract(image)

    img = _ensure_bgr(image)

    try:
        results = reader.readtext(
            img, allowlist=_DIGITS_ALLOWLIST, detail=1
        )
    except Exception:
        return _classify_denomination_tesseract(image)

    exact_score = {}
    confused_score = {}

    for _, text, conf in results:

        text = text.strip().upper().replace(" ", "")
        if not text:
            continue
        # Permissive floor: EasyOCR can return correct digit
        # reads at 0.15-0.25 on tilted phone photos where the
        # "₹500" glyph confuses the recogniser. The denom
        # voting + palette tiebreak filters genuine noise.
        if conf < 0.15:
            continue

        candidates = list(_denom_candidates(text))
        if not candidates:
            continue

        conf_int = int(conf * 100)

        # First candidate is the literal read; remaining
        # candidates come from OCR-confusion remapping.
        head = candidates[0]
        if head == text:
            exact_score[head] = (
                exact_score.get(head, 0) + conf_int
            )
            rest = candidates[1:]
        else:
            rest = candidates

        for alt in rest:
            # Soft palette gate: full weight when palette
            # matches, severe downweight (not drop) when it
            # doesn't. Hard-dropping is wrong on small
            # phone-camera crops where most pixels are
            # background and the note's saturated colour
            # gets median-washed out.
            weight = 1.0 if _palette_match(img, alt) else 0.25
            confused_score[alt] = (
                confused_score.get(alt, 0)
                + int(conf_int * 0.7 * weight)
            )

    score_by_denom = dict(exact_score)
    for denom, score in confused_score.items():
        score_by_denom[denom] = (
            score_by_denom.get(denom, 0) + int(score * 0.5)
        )

    # Pure-colour fallback: OCR found nothing usable, but
    # the palette uniquely fits one denomination. Helps
    # blurry phone shots where the numeral is unreadable
    # but the note tint is clear. Disabled when the image
    # is mostly background (low saturated-pixel count)
    # because median-hue is meaningless there.
    if not score_by_denom:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_pixels = int(np.count_nonzero(hsv[:, :, 1] > 25))
        if sat_pixels >= img.shape[0] * img.shape[1] * 0.10:
            palette_hits = [
                d for d in _KNOWN_DENOMINATIONS
                if _palette_match(img, d)
            ]
            if len(palette_hits) == 1:
                score_by_denom[palette_hits[0]] = 40

    if not score_by_denom:
        return {
            "status": "FAIL",
            "details": "Could not read denomination",
            "value": None,
        }

    ranked = sorted(
        score_by_denom.items(),
        key=lambda kv: kv[1], reverse=True
    )
    value, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0

    # Demand a clear win, otherwise palette breaks the tie.
    if top_score < runner * 1.25 and top_score < 80:

        if (
            _palette_match(img, value)
            and not _palette_match(img, ranked[1][0])
        ):
            pass  # top stays
        elif (
            _palette_match(img, ranked[1][0])
            and not _palette_match(img, value)
        ):
            value, top_score = ranked[1]
        else:
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
            f"(EasyOCR score {int(top_score)})"
        ),
        "value": value,
    }


def _image_borders_uniform(
    img,
    max_border_std: float = 25.0,
    min_interior_std: float = 15.0,
) -> bool:
    """True when the outer border is roughly uniform AND the
    interior has content — i.e. the image is an isolated scan
    of a note (not a busy phone frame AND not a blank canvas).
    Only when both conditions hold does the frame aspect
    reliably approximate the note aspect."""

    h, w = img.shape[:2]
    bw = max(3, min(h, w) // 50)  # ~2% of the shorter edge
    if h <= 2 * bw or w <= 2 * bw:
        return False

    border = np.concatenate([
        img[:bw, :, :].reshape(-1, 3),
        img[-bw:, :, :].reshape(-1, 3),
        img[:, :bw, :].reshape(-1, 3),
        img[:, -bw:, :].reshape(-1, 3),
    ])
    interior = img[bw:-bw, bw:-bw, :]

    return (
        float(np.std(border)) < max_border_std
        and float(np.std(interior)) > min_interior_std
    )


def analyze_proportions(image, denomination=None):
    """Compare the detected note quad's aspect against the
    canonical RBI aspect for the OCR'd denomination.

    A counterfeit can be a real-note image that has been
    digitally stretched/squashed, or a printed fake on the
    wrong paper size. Either shows up as the note quad's
    aspect ratio deviating from the canonical value.

    Returns:
      PASS  — detected aspect within `_PROPORTION_TOLERANCE_PCT`
              of canonical
      FAIL  — deviation exceeds the tolerance (likely digital
              stretching or wrong-size paper)
      INFO  — no quad detectable, or denomination unknown.
              We do NOT FAIL in this case — absence of signal
              isn't proof of fakery.

    The `value` payload carries the raw numbers so the frontend
    and the diagnostic harness can render them: {actual_aspect,
    expected_aspect, deviation_pct}. None when we can't compute."""

    if not denomination or denomination not in _RBI_EXPECTED_ASPECT:
        return {
            "status": "INFO",
            "details": (
                "Denomination unknown — cannot compare against "
                "canonical proportions"
            ),
            "value": None,
        }

    img = _ensure_bgr(image)

    # Primary: detect the note quad in the image. Works for
    # phone photos with visible background around the note.
    detection = _detect_note_quad(img)
    if detection is not None:
        actual_aspect = float(detection["aspect"])
        method = "quad"
    elif _image_borders_uniform(img):
        # Fallback: image borders are uniform (clean scan or
        # tight-cropped upload), so the frame aspect equals
        # the note aspect. Don't use this fallback on busy
        # phone-camera frames — there the frame aspect is the
        # photo's aspect, not the note's, and we'd false-
        # positive every real upload as "stretched".
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return {
                "status": "INFO",
                "details": "Image too small to measure proportions",
                "value": None,
            }
        long_edge = max(w, h)
        short_edge = max(min(w, h), 1)
        actual_aspect = long_edge / short_edge
        method = "frame"
    else:
        return {
            "status": "INFO",
            "details": (
                "Note edges not detectable and image is not "
                "an isolated scan — cannot measure proportions"
            ),
            "value": None,
        }
    expected_aspect = _RBI_EXPECTED_ASPECT[denomination]
    deviation_pct = (
        abs(actual_aspect - expected_aspect) / expected_aspect
        * 100.0
    )

    value = {
        "actual_aspect": round(actual_aspect, 3),
        "expected_aspect": round(expected_aspect, 3),
        "deviation_pct": round(deviation_pct, 1),
        "measurement": method,
    }

    if deviation_pct <= _PROPORTION_TOLERANCE_PCT:
        return {
            "status": "PASS",
            "details": (
                f"Proportions match Rs {denomination} canonical "
                f"({actual_aspect:.2f} vs {expected_aspect:.2f}, "
                f"{deviation_pct:.1f}% deviation, via {method})"
            ),
            "value": value,
        }

    return {
        "status": "FAIL",
        "details": (
            f"Proportions off for Rs {denomination} "
            f"({actual_aspect:.2f} vs canonical "
            f"{expected_aspect:.2f}, {deviation_pct:.1f}% "
            f"deviation via {method} — likely digital "
            f"stretching or wrong-size paper)"
        ),
        "value": value,
    }


def _classify_denomination_tesseract(image):
    """Legacy Tesseract path. Kept as silent fallback."""

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
        # dominate the crop. We use slightly lenient bounds
        # so partially-occluded numerals on phone photos
        # still pass.
        if label == "whole":
            min_frac = 0.03
        else:
            min_frac = 0.12

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
                    if conf < 25:
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

    score_by_denom = dict(exact_score)

    # Layer in confused-score (digit-confusion remaps) at half
    # weight so they can vote when nothing matches exactly,
    # but never override a strong exact read.
    for denom, score in confused_score.items():
        score_by_denom[denom] = score_by_denom.get(denom, 0) + score * 0.5

    # Pure-colour fallback: if OCR couldn't find anything but
    # the colour palette uniquely matches one denomination,
    # surface that. Useful for blurry phone shots where the
    # numeral can't be read at all but the note tint is clear.
    if not score_by_denom:
        palette_hits = [
            d for d in _KNOWN_DENOMINATIONS
            if _palette_match(img, d)
        ]
        if len(palette_hits) == 1:
            score_by_denom[palette_hits[0]] = 40

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

    # Demand a clear win — otherwise palette check breaks ties.
    if top_score < runner * 1.25 and top_score < 80:

        # Use colour palette to break the tie.
        if _palette_match(img, value) and not _palette_match(img, ranked[1][0]):
            pass  # top stays
        elif _palette_match(img, ranked[1][0]) and not _palette_match(img, value):
            value, top_score = ranked[1]
        else:
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

    Pre-processing applied here (not in the individual
    checks so they remain unit-testable in isolation):
      1. _locate_note — find the banknote quad in the
         frame and perspective-rectify it to a canonical
         landscape crop. Strips out desk / hand / sky
         background so downstream checks see the note,
         not the room. Falls through unchanged when no
         quad is found.

    Post-processing — analyze_proportions runs AFTER the
    main checks because it consumes the denomination result.
    It measures the quad on the ORIGINAL (pre-crop) image
    because the auto-crop step has already normalised the
    aspect away from the input geometry we want to evaluate.
    """

    original = _ensure_bgr(image)
    image = _locate_note(original)

    checks = {
        "structural_sanity": structural_sanity,
        "uv_light_detection": analyze_uv_features,
        "watermark_detection": detect_watermark,
        "ocr_serial_number": extract_serial_number,
        "gandhi_face_analysis": analyze_gandhi_face,
        "security_thread_detection": detect_security_thread,
        "serial_typography_analysis": analyze_serial_typography,
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

    # Proportion check depends on the denomination output and
    # must see the original (pre-crop) image — wired here as
    # a post-pass rather than inside the `checks` dict.
    try:
        denom_value = (
            results.get("denomination_classification", {})
            .get("value")
        )
        results["proportion_analysis"] = analyze_proportions(
            original, denom_value
        )
    except Exception as exc:
        results["proportion_analysis"] = {
            "status": "INFO",
            "details": f"Error: {exc}",
            "value": None,
        }

    results["modular_ai_pipeline"] = {
        "status": "PASS",
        "details": "Pipeline executed successfully"
    }

    return results
