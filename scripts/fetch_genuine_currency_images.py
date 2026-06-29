"""
Fetch a broader set of GENUINE banknote images (multiple denominations across
several currencies) into the locked foreign layout, for the multi-currency
IDENTIFICATION benchmark and genuine-only (one-class) screening experiments.

Genuine only. Public counterfeit polymer images don't exist, so this never
chases fakes. Source: Wikipedia per-denomination page lead images (reproducible,
no manual browser work). The dataset/foreign tree is git-ignored.

Layout:
    dataset/foreign/<ccy>/full_note/real/<ccy>_<denom>_obverse.<ext> (+ .json)

Run:
    venv\\Scripts\\python.exe scripts\\fetch_genuine_currency_images.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset", "foreign")

# Wikipedia asks for a descriptive User-Agent and polite request rates.
_UA = "CounterfeitCurrencyDetection/1.0 (academic research; genuine reference images)"
_THROTTLE = 1.2  # seconds between API calls (avoid HTTP 429)

# (country, currency, substrate, [(denomination, wikipedia page title), ...])
CURRENCIES = [
    ("Australia", "AUD", "polymer", [
        ("5", "Australian five-dollar note"),
        ("10", "Australian ten-dollar note"),
        ("20", "Australian twenty-dollar note"),
        ("50", "Australian fifty-dollar note"),
        ("100", "Australian one-hundred-dollar note"),
    ]),
    ("Canada", "CAD", "polymer", [
        ("5", "Canadian five-dollar note"),
        ("10", "Canadian ten-dollar note"),
        ("20", "Canadian twenty-dollar note"),
        ("50", "Canadian fifty-dollar note"),
        ("100", "Canadian one-hundred-dollar note"),
    ]),
    ("United Kingdom", "GBP", "polymer", [
        ("5", "Bank of England £5 note"),
        ("10", "Bank of England £10 note"),
        ("20", "Bank of England £20 note"),
        ("50", "Bank of England £50 note"),
    ]),
    ("Philippines", "PHP", "polymer", [
        ("50", "Philippine fifty-peso note"),
        ("100", "Philippine one hundred-peso note"),
        ("200", "Philippine two hundred-peso note"),
        ("500", "Philippine five hundred-peso note"),
        ("1000", "Philippine one thousand-peso note"),
    ]),
    ("United States", "USD", "paper", [
        ("1", "United States one-dollar bill"),
        ("5", "United States five-dollar bill"),
        ("10", "United States ten-dollar bill"),
        ("20", "United States twenty-dollar bill"),
        ("50", "United States fifty-dollar bill"),
        ("100", "United States one-hundred-dollar bill"),
    ]),
    ("Eurozone", "EUR", "paper", [
        ("5", "5 euro note"),
        ("10", "10 euro note"),
        ("20", "20 euro note"),
        ("50", "50 euro note"),
        ("100", "100 euro note"),
    ]),
]


def _get(url: str) -> dict:
    """Throttled GET with one 429 backoff retry."""
    for attempt in (0, 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(_THROTTLE)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(5.0)
                continue
            raise
    return {}


def _resolve(title: str) -> str:
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&list=search&srlimit=1&srsearch=" + urllib.parse.quote(title, safe=""))
    hits = _get(url).get("query", {}).get("search", [])
    return hits[0]["title"] if hits else title


def _image_info_url(file_title: str) -> str | None:
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&prop=imageinfo&iiprop=url&titles=" + urllib.parse.quote(file_title, safe=""))
    for page in _get(url).get("query", {}).get("pages", {}).values():
        items = page.get("imageinfo") or []
        if items and items[0].get("url"):
            return items[0]["url"]
    return None


def _page_image_fallback(resolved: str) -> str | None:
    """When pageimages has no lead, scan the page's images for an obverse/front."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&prop=images&imlimit=max&titles=" + urllib.parse.quote(resolved, safe=""))
    candidates = []
    for page in _get(url).get("query", {}).get("pages", {}).values():
        for item in page.get("images", []):
            name = item.get("title", "")
            low = name.lower()
            if not low.endswith((".jpg", ".jpeg", ".png")):
                continue
            if any(t in low for t in ("reverse", "back")):
                continue
            score = (("obverse" in low) * 5 + ("front" in low) * 4
                     + ("note" in low or "banknote" in low) * 2 + ("polymer" in low) * 1)
            if score:
                candidates.append((score, name))
    for _s, file_title in sorted(candidates, reverse=True):
        u = _image_info_url(file_title)
        if u:
            return u
    return None


def _lead_image(title: str) -> str | None:
    resolved = _resolve(title)
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           "&prop=pageimages&piprop=original|thumbnail&pithumbsize=1800&titles="
           + urllib.parse.quote(resolved, safe=""))
    for page in _get(url).get("query", {}).get("pages", {}).values():
        src = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")
        if src:
            return src
    return _page_image_fallback(resolved)


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def main() -> int:
    wrote, failed = 0, []
    for country, ccy, substrate, denoms in CURRENCIES:
        out_dir = os.path.join(BASE, ccy.lower(), "full_note", "real")
        os.makedirs(out_dir, exist_ok=True)
        for denom, title in denoms:
            stem = f"{ccy.lower()}_{denom}_obverse"
            # skip if any image with this stem already exists
            if any(os.path.exists(os.path.join(out_dir, stem + e))
                   for e in (".jpg", ".jpeg", ".png", ".svg")):
                continue
            try:
                img_url = _lead_image(title)
                if not img_url:
                    failed.append(f"{ccy} {denom} (no lead image)")
                    continue
                ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1].lower() or ".jpg"
                if ext == ".svg":  # vector lead image — skip, not a photo
                    failed.append(f"{ccy} {denom} (svg only)")
                    continue
                img_path = os.path.join(out_dir, stem + ext)
                _download(img_url, img_path)
                with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as fh:
                    json.dump({
                        "country": country, "currency": ccy, "substrate": substrate,
                        "denomination": denom, "side": "obverse", "label": "real",
                        "source": "Wikipedia lead image",
                        "source_page": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                        "image_url": img_url,
                    }, fh, indent=2)
                wrote += 1
                print(f"  {ccy} {denom}: {os.path.basename(img_path)}")
            except Exception as exc:
                failed.append(f"{ccy} {denom} ({exc})")

    print(f"\nWrote {wrote} genuine image(s) under dataset/foreign/")
    if failed:
        print(f"Skipped/failed {len(failed)}: " + "; ".join(failed[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
