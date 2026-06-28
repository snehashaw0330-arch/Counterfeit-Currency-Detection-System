"""
Fetch a small genuine polymer-note reference set into the locked foreign layout.

Layout:
    dataset/foreign/<ccy>/full_note/real/<name>.<ext>
    dataset/foreign/<ccy>/full_note/real/<name>.json

These are genuine reference images only. Public counterfeit polymer-note image
sets are scarce; this script deliberately does not chase them.

Sources are Wikipedia page-lead images for well-known polymer banknote pages.
That keeps the fetch reproducible without manual browser work.

Run:
    venv\\Scripts\\python.exe scripts\\fetch_polymer_reference_images.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset", "foreign")

REFS = [
    {
        "title": "Australian five-dollar note",
        "commons_file": "2016 Australian five dollar note obverse.jpg",
        "country": "Australia",
        "currency": "AUD",
        "denomination": "5",
        "side": "obverse",
        "substrate": "polymer",
        "source": "Wikipedia lead image",
    },
    {
        "title": "Canadian twenty-dollar note",
        "direct_image_url": "https://www.bankofcanada.ca/wp-content/uploads/2015/09/20_front.png",
        "country": "Canada",
        "currency": "CAD",
        "denomination": "20",
        "side": "obverse",
        "substrate": "polymer",
        "source": "Bank of Canada official page image",
    },
    {
        "title": "Bank of England £20 note",
        "direct_image_url": "https://www.bankofengland.co.uk/-/media/boe/images/banknotes/20/polymer/polymer-20-specimen-front-(1).jpg",
        "country": "United Kingdom",
        "currency": "GBP",
        "denomination": "20",
        "side": "obverse",
        "substrate": "polymer",
        "source": "Bank of England official page image",
    },
    {
        "title": "Philippine one thousand-peso note",
        "country": "Philippines",
        "currency": "PHP",
        "denomination": "1000",
        "side": "obverse",
        "substrate": "polymer",
        "source": "Wikipedia lead image",
    },
]


def _json_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_title(title: str) -> str:
    encoded = urllib.parse.quote(title, safe="")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&list=search&srlimit=1&srsearch=" + encoded
    )
    payload = _json_get(url)
    hits = payload.get("query", {}).get("search", [])
    if hits:
        return hits[0].get("title", title)
    return title


def _image_info_url(file_title: str) -> str | None:
    encoded = urllib.parse.quote(file_title, safe="")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&prop=imageinfo&iiprop=url&titles=" + encoded
    )
    payload = _json_get(url)
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        items = page.get("imageinfo") or []
        if items and items[0].get("url"):
            return items[0]["url"]
    return None


def _page_image_fallback(title: str) -> str | None:
    resolved = _resolve_title(title)
    encoded = urllib.parse.quote(resolved, safe="")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&prop=images&imlimit=max&titles=" + encoded
    )
    payload = _json_get(url)
    pages = payload.get("query", {}).get("pages", {})
    candidates = []
    for page in pages.values():
        for item in page.get("images", []):
            name = item.get("title", "")
            low = name.lower()
            if any(tok in low for tok in ("reverse", "back")):
                continue
            score = 0
            if "obverse" in low:
                score += 5
            if "front" in low:
                score += 4
            if "polymer" in low:
                score += 3
            if "note" in low or "banknote" in low:
                score += 2
            if score:
                candidates.append((score, name))
    candidates.sort(reverse=True)
    for _score, file_title in candidates:
        img_url = _image_info_url(file_title)
        if img_url:
            return img_url
    return None


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _lead_image(title: str) -> str:
    resolved = _resolve_title(title)
    encoded = urllib.parse.quote(resolved, safe="")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&prop=pageimages&piprop=original|thumbnail&pithumbsize=1800&titles=" + encoded
    )
    payload = _json_get(url)
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        src = page.get("original", {}).get("source")
        if not src:
            src = page.get("thumbnail", {}).get("source")
        if src:
            return src
    src = _page_image_fallback(title)
    if src:
        return src
    raise RuntimeError(f"No lead image found for {title} (resolved as {resolved})")


def _file_urls(filename: str) -> list[str]:
    encoded = urllib.parse.quote(filename, safe="")
    return [
        "https://en.wikipedia.org/wiki/Special:FilePath/" + encoded,
        "https://commons.wikimedia.org/wiki/Special:FilePath/" + encoded,
    ]


def _download_any(urls: list[str], dest: str) -> str:
    last_exc = None
    for url in urls:
        try:
            _download(url, dest)
            return url
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError("all download URLs failed")


def _safe_stem(meta: dict) -> str:
    return f"{meta['currency'].lower()}_{meta['denomination']}_{meta['side']}"


def main() -> int:
    wrote = 0
    for meta in REFS:
        img_url = meta.get("direct_image_url")
        if meta.get("direct_image_url"):
            ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1] or ".jpg"
        elif meta.get("commons_file"):
            ext = os.path.splitext(meta["commons_file"])[1] or ".jpg"
        else:
            img_url = _lead_image(meta["title"])
            ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1] or ".jpg"
        out_dir = os.path.join(BASE, meta["currency"].lower(), "full_note", "real")
        os.makedirs(out_dir, exist_ok=True)
        stem = _safe_stem(meta)
        img_path = os.path.join(out_dir, stem + ext)
        json_path = os.path.join(out_dir, stem + ".json")

        if not os.path.exists(img_path):
            if meta.get("direct_image_url"):
                _download(img_url, img_path)
            elif meta.get("commons_file"):
                img_url = _download_any(_file_urls(meta["commons_file"]), img_path)
            else:
                _download(img_url, img_path)
            wrote += 1

        payload = {
            "country": meta["country"],
            "currency": meta["currency"],
            "substrate": meta["substrate"],
            "denomination": meta["denomination"],
            "side": meta["side"],
            "label": "real",
            "source": meta["source"],
            "source_page": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(meta['title'].replace(' ', '_'))}",
            "image_url": img_url,
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    print(f"Wrote {wrote} new polymer reference image(s) under dataset/foreign/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
