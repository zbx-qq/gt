#!/usr/bin/env python3
"""Collect CC0 nature photos and public-domain/CC0 photographic portraits.
Source records and rejection counts are part of the deliverables.
No scraping HTML, no bypassing rate limits, no synthetic image substitutions.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import io
import json
import random
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from pathlib import Path
from urllib.parse import quote
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError

ROOT = Path("work")
OUT = Path("output")
ROOT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
_tls = local()
USER_AGENT = "OpenLicensedPhotoPack/1.1 (https://github.com/zbx-qq/gt; cite source and license)"
START = time.monotonic()
RUN_BUDGET = 26 * 60
BACKGROUND_WORKERS = 12
PORTRAIT_WORKERS = 8
print(f"[SETUP] background_workers={BACKGROUND_WORKERS} portrait_workers={PORTRAIT_WORKERS} retries=3 connect_timeout=5s read_timeout=12s", flush=True)
random.seed(20261015)
MW = "https://commons.wikimedia.org/w/api.php"
BG_REPO = "Salford-PsyTech/Salford_Nature_Environments_Database"
if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
    raise RuntimeError("OpenCV face detector unavailable: refusing to produce unvalidated avatar pack")
def face_detector():
    detector = getattr(_tls, "haar", None)
    if detector is None:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("OpenCV frontal face classifier missing")
        _tls.haar = detector
    return detector
print("[SETUP] cv2 version", cv2.__version__, "face_detector=True", flush=True)
stats = collections.Counter()

def get(url, params=None, binary=False):
    # Each worker gets its own session; GET calls are not serialized.
    session = getattr(_tls, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _tls.session = session
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=(5, 12))
            # Do not retry forbidden, missing or removed resources.
            if r.status_code in (401, 403, 404, 410):
                r.raise_for_status()
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"temporary HTTP {r.status_code}")
            r.raise_for_status()
            return r.content if binary else r.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                raise
            # Retry only briefly; never wait tens of seconds for one picture.
            time.sleep(min(2.5, .5 * 2 ** attempt + random.uniform(0, .3)))
    raise RuntimeError("Download attempts exhausted")

def make_jpg(blob, width, height, blackwhite=False, face=False):
    im = Image.open(io.BytesIO(blob))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width < 450 or im.height < 450:
        raise ValueError("too small")
    if face:
        arr = np.array(im)
        scale = min(1.0, 960/max(im.size))
        sm = cv2.resize(arr, None, fx=scale, fy=scale) if scale < 1.0 else arr
        gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
        faces = face_detector().detectMultiScale(gray, 1.08, 4, minSize=(55,55))
        if len(faces) != 1:
            raise ValueError("not exactly one clear forward-facing face")
        x,y,w,h = [int(t/scale) for t in faces[0]]
        if w < 0.11 * im.width:
            raise ValueError("face too small")
        side = int(max(w * 2.2, h * 2.0))
        cx = x + w//2
        cy = y + h//2 + int(h*.12)
        left = max(0, min(im.width-side, cx-side//2))
        top = max(0, min(im.height-side, cy-side//2))
        right = min(im.width, left + side)
        bottom = min(im.height, top + side)
        im = im.crop((left, top, right, bottom))
    im = ImageOps.fit(im, (width, height), method=Image.Resampling.LANCZOS, centering=(.5,.5))
    if blackwhite:
        im = ImageOps.colorize(ImageOps.grayscale(im), black="#0d0d0d", white="#f3f3f3")
        im = ImageEnhance.Contrast(im).enhance(1.12)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True, subsampling=0)
    arr = np.asarray(im.resize((9,8)).convert("L"), dtype=np.int16)
    phash = sum(int(v) << i for i,v in enumerate((arr[:,1:] > arr[:,:-1]).ravel()))
    return buf.getvalue(), phash

def too_similar(value, prev):
    return any((value ^ other).bit_count() <= 6 for other in prev)

def write_zip(folder, zip_name, rows):
    out = OUT / zip_name
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(folder.glob("*.jpg")):
            z.write(p, arcname=p.name)
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=["filename","source","license","creator"])
        writer.writeheader()
        writer.writerows(rows)
        z.writestr("SOURCES_LICENSES.csv", f.getvalue().encode("utf-8-sig"))
        z.writestr("READ_ME_FIRST.txt",
            "Images are actual photographic source images, not generated substitutes.\n"
            "Photo licenses are supplied in SOURCES_LICENSES.csv.\n"
            "Public-domain copyright status does not remove privacy, publicity, or likeness rights.\n"
            "Do not pretend to be a pictured person or falsely imply their endorsement.\n"
            "For social profiles, use only when you have the subject's permission to represent them.\n")
    print("ZIP", out, out.stat().st_size, "photo_count", len(rows), flush=True)

def bg_worker(name):
    uri = f"https://raw.githubusercontent.com/{BG_REPO}/main/Images/{quote(name)}"
    content = get(uri, binary=True)
    data, h = make_jpg(content, 1280, 720)
    return name, uri, data, h

def backgrounds():
    folder = ROOT / "backgrounds"
    folder.mkdir(exist_ok=True)
    listing = get(f"https://api.github.com/repos/{BG_REPO}/contents/Images")
    all_files = [d["name"] for d in listing if d.get("type") == "file"
                 and d["name"].lower().endswith(".jpg") and not d["name"].startswith("control")]
    random.shuffle(all_files)
    rows, hashes = [], set()
    print(f"[BACKGROUNDS] {len(all_files)} candidate landscape files found", flush=True)
    with ThreadPoolExecutor(max_workers=BACKGROUND_WORKERS) as pool:
        for offset in range(0, len(all_files), BACKGROUND_WORKERS * 2):
            if len(rows) >= 300 or time.monotonic() - START > RUN_BUDGET:
                break
            batch = all_files[offset:offset + BACKGROUND_WORKERS * 2]
            futures = [pool.submit(bg_worker, name) for name in batch]
            for future in as_completed(futures):
                try:
                    name, uri, data, h = future.result()
                    if len(rows) >= 300:
                        continue
                    if too_similar(h, hashes):
                        stats["background_duplicates"] += 1
                        continue
                    fname = f"background_{len(rows)+1:04d}.jpg"
                    (folder/fname).write_bytes(data)
                    rows.append(dict(filename=fname, source=uri,
                        license="CC0 (per SNED source records)",
                        creator="Listed in SNED ImagesBook_CollatedImagesData.csv"))
                    hashes.add(h)
                    if len(rows) % 25 == 0:
                        print(f"[BACKGROUNDS] {len(rows)}/300; tried={offset+len(batch)}; elapsed={time.monotonic()-START:.0f}s", flush=True)
                except Exception as exc:
                    stats["background_errors"] += 1
                    if stats["background_errors"] <= 8:
                        print("[BACKGROUNDS] skipped:", str(exc)[:140], flush=True)
    write_zip(folder, "300_landscape_photographs_CC0.zip", rows)
    print(f"[BACKGROUNDS] FINAL {len(rows)}/300", flush=True)
    return len(rows)

def permitted_license(meta):
    name = meta.get("LicenseShortName",{}).get("value","").lower().strip()
    license_url = meta.get("LicenseUrl",{}).get("value","").lower()
    rights = meta.get("UsageTerms",{}).get("value","").lower()
    if name == "cc0" or "publicdomain/zero" in license_url:
        return "CC0"
    if name in ("public domain","pd") or ("public domain" in rights and "cc by" not in rights):
        return "Public domain"
    if "publicdomain/mark" in license_url:
        return "Public Domain Mark"
    return None

def strip_html(s):
    return re.sub(r"<[^>]*>", "", s).strip()[:240]

def wikimedia_pages(category):
    cont = {}
    visited = 0
    while visited < 30000:
        params = {"action":"query","format":"json","generator":"categorymembers","gcmtitle":"Category:"+category,
            "gcmtype":"file","gcmlimit":50, "prop":"imageinfo","iiprop":"url|size|extmetadata|mediatype",
            "iiurlwidth":900, "maxlag":5}
        params.update(cont)
        try:
            data = get(MW, params=params)
        except Exception as exc:
            print("category stopped", category, exc,flush=True)
            break
        if "error" in data:
            print("Wiki API error",data["error"],flush=True)
            break
        pages = list(data.get("query",{}).get("pages",{}).values())
        random.shuffle(pages)
        for item in pages:
            visited += 1
            yield item
        cont = data.get("continue",{})
        if not cont:
            break
        time.sleep(.25)

def portrait_candidate(item):
    title = item.get("title", "")
    ii = item.get("imageinfo", [{}])[0]
    if not ii or not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return None
    if min(ii.get("width", 0), ii.get("height", 0)) < 600:
        return None
    if max(ii.get("width", 0), ii.get("height", 0)) > 16000:
        return None
    license_name = permitted_license(ii.get("extmetadata", {}))
    if not license_name:
        stats["unlicensed_for_pack"] += 1
        return None
    uri = ii.get("thumburl") or ii.get("url")
    if not uri:
        return None
    meta = ii.get("extmetadata", {})
    return dict(uri=uri, source=ii.get("descriptionurl",
                "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")),
                license=license_name, creator=strip_html(meta.get("Artist", {}).get("value", "")))

def portrait_worker(candidate):
    # Black-and-white profile photo; real photo and actual detected forward face.
    blob = get(candidate["uri"], binary=True)
    photo, h = make_jpg(blob, 512, 512, blackwhite=True, face=True)
    return candidate, photo, h

def portraits():
    folder = ROOT / "portraits"
    folder.mkdir(exist_ok=True)
    rows, hashes, titles = [], set(), set()
    categories = [
        "Portrait photographs of women", "Portrait photographs of men",
        "Front view portrait photographs of females", "Front view portrait photographs of male people",
        "Black and white portrait photographs of women", "Black and white portrait photographs of men",
    ]
    sources = iter(round_robin(wikimedia_pages(c) for c in categories))
    source_exhausted = False
    inspected = 0
    print("[PORTRAITS] Start CC0/public-domain metadata screening and parallel image evaluation", flush=True)
    with ThreadPoolExecutor(max_workers=PORTRAIT_WORKERS) as pool:
        while len(rows) < 500 and not source_exhausted and time.monotonic()-START < RUN_BUDGET:
            candidates = []
            while len(candidates) < PORTRAIT_WORKERS * 3:
                try:
                    item = next(sources)
                except StopIteration:
                    source_exhausted = True
                    break
                title = item.get("title", "")
                if title in titles:
                    continue
                titles.add(title)
                inspected += 1
                c = portrait_candidate(item)
                if c:
                    candidates.append(c)
                if inspected % 250 == 0:
                    print(f"[PORTRAITS] screened={inspected}; accepted={len(rows)}/500; candidates_this_batch={len(candidates)}", flush=True)
            if not candidates:
                continue
            futures = [pool.submit(portrait_worker, c) for c in candidates]
            for f in as_completed(futures):
                try:
                    candidate, photo, h = f.result()
                    if len(rows) >= 500:
                        continue
                    if too_similar(h, hashes):
                        stats["portrait_duplicates"] += 1
                        continue
                    fname = f"portrait_{len(rows)+1:04d}.jpg"
                    (folder/fname).write_bytes(photo)
                    rows.append(dict(filename=fname,source=candidate["source"],
                        license=candidate["license"],creator=candidate["creator"]))
                    hashes.add(h)
                    if len(rows) % 25 == 0:
                        print(f"[PORTRAITS] {len(rows)}/500; examined={inspected}; elapsed={time.monotonic()-START:.0f}s",flush=True)
                except (UnidentifiedImageError, OSError, ValueError):
                    stats["face_or_format_rejected"] += 1
                except Exception as exc:
                    stats["portrait_errors"] += 1
                    if stats["portrait_errors"] <= 8:
                        print("[PORTRAITS] download/detection error:", str(exc)[:140],flush=True)
    write_zip(folder, "500_real_portrait_photographs_CC0_PD.zip", rows)
    print(f"[PORTRAITS] FINAL {len(rows)}/500; screened={inspected}; rejected_faces={stats['face_or_format_rejected']}", flush=True)
    return len(rows)

def round_robin(iters):
    iters = [iter(it) for it in iters]
    while iters:
        active = []
        for it in iters:
            try:
                yield next(it)
                active.append(it)
            except StopIteration:
                pass
        iters = active

def main():
    b=backgrounds()
    p=portraits()
    result={"background_count":b,"portrait_count":p,"request":{"background":300,"portrait":500},"stats":dict(stats)}
    (OUT/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FINAL MANIFEST",json.dumps(result),flush=True)

if __name__ == "__main__":
    main()
