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
S = requests.Session()
S.headers.update({"User-Agent": "OpenLicensedPhotoPack/1.0 (https://github.com/zbx-qq/gt; educational metadata and copyright attribution)"})
random.seed(20261015)
MW = "https://commons.wikimedia.org/w/api.php"
BG_REPO = "Salford-PsyTech/Salford_Nature_Environments_Database"
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
stats = collections.Counter()

def get(url, params=None, binary=False):
    for n in range(7):
        try:
            r = S.get(url, params=params, timeout=45)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError("HTTP " + str(r.status_code))
            r.raise_for_status()
            return r.content if binary else r.json()
        except (requests.RequestException, ValueError, RuntimeError) as e:
            if n == 6:
                print("GIVE UP:", str(e), str(url)[:90], flush=True)
                raise
            time.sleep(min(35, 2 ** n + random.random()))
    raise RuntimeError("unreachable")

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
        faces = haar.detectMultiScale(gray, 1.08, 4, minSize=(55,55))
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

def backgrounds():
    folder = ROOT / "backgrounds"
    folder.mkdir(exist_ok=True)
    listing = get(f"https://api.github.com/repos/{BG_REPO}/contents/Images")
    all_files = [d["name"] for d in listing if d.get("type")=="file" and d["name"].lower().endswith(".jpg") and not d["name"].startswith("control")]
    random.shuffle(all_files)
    rows, hashes = [], set()
    for name in all_files:
        if len(rows) >= 300: break
        try:
            uri = f"https://raw.githubusercontent.com/{BG_REPO}/main/Images/{quote(name)}"
            content = get(uri, binary=True)
            data, h = make_jpg(content, 1280, 720)
            if too_similar(h, hashes):
                stats["background_duplicates"] += 1
                continue
            fname = f"background_{len(rows)+1:04d}.jpg"
            (folder/fname).write_bytes(data)
            rows.append(dict(filename=fname, source=uri, license="CC0 (SNED source collection)",creator="Listed in SNED ImagesBook_CollatedImagesData.csv"))
            hashes.add(h)
            if len(rows) % 25 == 0: print("backgrounds",len(rows), flush=True)
        except Exception as exc:
            stats["background_errors"] += 1
            print("background error", name, str(exc)[:120], flush=True)
    write_zip(folder, "300_landscape_photographs_CC0.zip", rows)
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
            "gcmtype":"file","gcmlimit":50, "prop":"imageinfo","iiprop":"url|size|extmetadata",
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

def portraits():
    folder = ROOT / "portraits"
    folder.mkdir(exist_ok=True)
    rows, hashes, titles = [], set(), set()
    categories = ["Portrait photographs of women", "Portrait photographs of men",
                   "Front view portrait photographs of females", "Front view portrait photographs of male people",
                   "Black and white portrait photographs of women", "Black and white portrait photographs of men"]
    streams = [wikimedia_pages(c) for c in categories]
    for item in round_robin(streams):
        if len(rows) >= 500: break
        title = item.get("title","")
        if title in titles: continue
        titles.add(title)
        try:
            ii = item.get("imageinfo",[{}])[0]
            if not ii: continue
            mime = ii.get("mime","")
            if mime not in ("image/jpeg","image/png","image/webp"): continue
            if min(ii.get("width",0),ii.get("height",0)) < 600: continue
            if max(ii.get("width",0),ii.get("height",0)) > 16000: continue
            license_name = permitted_license(ii.get("extmetadata",{}))
            if not license_name:
                stats["unlicensed_for_pack"] += 1
                continue
            u = ii.get("thumburl") or ii.get("url")
            if not u: continue
            blob = get(u, binary=True)
            photo, h = make_jpg(blob, 512, 512, blackwhite=True, face=True)
            if too_similar(h, hashes):
                stats["portrait_duplicates"] += 1
                continue
            fname = f"portrait_{len(rows)+1:04d}.jpg"
            (folder/fname).write_bytes(photo)
            meta=ii.get("extmetadata",{})
            rows.append(dict(filename=fname,source=ii.get("descriptionurl", "https://commons.wikimedia.org/wiki/"+title.replace(" ","_")),
                license=license_name, creator=strip_html(meta.get("Artist",{}).get("value",""))))
            hashes.add(h)
            if len(rows) % 25 == 0: print("portraits",len(rows),"inspected",len(titles),flush=True)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            stats["face_or_format_rejected"] += 1
        except Exception as e:
            stats["portrait_errors"] += 1
            print("portrait error",title,str(e)[:130],flush=True)
    write_zip(folder, "500_real_portrait_photographs_CC0_PD.zip", rows)
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
