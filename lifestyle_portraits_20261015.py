#!/usr/bin/env python3
"""Produce genuinely COLOR modern adult face photographs; no artificial recoloring.
Sources: Wikimedia Commons licenses. See per-image provenance CSV and README.
This is distinct from the old grayscale artifact.
"""
from __future__ import annotations
import collections
import csv
import io
import json
import random
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import cv2
from asset_pack.collect import get, too_similar, wikimedia_pages, round_robin, strip_html, face_detector

random.seed(20261015)
OUT = Path("output")
OUT.mkdir(exist_ok=True)
IMAGES = OUT/"lifestyle_portraits"
IMAGES.mkdir(exist_ok=True)
T0 = time.monotonic()
LIMIT_SECONDS = 30*60
WORKERS = 12
TARGET = 500
MW = "https://commons.wikimedia.org/wiki/"
stats = collections.Counter()
records = []
hashes = set()
seen = set()

def color_license(md):
    name = md.get("LicenseShortName", {}).get("value", "").strip()
    url = md.get("LicenseUrl", {}).get("value", "").strip()
    check = name.lower()
    norm = url.lower()
    if check == "cc0" or "publicdomain/zero" in norm:
        return ("CC0", url)
    if check in ("public domain", "pd") or "publicdomain/mark" in norm:
        return ("Public domain", url)
    if re.match(r"^cc by(-sa)?[ -]\d",check):
        return (name, url)
    return None

def candidate(item):
    title=item.get("title","")
    # Exclude obvious celebrities, officials, uniforms and historical publicity images.
    BLOCK = re.compile(
        r"astronaut|nasa|spacesuit|space suit|cosmonaut|official.?portrait|"
        r"senator|congress|parliament|minister|politician|president|prime.minister|"
        r"actor|actress|celebrity|famous|movie.star|film.festival|red.carpet|"
        r"musician|singer|performer|stage|award|press.conference|"
        r"military|army|uniform|general|police|pilot|business.portrait|"
        r"passport|identity.card|headshot|mugshot|yearbook|historical|"
        r"baby|infant|toddler|school.children|children|kids|teenage|minor|"
        r"painting|sketch|drawing|cartoon|digital.art|illustration",
        re.IGNORECASE)
    if BLOCK.search(title):
        stats["unwanted_subject_title"] += 1
        return None
    if not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        stats["not_a_raster_photo"] += 1
        return None
    info=item.get("imageinfo",[{}])[0]
    if min(info.get("width",0),info.get("height",0))<500:
        stats["too_small"] += 1
        return None
    info_md=info.get("extmetadata",{})
    license_data=color_license(info_md)
    if not license_data:
        stats["license_not_suitable"] += 1
        return None
    thumb=info.get("thumburl") or info.get("url")
    if not thumb:
        return None
    return {"url": thumb,
            "source": info.get("descriptionurl", MW+title.replace(" ","_")),
            "license": license_data[0], "license_url": license_data[1],
            "creator": strip_html(info_md.get("Artist",{}).get("value","")),
            "name": title}

def detect_color(blob):
    # Reject intrinsically grayscale/historical B&W images, rather than making fake color.
    with Image.open(io.BytesIO(blob)) as im:
        rgb = np.asarray(im.convert("RGB").resize((72,72)), dtype=np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return float(chroma.mean()) >= 7.5 and float((chroma > 18).mean()) >= .15

def make_lifestyle_jpg(blob):
    """Fit one clear person into a casual, half-body square composition; no tight face crop."""
    with Image.open(io.BytesIO(blob)) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    if min(im.size) < 600:
        raise ValueError("photo below 600 px")
    arr = np.asarray(im)
    scale = min(1.0, 900/max(im.size))
    sm = cv2.resize(arr,None,fx=scale,fy=scale) if scale<1 else arr
    gray = cv2.cvtColor(sm,cv2.COLOR_RGB2GRAY)
    faces = face_detector().detectMultiScale(gray,scaleFactor=1.08,minNeighbors=5,minSize=(38,38))
    if len(faces) != 1:
        raise ValueError("not exactly one recognizable person")
    x,y,w,h = (int(v/scale) for v in faces[0])
    # Face should be sufficiently visible but not dominate frame.
    if h < 0.055 * im.height or h > 0.30 * im.height:
        raise ValueError("face is either too tiny or too close")
    # Keep shoulders and torso, avoid cropping to a giant face.
    side = int(max(w*5.4,h*5.4, min(im.size)*0.68))
    side = min(side,*im.size)
    if h / side > 0.26:
        raise ValueError("face too dominant for half-body photo")
    cx=x+w/2
    cy=y+h/2
    left=int(max(0,min(im.width-side,cx-side*.5)))
    top=int(max(0,min(im.height-side,cy-side*.25)))
    im=im.crop((left,top,left+side,top+side))
    im=im.resize((512,512),Image.Resampling.LANCZOS)
    target=io.BytesIO()
    im.save(target,"JPEG",quality=89,optimize=True)
    v=np.asarray(im.resize((9,8)).convert("L"),dtype=np.int16)
    hsh=sum(int(t)<<i for i,t in enumerate((v[:,1:]>v[:,:-1]).ravel()))
    return target.getvalue(),hsh

def download_candidate(c):
    blob=get(c["url"],binary=True)
    if not detect_color(blob):
        raise ValueError("original image is grayscale or near-monochrome")
    jpg, similarity=make_lifestyle_jpg(blob)
    if not detect_color(jpg):
        raise ValueError("cropped portrait is grayscale or near-monochrome")
    return c,jpg,similarity

def save_result(c,jpg,h):
    if too_similar(h,hashes):
        stats["similar_rejected"]+=1
        return
    hashes.add(h)
    fname=f"lifestyle_portrait_{len(records)+1:04d}.jpg"
    (IMAGES/fname).write_bytes(jpg)
    records.append({"filename":fname,
         "source":c["source"],"license":c["license"],"license_url":c["license_url"],"creator":c["creator"]})
    if len(records)%25==0:
        print(f"[LIFESTYLE PROGRESS] {len(records)}/{TARGET} images; scanned={len(seen)}; elapsed={time.monotonic()-T0:.0f}s; {dict(stats)}",flush=True)

def main():
    categories=[
        "Selfies of standing women", "Selfies of standing men",
        "Selfies of sitting women", "Selfies of sitting men",
        "People in parks", "People on park benches", "People in gardens and parks",
        "People smiling", "People outdoors", "People at beaches",
        "People walking", "Outdoor portrait photographs",
        "21st-century self-portrait photographs of women",
        "21st-century self-portrait photographs of men",
        "Portrait photographs of women", "Portrait photographs of men"
    ]
    stream=iter(round_robin(wikimedia_pages(c) for c in categories))
    exhausted=False
    print("[LIFESTYLE] Start: 500 real natural lifestyle half-body images; exclude tight closeups and obvious celebrity/publicity portraits",flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        while len(records)<TARGET and time.monotonic()-T0 < LIMIT_SECONDS and not exhausted:
            candidates=[]
            while len(candidates)<WORKERS*3 and time.monotonic()-T0 < LIMIT_SECONDS:
                try:
                    info=next(stream)
                except StopIteration:
                    exhausted=True
                    break
                title=info.get("title","")
                if title in seen:
                    continue
                seen.add(title)
                c=candidate(info)
                if c: candidates.append(c)
                if len(seen)%400==0:
                    print(f"[LIFESTYLE SEARCH] scanned={len(seen)} candidates={len(candidates)} saved={len(records)}",flush=True)
            if not candidates:
                continue
            futs=[executor.submit(download_candidate,c) for c in candidates]
            for fut in as_completed(futs):
                try:
                    c,jpg,h=fut.result()
                    if len(records)<TARGET:
                        save_result(c,jpg,h)
                except ValueError as ex:
                    if "grayscale" in str(ex):
                        stats["grayscale_original_rejected"]+=1
                    else:
                        stats["face_or_quality_rejected"]+=1
                except Exception as ex:
                    stats["download_or_detector_error"]+=1
                    if stats["download_or_detector_error"]<=6:
                        print("[LIFESTYLE ERROR]",repr(ex)[:200],flush=True)
    filename=f"{len(records)}_lifestyle_color_portraits.zip"
    target=OUT/filename
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED) as z:
        for img in sorted(IMAGES.glob("*.jpg")):
            z.write(img,img.name)
        stream=io.StringIO()
        w=csv.DictWriter(stream, fieldnames=["filename","source","license","license_url","creator"])
        w.writeheader();w.writerows(records)
        z.writestr("SOURCES_AND_LICENSES.csv",stream.getvalue().encode("utf-8-sig"))
        z.writestr("USAGE_AND_ATTRIBUTION.txt",
        "Images are real color photographs, framed to retain shoulders and context instead of tight face crops.\n"
        "Each file's copyright license, photographer and source URL are in SOURCES_AND_LICENSES.csv.\n"
        "CC BY and CC BY-SA images require attribution, and SA may require licensing adaptations similarly.\n"
        "You must check each image's license and provide proper attribution in each use.\n"
        "Photographic copyright licenses DO NOT provide permission to impersonate real people.\n"
        "Do not falsely present the person shown as the account owner or as endorsing your business.\n"
        "Model/publicity/privacy rights still apply. Use photographs as social profile avatars only with the depicted person's permission where needed.\n")
    manifest={"count":len(records),"target":TARGET,"duration_seconds":round(time.monotonic()-T0,1),
              "stats":dict(stats),"archive":filename,"genuine_color_not_recolored":True,"composition":"half body / natural scene, filtered by single face size"}
    (OUT/"lifestyle_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("[LIFESTYLE FINAL]",json.dumps(manifest),flush=True)
    if len(records)!=TARGET: sys.exit("Fewer than 500 qualifying lifestyle photos; partial archive uploaded for review")

if __name__=="__main__":
    main()
