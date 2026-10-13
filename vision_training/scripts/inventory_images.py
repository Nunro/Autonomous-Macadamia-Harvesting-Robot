"""
inventory_images.py
MCTR 474 Final Year Project, Step 1: Read-only audit of the raw vision image archive.

THIS SCRIPT NEVER MODIFIES, MOVES OR RENAMES ANY IMAGE.
It only reads the images and writes two CSV reports plus a console summary.

Outputs (in OUTPUT_DIR):
  image_inventory.csv   one row per image (resolution, EXIF, session, hashes, quality metrics)
  near_duplicates.csv   pairs of visually near-identical images (burst shots, same scene)

Install once in your PyCharm virtual environment:
  pip install pillow numpy opencv-python
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import hashlib
import sys

import numpy as np
import cv2
from PIL import Image

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\nunro\OneDrive\Documents\NWU\2026\Semester 1"
                  r"\MCTR 474 (Final Year Project)\Vision_Images")
OUTPUT_DIR = Path(r"C:\MCTR474_Vision\reports")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
SESSION_GAP_MIN = 30      # a gap longer than this between photos starts a new session
NEAR_DUP_HAMMING = 4      # perceptual-hash bit difference counted as "near-duplicate" (0-64)
BLUR_PERCENTILE = 5       # the blurriest 5 % of images get flagged for manual review
DARK_THRESHOLD = 40       # mean grey level (0-255) below this is flagged "dark"
BRIGHT_THRESHOLD = 215    # mean grey level above this is flagged "overexposed"
METRIC_WIDTH = 640        # images are resized to this width before computing sharpness,
                          # so the metric is comparable across different resolutions


def read_exif(img):
    """Return (capture datetime, orientation tag, camera model) from EXIF, if present."""
    exif = img.getexif()
    sub_ifd = exif.get_ifd(0x8769)                      # Exif sub-IFD
    raw = sub_ifd.get(36867) or exif.get(306)           # DateTimeOriginal, else DateTime
    dt = None
    if raw:
        try:
            dt = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            dt = None
    orientation = exif.get(274, 1)                       # 1 = normal, anything else = rotated
    model = str(exif.get(272, "")).strip("\x00 ")
    return dt, orientation, model


def average_hash(img):
    """64-bit perceptual hash: similar-looking images give similar hashes."""
    small = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def quality_metrics(path):
    """Mean brightness and Laplacian-variance sharpness (higher = sharper)."""
    # np.fromfile + imdecode is used because cv2.imread is unreliable with some Windows paths
    grey = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise ValueError("OpenCV could not decode the image")
    scale = METRIC_WIDTH / grey.shape[1]
    grey = cv2.resize(grey, (METRIC_WIDTH, int(grey.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    return float(grey.mean()), float(cv2.Laplacian(grey, cv2.CV_64F).var())


def main():
    if not SOURCE_DIR.exists():
        sys.exit(f"Source folder not found:\n{SOURCE_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in SOURCE_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit("No image files found. Check SOURCE_DIR and EXTENSIONS.")
    print(f"Found {len(files)} image files. Auditing...\n")

    rows = []
    for i, path in enumerate(files, start=1):
        row = {"relative_path": str(path.relative_to(SOURCE_DIR)),
               "folder": str(path.parent.relative_to(SOURCE_DIR)),
               "extension": path.suffix.lower(),
               "size_kb": round(path.stat().st_size / 1024, 1),
               "md5": hashlib.md5(path.read_bytes()).hexdigest(),
               "error": ""}
        try:
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
                dt, row["orientation_tag"], row["camera_model"] = read_exif(img)
                row["datetime"] = dt
                row["_ahash"] = average_hash(img)
            row["brightness"], row["sharpness"] = quality_metrics(path)
        except Exception as e:                           # unreadable / HEIC / corrupt
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if i % 100 == 0:
            print(f"  processed {i}/{len(files)}")

    ok = [r for r in rows if not r["error"]]

    # --- Suggested capture sessions from EXIF time gaps ---
    timed = sorted((r for r in ok if r.get("datetime")), key=lambda r: r["datetime"])
    session, previous = 0, None
    for r in timed:
        if previous is None or (r["datetime"] - previous).total_seconds() > SESSION_GAP_MIN * 60:
            session += 1
        r["suggested_session"] = f"S{session:02d}"
        previous = r["datetime"]

    # --- Exact duplicates (identical file bytes) ---
    md5_counts = Counter(r["md5"] for r in rows)
    for r in rows:
        r["exact_duplicate"] = md5_counts[r["md5"]] > 1

    # --- Near-duplicates (perceptual hash), excluding exact duplicates ---
    near_pairs = []
    for a in range(len(ok)):
        for b in range(a + 1, len(ok)):
            if ok[a]["md5"] == ok[b]["md5"]:
                continue
            dist = bin(ok[a]["_ahash"] ^ ok[b]["_ahash"]).count("1")
            if dist <= NEAR_DUP_HAMMING:
                near_pairs.append((ok[a]["relative_path"], ok[b]["relative_path"], dist))

    # --- Quality flags (for manual review, never automatic deletion) ---
    blur_limit = np.percentile([r["sharpness"] for r in ok], BLUR_PERCENTILE) if ok else 0
    for r in ok:
        flags = []
        if r["sharpness"] <= blur_limit:
            flags.append("blurry")
        if r["brightness"] < DARK_THRESHOLD:
            flags.append("dark")
        if r["brightness"] > BRIGHT_THRESHOLD:
            flags.append("overexposed")
        if r["orientation_tag"] not in (1, None):
            flags.append("exif_rotated")
        r["review_flags"] = ";".join(flags)

    # --- Write reports ---
    fields = ["relative_path", "folder", "extension", "size_kb", "width", "height",
              "orientation_tag", "camera_model", "datetime", "suggested_session", "md5",
              "exact_duplicate", "brightness", "sharpness", "review_flags", "error"]
    with open(OUTPUT_DIR / "image_inventory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = dict(r)
            for key in ("brightness", "sharpness"):
                if isinstance(out.get(key), float):
                    out[key] = round(out[key], 2)
            writer.writerow(out)
    with open(OUTPUT_DIR / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "hamming_distance"])
        writer.writerows(near_pairs)

    # --- Console summary (paste this back into the chat) ---
    print("\n================ AUDIT SUMMARY ================")
    print(f"Total files:              {len(rows)}")
    print(f"Readable:                 {len(ok)}")
    print(f"Unreadable / errors:      {len(rows) - len(ok)}")
    print(f"Extensions:               {dict(Counter(r['extension'] for r in rows))}")
    print(f"Sub-folders:              {dict(Counter(r['folder'] for r in rows))}")
    print(f"Resolutions (WxH):        {dict(Counter((r['width'], r['height']) for r in ok).most_common(5))}")
    print(f"Camera models:            {dict(Counter(r['camera_model'] or 'none' for r in ok))}")
    print(f"EXIF-rotated images:      {sum('exif_rotated' in r['review_flags'] for r in ok)}")
    print(f"Images with timestamp:    {len(timed)}")
    if timed:
        print(f"Date range:               {timed[0]['datetime']}  to  {timed[-1]['datetime']}")
        print(f"Suggested sessions:       {dict(Counter(r['suggested_session'] for r in timed))}")
    print(f"Exact duplicate files:    {sum(r['exact_duplicate'] for r in rows)}")
    print(f"Near-duplicate pairs:     {len(near_pairs)}")
    print(f"Flagged blurry:           {sum('blurry' in r['review_flags'] for r in ok)}")
    print(f"Flagged dark:             {sum('dark' in r['review_flags'] for r in ok)}")
    print(f"Flagged overexposed:      {sum('overexposed' in r['review_flags'] for r in ok)}")
    print(f"\nReports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
