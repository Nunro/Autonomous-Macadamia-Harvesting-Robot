"""
standardise_rename.py
MCTR 474 Final Year Project, Step 3: Standardise and rename the image dataset.

THE ORIGINAL IMAGES ARE ONLY READ, NEVER MODIFIED, MOVED OR RENAMED.
New standardised copies are written to OUTPUT_DIR.

Needs (from Steps 1 and 1b):
  reports/image_inventory.csv, reports/duplicate_groups.csv, reports/clusters.csv

For every kept image:
  1. Applies the EXIF rotation permanently, then strips all metadata.
  2. Converts the colour profile to sRGB (iPhone photos are stored in Display P3).
  3. Converts to RGB (transparency, if any, is flattened onto white).
  4. Resizes so the longest side is at most MAX_SIDE pixels (never upscales).
  5. Saves as JPEG (quality 95) named dataset_0001.jpg, dataset_0002.jpg, ...
     numbered in capture-time order.

Outputs:
  data/standardised/dataset_XXXX.jpg
  reports/rename_mapping.csv   (full audit trail, including skipped duplicates)
"""

from pathlib import Path
from collections import Counter
from datetime import datetime
from io import BytesIO
import csv
import re
import sys

from PIL import Image, ImageOps, ImageCms

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\nunro\OneDrive\Documents\NWU\2026\Semester 1"
                  r"\MCTR 474 (Final Year Project)\Vision_Images")
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
REPORT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardised"

MAX_SIDE = 1920          # longest side after resizing (pixels)
JPEG_QUALITY = 95
NAME_PREFIX = "dataset_"

WHATSAPP_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})")
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def read_csv(path):
    if not path.exists():
        sys.exit(f"Missing {path}. Run the Step 1 and Step 1b scripts first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def source_type(row):
    name = row["relative_path"].lower()
    model = row["camera_model"].lower()
    if "whatsapp" in name:
        return "WhatsApp"
    if "canon" in model:
        return "Canon"
    if "iphone" in model:
        return "iPhone"
    return "No-EXIF"


def capture_time(row):
    """Return (datetime or None, source of the timestamp)."""
    if row["datetime"]:
        return datetime.fromisoformat(row["datetime"]), "exif"
    m = WHATSAPP_TIME.search(Path(row["relative_path"]).name)
    if m:
        date, hh, mm, ss = m.groups()
        return datetime.fromisoformat(f"{date} {int(hh):02d}:{mm}:{ss}"), "whatsapp_filename"
    return None, "none"


def standardise(path):
    """Load one image and return (standardised RGB image, info dict)."""
    info = {}
    with Image.open(path) as img:
        img.load()
        info["exif_rotated"] = "yes" if img.getexif().get(274, 1) not in (1, None) else "no"
        icc = img.info.get("icc_profile")
        img = ImageOps.exif_transpose(img)
        info["orig_w"], info["orig_h"] = img.size

        # Flatten transparency onto white, then make sure we have RGB
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # Colour profile to sRGB (keeps colours consistent once metadata is stripped)
        info["icc_converted"] = "no"
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(BytesIO(icc))
                img = ImageCms.profileToProfile(img, src, SRGB_PROFILE, outputMode="RGB")
                info["icc_converted"] = "yes"
            except (ImageCms.PyCMSError, OSError):
                info["icc_converted"] = "failed_kept_as_is"

    # Resize, never upscale
    scale = min(1.0, MAX_SIDE / max(img.size))
    if scale < 1.0:
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    info["new_w"], info["new_h"] = img.size
    info["scale"] = round(scale, 4)
    return img, info


def main():
    # --- Safety checks ---
    if not SOURCE_DIR.exists():
        sys.exit(f"Source folder not found:\n{SOURCE_DIR}")
    if OUTPUT_DIR.resolve().is_relative_to(SOURCE_DIR.resolve()):
        sys.exit("OUTPUT_DIR must not be inside SOURCE_DIR.")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        sys.exit(f"{OUTPUT_DIR} is not empty. Delete its contents first if you want to re-run.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = [r for r in read_csv(REPORT_DIR / "image_inventory.csv") if not r["error"]]
    duplicates = read_csv(REPORT_DIR / "duplicate_groups.csv")
    clusters = {r["relative_path"]: r for r in read_csv(REPORT_DIR / "clusters.csv")}

    # Duplicates: keep the copy WITHOUT a "(1)"-style suffix (the original),
    # then the shortest name; map each skipped copy to the copy that is kept
    groups = {}
    for r in duplicates:
        groups.setdefault(r["group"], []).append(r["relative_path"])
    skipped = {}
    for paths in groups.values():
        keep = min(paths, key=lambda p: (bool(re.search(r"\(\d+\)", Path(p).name)), len(p), p))
        skipped.update({p: keep for p in paths if p != keep})

    # Order kept images by capture time; untimed images last, then by original name
    kept = []
    for r in inventory:
        if r["relative_path"] in skipped:
            continue
        r["ts"], r["ts_source"] = capture_time(r)
        kept.append(r)
    kept.sort(key=lambda r: (r["ts"] is None, r["ts"] or datetime.max, r["relative_path"]))

    print(f"{len(inventory)} images in inventory, {len(skipped)} duplicate copies skipped, "
          f"{len(kept)} to standardise.\nThis takes a few minutes (full-resolution decoding)...\n")

    mapping = []
    for number, r in enumerate(kept, start=1):
        new_name = f"{NAME_PREFIX}{number:04d}.jpg"
        img, info = standardise(SOURCE_DIR / r["relative_path"])
        img.save(OUTPUT_DIR / new_name, "JPEG", quality=JPEG_QUALITY, optimize=True)
        c = clusters.get(r["relative_path"], {})
        mapping.append({
            "new_filename": new_name,
            "original_filename": r["relative_path"],
            "status": "kept",
            "duplicate_of": "",
            "source": source_type(r),
            "camera_model": r["camera_model"],
            "original_width": info["orig_w"], "original_height": info["orig_h"],
            "new_width": info["new_w"], "new_height": info["new_h"],
            "scale_factor": info["scale"],
            "timestamp": r["ts"].isoformat(sep=" ") if r["ts"] else "",
            "timestamp_source": r["ts_source"],
            "exif_rotated": info["exif_rotated"],
            "icc_converted": info["icc_converted"],
            "cluster_id": c.get("cluster_id", ""),
            "cluster_size": c.get("cluster_size", ""),
            "review_flags": r["review_flags"],
        })
        if number % 25 == 0:
            print(f"  written {number}/{len(kept)}")

    for path, kept_copy in skipped.items():
        row = next(r for r in inventory if r["relative_path"] == path)
        mapping.append({
            "new_filename": "", "original_filename": path, "status": "skipped_duplicate",
            "duplicate_of": kept_copy, "source": source_type(row),
            "camera_model": row["camera_model"], "review_flags": row["review_flags"],
        })

    fields = ["new_filename", "original_filename", "status", "duplicate_of", "source",
              "camera_model", "original_width", "original_height", "new_width", "new_height",
              "scale_factor", "timestamp", "timestamp_source", "exif_rotated",
              "icc_converted", "cluster_id", "cluster_size", "review_flags"]
    with open(REPORT_DIR / "rename_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(mapping)

    # --- Verification: re-open every output file ---
    written = sorted(OUTPUT_DIR.glob(f"{NAME_PREFIX}*.jpg"))
    problems = []
    for p in written:
        with Image.open(p) as im:
            if im.mode != "RGB" or max(im.size) > MAX_SIDE or im.getexif().get(274):
                problems.append(p.name)
    size_mb = sum(p.stat().st_size for p in written) / 1024 ** 2

    kept_rows = [m for m in mapping if m["status"] == "kept"]
    print("\n============ STANDARDISE AND RENAME SUMMARY ============")
    print(f"Images written:           {len(written)} ({written[0].name} to {written[-1].name})")
    print(f"Duplicate copies skipped: {len(skipped)}")
    print(f"By source:                {dict(Counter(m['source'] for m in kept_rows))}")
    print(f"Resized (downscaled):     {sum(m['scale_factor'] < 1 for m in kept_rows)}")
    print(f"Rotation applied:         {sum(m['exif_rotated'] == 'yes' for m in kept_rows)}")
    print(f"Colour profile to sRGB:   {dict(Counter(m['icc_converted'] for m in kept_rows))}")
    print(f"Timestamp sources:        {dict(Counter(m['timestamp_source'] for m in kept_rows))}")
    print(f"Output resolutions:       "
          f"{dict(Counter((m['new_width'], m['new_height']) for m in kept_rows).most_common(6))}")
    print(f"Total output size:        {size_mb:.1f} MB")
    print(f"Verification problems:    {len(problems)} {problems[:10] if problems else ''}")
    print(f"\nImages:  {OUTPUT_DIR}\nMapping: {REPORT_DIR / 'rename_mapping.csv'}")


if __name__ == "__main__":
    main()
