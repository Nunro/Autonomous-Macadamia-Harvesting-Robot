"""
review_similarity.py
MCTR 474 Final Year Project, Step 1b: Read-only similarity review of the raw image archive.

THIS SCRIPT NEVER MODIFIES, MOVES OR RENAMES ANY IMAGE.
It needs reports/image_inventory.csv from inventory_images.py (Step 1).

What it does:
  1. Lists exact duplicate groups (identical file bytes).
  2. Groups visually similar images into clusters. These clusters become the
     split groups later, so similar images never end up in both train and test.
  3. Finds similar pairs from DIFFERENT sources (e.g. a WhatsApp copy of an iPhone photo).
  4. Builds contact sheets (thumbnail grids) so you can judge everything by eye.

Outputs (in reports/):
  duplicate_groups.csv, clusters.csv, cross_source_pairs.csv
  contact_sheets/*.jpg
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import sys

import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\nunro\OneDrive\Documents\NWU\2026\Semester 1"
                  r"\MCTR 474 (Final Year Project)\Vision_Images")
REPORT_DIR = Path(r"C:\MCTR474_Vision\reports")
INVENTORY_CSV = REPORT_DIR / "image_inventory.csv"
SHEET_DIR = REPORT_DIR / "contact_sheets"

THUMB_SIZE = 32              # images are compared as 32x32 greyscale thumbnails
SIMILARITY_THRESHOLD = 0.92  # correlation (0 to 1) above which two images count as "similar"
CLUSTERS_TO_SHOW = 8         # contact sheets are made for this many of the largest clusters
SHEET_THUMB = 220            # thumbnail size on the contact sheets (pixels)
SHEET_COLUMNS = 6
SHEET_MAX_IMAGES = 36        # max thumbnails per contact sheet


def load_rgb(path):
    """Open an image fast (JPEG draft mode), apply EXIF rotation, return RGB."""
    with Image.open(path) as img:
        if img.format == "JPEG":
            img.draft("RGB", (512, 512))        # decode at reduced size, much faster
        img = ImageOps.exif_transpose(img)       # so rotated copies still match
        return img.convert("RGB")


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


def short_label(name, extra=""):
    label = name if len(name) <= 30 else ".." + name[-28:]
    return f"{label} {extra}".strip()


def contact_sheet(items, title, out_path):
    """items: list of (thumbnail, label). Saves a labelled thumbnail grid."""
    items = items[:SHEET_MAX_IMAGES]
    if not items:
        return
    rows = math.ceil(len(items) / SHEET_COLUMNS)
    cell_w, cell_h = SHEET_THUMB, SHEET_THUMB + 18
    sheet = Image.new("RGB", (SHEET_COLUMNS * cell_w, rows * cell_h + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for k, (thumb, label) in enumerate(items):
        x = (k % SHEET_COLUMNS) * cell_w
        y = 30 + (k // SHEET_COLUMNS) * cell_h
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        draw.text((x + 4, y + SHEET_THUMB + 2), label, fill="black")
    sheet.save(out_path, quality=90)


def main():
    if not INVENTORY_CSV.exists():
        sys.exit(f"Run inventory_images.py first, {INVENTORY_CSV} not found.")
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    with open(INVENTORY_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    n = len(rows)
    print(f"Loading {n} images (reduced size, read-only)...")

    vectors, thumbs = [], []
    for i, r in enumerate(rows, start=1):
        img = load_rgb(SOURCE_DIR / r["relative_path"])
        grey = img.convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BOX)
        v = np.asarray(grey, dtype=np.float64).flatten()
        v -= v.mean()
        norm = np.linalg.norm(v)
        vectors.append(v / norm if norm > 0 else v)
        img.thumbnail((SHEET_THUMB, SHEET_THUMB))
        thumbs.append(img)
        r["source"] = source_type(r)
        if i % 100 == 0:
            print(f"  loaded {i}/{n}")

    # Pearson correlation between every pair of thumbnails
    V = np.vstack(vectors)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)

    # --- 1. Exact duplicates ---
    by_md5 = defaultdict(list)
    for i, r in enumerate(rows):
        by_md5[r["md5"]].append(i)
    dup_groups = [g for g in by_md5.values() if len(g) > 1]
    with open(REPORT_DIR / "duplicate_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "relative_path", "keep"])
        for gi, g in enumerate(dup_groups, start=1):
            for k, i in enumerate(g):
                w.writerow([gi, rows[i]["relative_path"], "yes" if k == 0 else "no"])

    # --- 2. Clusters of similar images (union-find on similar pairs) ---
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where(np.triu(S, k=1) >= SIMILARITY_THRESHOLD)
    for a, b in zip(ii, jj):
        parent[find(a)] = find(b)
    members = defaultdict(list)
    for i in range(n):
        members[find(i)].append(i)
    clusters = sorted(members.values(), key=len, reverse=True)

    with open(REPORT_DIR / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "source", "cluster_id", "cluster_size",
                    "max_similarity", "review_flags"])
        for ci, c in enumerate(clusters, start=1):
            for i in c:
                w.writerow([rows[i]["relative_path"], rows[i]["source"], f"C{ci:03d}",
                            len(c), round(float(S[i].max()), 3), rows[i]["review_flags"]])

    # --- 3. Similar pairs from different sources ---
    cross = [(a, b, S[a, b]) for a, b in zip(ii, jj)
             if rows[a]["source"] != rows[b]["source"]]
    cross.sort(key=lambda t: -t[2])
    with open(REPORT_DIR / "cross_source_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "source_a", "image_b", "source_b", "similarity"])
        for a, b, s in cross:
            w.writerow([rows[a]["relative_path"], rows[a]["source"],
                        rows[b]["relative_path"], rows[b]["source"], round(float(s), 3)])

    # --- 4. Contact sheets ---
    blurry = sorted((i for i in range(n) if "blurry" in rows[i]["review_flags"]),
                    key=lambda i: float(rows[i]["sharpness"]))
    contact_sheet([(thumbs[i], short_label(rows[i]["relative_path"],
                                           f"s={float(rows[i]['sharpness']):.0f}"))
                   for i in blurry],
                  "Blurriest 5% (sharpest last). Keep mild blur, drop only unusable.",
                  SHEET_DIR / "01_blurriest.jpg")

    contact_sheet([(thumbs[i], short_label(rows[i]["relative_path"], f"G{gi}"))
                   for gi, g in enumerate(dup_groups, start=1) for i in g],
                  "Exact duplicates (same G number = identical files)",
                  SHEET_DIR / "02_exact_duplicates.jpg")

    pair_items = []
    for a, b, s in cross[:SHEET_MAX_IMAGES // 2]:
        pair_items += [(thumbs[a], short_label(rows[a]["relative_path"], rows[a]["source"])),
                       (thumbs[b], short_label(rows[b]["relative_path"], f"r={s:.2f}"))]
    contact_sheet(pair_items, "Cross-source similar pairs (left/right in each row pair)",
                  SHEET_DIR / "03_cross_source_pairs.jpg")

    shown = [c for c in clusters if len(c) > 1][:CLUSTERS_TO_SHOW]
    for ci, c in enumerate(shown, start=1):
        contact_sheet([(thumbs[i], short_label(rows[i]["relative_path"])) for i in c],
                      f"Cluster {ci}: {len(c)} images (first {min(len(c), SHEET_MAX_IMAGES)} shown)",
                      SHEET_DIR / f"cluster_{ci:02d}.jpg")

    # --- Console summary ---
    sizes = [len(c) for c in clusters]
    print("\n============ SIMILARITY REVIEW SUMMARY ============")
    print(f"Images reviewed:            {n}")
    print(f"Sources:                    {dict(Counter(r['source'] for r in rows))}")
    print(f"Similarity threshold:       {SIMILARITY_THRESHOLD}")
    print(f"Exact duplicate groups:     {len(dup_groups)} "
          f"({sum(len(g) - 1 for g in dup_groups)} files to remove)")
    print(f"Total clusters:             {len(clusters)}")
    print(f"Singletons (unique images): {sizes.count(1)}")
    print(f"Clusters with 2+ images:    {sum(s > 1 for s in sizes)}")
    print(f"Largest 10 cluster sizes:   {sizes[:10]}")
    for ci, c in enumerate(shown[:5], start=1):
        print(f"  Cluster {ci} sources:     {dict(Counter(rows[i]['source'] for i in c))}")
    print(f"Cross-source similar pairs: {len(cross)}")
    print(f"\nCSV reports and contact sheets written to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
