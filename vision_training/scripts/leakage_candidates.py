"""
leakage_candidates.py
MCTR 474 Final Year Project: Build side-by-side review sheets of possible test-set leakage.

For every TEST image, its possible "twins" in the TRAINING set are:
  - neighbours in capture order: dataset numbers within +-NEIGHBOUR_RANGE that are in train
    (photos taken seconds apart, before the balls were rearranged)
  - its same-layout match from leakage_check.py (if any)
  - its most similar training image from leakage_check.py (if similarity >= SIM_SHOW)

Each sheet row shows the TEST image (left, red label) next to its candidate twins,
with the labelled ball boxes drawn in, so layouts are easy to compare.

YOU then write the names of the test images that show THE SAME SCENE as a training image
into reports/leakage_review/contaminated_test_images.txt (one name per line).

Reads (never modifies):  data/dataset_v1/, reports/split_assignment.csv,
                         reports/leakage_check/leakage_pairs.csv
Writes:                  reports/leakage_review/sheet_XX.jpg
                         reports/leakage_review/contaminated_test_images.txt (empty template)
"""

from pathlib import Path
import csv
import re

from PIL import Image, ImageDraw, ImageOps

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
DATASET_DIR = PROJECT_DIR / "data" / "dataset_v1"
SPLIT_CSV = PROJECT_DIR / "reports" / "split_assignment.csv"
PAIRS_CSV = PROJECT_DIR / "reports" / "leakage_check" / "leakage_pairs.csv"
OUT_DIR = PROJECT_DIR / "reports" / "leakage_review"

NEIGHBOUR_RANGE = 2     # look at training images up to 2 numbers before/after
SIM_SHOW = 0.70         # also show the most similar training image if it scores this high
MAX_PARTNERS = 4        # max training images shown next to one test image
ROWS_PER_SHEET = 8
CELL = 360


def number(name):
    return int(re.search(r"(\d{4})", name).group(1))


def thumb(split, name, caption, colour):
    with Image.open(DATASET_DIR / "images" / split / name) as im:
        im.draft("RGB", (1024, 1024))
        im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for line in (DATASET_DIR / "labels" / split / f"{Path(name).stem}.txt").read_text().splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                        outline=(255, 0, 255), width=max(2, W // 300))
    t = ImageOps.contain(im, (CELL - 6, CELL - 30))
    cell = Image.new("RGB", (CELL, CELL), "white")
    cell.paste(t, (3, 26))
    ImageDraw.Draw(cell).rectangle([0, 0, CELL, 22], fill=colour)
    ImageDraw.Draw(cell).text((5, 5), caption, fill="white")
    return cell


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        split_of = {r["name"]: r["split"] for r in csv.DictReader(f)}
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        pairs = {r["test_image"]: r for r in csv.DictReader(f)}
    train_by_number = {number(n): n for n, s in split_of.items() if s == "train"}
    tests = sorted(n for n, s in split_of.items() if s == "test")

    rows = []
    for t in tests:
        partners = {}
        p = pairs.get(t, {})
        if p.get("same_layout") == "YES":
            partners[p["closest_layout_train_image"]] = "SAME LAYOUT?"
        if p and float(p["similarity"]) >= SIM_SHOW:
            partners.setdefault(p["nearest_train_image"], f"sim {float(p['similarity']):.2f}")
        offsets = sorted((k for k in range(-NEIGHBOUR_RANGE, NEIGHBOUR_RANGE + 1) if k),
                         key=abs)                                   # -1, +1, -2, +2
        for k in offsets:
            n = train_by_number.get(number(t) + k)
            if n:
                partners.setdefault(n, f"neighbour {k:+d}")
        if partners:
            rows.append((t, list(partners.items())[:MAX_PARTNERS]))

    # Contact sheets: TEST (red) | candidate training twins (grey)
    for s in range(0, len(rows), ROWS_PER_SHEET):
        chunk = rows[s:s + ROWS_PER_SHEET]
        sheet = Image.new("RGB", ((MAX_PARTNERS + 1) * CELL + 20, len(chunk) * (CELL + 10)), "white")
        for r, (t, partners) in enumerate(chunk):
            y = r * (CELL + 10)
            sheet.paste(thumb("test", t, f"TEST {t}", (200, 0, 0)), (0, y))
            for c, (n, why) in enumerate(partners, start=1):
                sheet.paste(thumb("train", n, f"train {n}  ({why})", (90, 90, 90)), (c * CELL + 20, y))
        sheet.save(OUT_DIR / f"sheet_{s // ROWS_PER_SHEET + 1:02d}.jpg", quality=88)

    template = OUT_DIR / "contaminated_test_images.txt"
    if not template.exists():
        template.write_text(
            "# Write one TEST image name per line (e.g. dataset_0029.jpg) for every test image\n"
            "# that shows THE SAME SCENE as one of its training partners on the sheets:\n"
            "# same balls in the same places on the same ground, regardless of small camera\n"
            "# movement or which way the balls are turned. Lines starting with # are ignored.\n")

    print("\n============ LEAKAGE REVIEW SHEETS ============")
    print(f"Test images:                     {len(tests)}")
    print(f"Test images with candidate twins: {len(rows)}")
    print(f"Sheets written:                  {(len(rows) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET}")
    print(f"\nSheets:   {OUT_DIR}\nYour list: {template}")


if __name__ == "__main__":
    main()
