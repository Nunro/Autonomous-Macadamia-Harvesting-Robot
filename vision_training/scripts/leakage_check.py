"""
leakage_check.py
MCTR 474 Final Year Project: Check the frozen test set for leakage from the training set.

Question answered: is any test image practically the same photo as a training image?

Two independent checks for every test image, against the 335 ORIGINAL training images
(augmented copies are left out, they are derived from those originals):
  1. Image similarity : correlation of 64x64 greyscale thumbnails (1.0 = identical).
  2. Ball layout      : same number of balls (3 or more) in the same arrangement,
                        allowing the whole layout to be shifted (camera moved slightly).
                        Layout is only judged for 3+ balls: one or two balls "match" too easily.

Reads (never modifies):  data/dataset_v1/images|labels/train|test
Writes:                  reports/leakage_check/leakage_pairs.csv
                         reports/leakage_check/01_most_similar_pairs.jpg
                         reports/leakage_check/02_layout_matches.jpg   (only if any are found)
"""

from pathlib import Path
import csv
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
DATASET_DIR = PROJECT_DIR / "data" / "dataset_v1"
OUT_DIR = PROJECT_DIR / "reports" / "leakage_check"

THUMB = 64                 # thumbnail size for the similarity measure
LAYOUT_MIN_BALLS = 3       # layouts with fewer balls are not judged
LAYOUT_TOLERANCE = 0.03    # max ball-centre mismatch after removing the shift (fraction of image)
PAIRS_TO_SHOW = 20         # most similar test/train pairs on the contact sheet
CELL = 300                 # thumbnail size on the contact sheets


def load(path):
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        return im.convert("RGB")


def vector(img):
    v = np.asarray(img.convert("L").resize((THUMB, THUMB), Image.Resampling.BOX),
                   dtype=np.float64).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def centres(label_path):
    pts = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            pts.append((x, y))
    return np.array(pts)


def layout_distance(a, b):
    """Mean centre mismatch after removing the average shift; None if not comparable."""
    if len(a) < LAYOUT_MIN_BALLS or len(a) != len(b):
        return None
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    used, dists = set(), []
    for p in a0:                                  # greedy nearest-centre matching
        d = [(np.linalg.norm(p - q), j) for j, q in enumerate(b0) if j not in used]
        dist, j = min(d)
        used.add(j)
        dists.append(dist)
    return max(dists)


def pair_sheet(pairs, title, out_path):
    """pairs: list of (test_img, train_img, caption). Two pairs per row."""
    if not pairs:
        return
    rows = (len(pairs) + 1) // 2
    sheet = Image.new("RGB", (4 * CELL + 30, rows * (CELL + 24) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for k, (a, b, caption) in enumerate(pairs):
        x0 = (k % 2) * (2 * CELL + 30)
        y0 = 30 + (k // 2) * (CELL + 24)
        for m, img in enumerate((a, b)):
            t = ImageOps.contain(img, (CELL - 4, CELL - 4))
            sheet.paste(t, (x0 + m * CELL + 2, y0))
        draw.text((x0 + 4, y0 + CELL + 4), caption, fill="black")
    sheet.save(out_path, quality=90)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_imgs = sorted((DATASET_DIR / "images" / "test").glob("*.jpg"))
    train_imgs = sorted((DATASET_DIR / "images" / "train").glob("*.jpg"))
    print(f"Comparing {len(test_imgs)} test images with {len(train_imgs)} training originals...")

    def info(paths, split):
        imgs = [load(p) for p in paths]
        vecs = np.vstack([vector(i) for i in imgs])
        cents = [centres(DATASET_DIR / "labels" / split / f"{p.stem}.txt") for p in paths]
        return imgs, vecs, cents

    te_img, te_vec, te_cent = info(test_imgs, "test")
    tr_img, tr_vec, tr_cent = info(train_imgs, "train")
    S = te_vec @ tr_vec.T                          # test x train similarity matrix

    rows, layout_hits = [], []
    for i, tp in enumerate(test_imgs):
        j = int(S[i].argmax())
        best_layout = None
        for k in range(len(train_imgs)):
            d = layout_distance(te_cent[i], tr_cent[k])
            if d is not None and (best_layout is None or d < best_layout[0]):
                best_layout = (d, k)
        hit = best_layout is not None and best_layout[0] <= LAYOUT_TOLERANCE
        if hit:
            layout_hits.append((i, best_layout[1], best_layout[0]))
        rows.append({
            "test_image": tp.name, "balls": len(te_cent[i]),
            "nearest_train_image": train_imgs[j].name, "similarity": round(float(S[i, j]), 3),
            "layout_judged": "yes" if len(te_cent[i]) >= LAYOUT_MIN_BALLS else "no",
            "closest_layout_train_image": train_imgs[best_layout[1]].name if best_layout else "",
            "layout_mismatch": round(best_layout[0], 4) if best_layout else "",
            "same_layout": "YES" if hit else "no",
        })

    rows.sort(key=lambda r: -r["similarity"])
    with open(OUT_DIR / "leakage_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    name_to_idx = {p.name: n for n, p in enumerate(test_imgs)}
    train_idx = {p.name: n for n, p in enumerate(train_imgs)}
    pair_sheet([(te_img[name_to_idx[r["test_image"]]], tr_img[train_idx[r["nearest_train_image"]]],
                 f"{r['test_image']} | {r['nearest_train_image']} | sim {r['similarity']:.3f}")
                for r in rows[:PAIRS_TO_SHOW]],
               "Most similar test (left) / train (right) pairs", OUT_DIR / "01_most_similar_pairs.jpg")
    pair_sheet([(te_img[i], tr_img[k], f"{test_imgs[i].name} | {train_imgs[k].name} | "
                 f"mismatch {d:.3f}") for i, k, d in layout_hits],
               "Same ball layout: test (left) / train (right)", OUT_DIR / "02_layout_matches.jpg")

    sims = [r["similarity"] for r in rows]
    judged = sum(r["layout_judged"] == "yes" for r in rows)
    print("\n================ LEAKAGE CHECK SUMMARY ================")
    print(f"Test images checked:              {len(rows)}")
    print(f"Nearest-train similarity:         min {min(sims):.3f}, median "
          f"{statistics.median(sims):.3f}, max {max(sims):.3f}")
    print(f"Test images with similarity >= 0.90: {sum(s >= 0.90 for s in sims)}")
    print(f"Test images with similarity >= 0.80: {sum(s >= 0.80 for s in sims)}")
    print(f"Layouts judged (3+ balls):        {judged}")
    print(f"Same layout found in training:    {len(layout_hits)}")
    for i, k, d in layout_hits:
        print(f"  - {test_imgs[i].name}  ~  {train_imgs[k].name}  (mismatch {d:.3f})")
    print(f"\nTop 5 most similar pairs:")
    for r in rows[:5]:
        print(f"  {r['test_image']}  ~  {r['nearest_train_image']}   sim {r['similarity']:.3f}")
    print(f"\nReports: {OUT_DIR}")


if __name__ == "__main__":
    main()
