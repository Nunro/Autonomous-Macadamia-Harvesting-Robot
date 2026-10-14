"""
split_dataset.py
MCTR 474 Final Year Project, Step 7: Split the labelled dataset into train / val / test.

Rules:
  - Images in the same similarity cluster (repeat shots of one layout) always go
    to the SAME split, so no near-copy of a test image is ever trained on.
  - The split is stratified by source (iPhone / Canon / WhatsApp / No-EXIF), so each
    split gets roughly the same mix of cameras.
  - A fixed random seed makes the split exactly reproducible.

Reads (never modifies):
  data/standardised/dataset_XXXX.jpg
  data/yolo_labels/full_export_v1/dataset_XXXX.txt
  reports/rename_mapping.csv
Writes (copies):
  data/dataset_v1/images/{train,val,test}/   <- folder MUST be called "images" for YOLO
  data/dataset_v1/labels/{train,val,test}/   <- YOLO finds labels by swapping images->labels
  data/dataset_v1/data.yaml
  reports/split_assignment.csv
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import shutil
import sys

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
IMAGE_DIR = PROJECT_DIR / "data" / "standardised"
LABEL_DIR = PROJECT_DIR / "data" / "yolo_labels" / "full_export_v1"
MAPPING_CSV = PROJECT_DIR / "reports" / "rename_mapping.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "dataset_v1"
ASSIGNMENT_CSV = PROJECT_DIR / "reports" / "split_assignment.csv"

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 474
CLASSES = ["golf_ball"]


def count_boxes(label_path):
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*")):
        sys.exit(f"{OUTPUT_DIR} is not empty. Delete it first if you want to re-run.")

    with open(MAPPING_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "kept"]

    # Collect every image with its label, cluster and source
    items = []
    for r in rows:
        name = r["new_filename"]
        img, lbl = IMAGE_DIR / name, LABEL_DIR / f"{Path(name).stem}.txt"
        if not img.exists() or not lbl.exists():
            sys.exit(f"Missing image or label for {name}. Run the earlier steps first.")
        items.append({"name": name, "cluster": r["cluster_id"], "source": r["source"],
                      "boxes": count_boxes(lbl)})

    # Group by cluster (a cluster never mixes sources, but key on both to be safe)
    groups = defaultdict(list)
    for it in items:
        groups[(it["source"], it["cluster"])].append(it)

    # Stratified greedy assignment: per source, shuffle its groups, then give each
    # group to the split that is furthest below its target share for that source
    rng = random.Random(SEED)
    split_of = {}
    for source in sorted({k[0] for k in groups}):
        keys = sorted(k for k in groups if k[0] == source)
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        filled = Counter()
        for k in keys:
            # Background images (no balls) go to train: only there do they teach anything
            if all(it["boxes"] == 0 for it in groups[k]):
                split_of[k] = "train"
                filled["train"] += len(groups[k])
                continue
            deficit = {s: FRACTIONS[s] * total - filled[s] for s in FRACTIONS}
            target = max(deficit, key=deficit.get)
            split_of[k] = target
            filled[target] += len(groups[k])

    # Copy files into the YOLO folder structure
    for s in FRACTIONS:
        (OUTPUT_DIR / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / s).mkdir(parents=True, exist_ok=True)
    assignment = []
    for k, members in groups.items():
        s = split_of[k]
        for it in members:
            stem = Path(it["name"]).stem
            shutil.copy2(IMAGE_DIR / it["name"], OUTPUT_DIR / "images" / s / it["name"])
            shutil.copy2(LABEL_DIR / f"{stem}.txt", OUTPUT_DIR / "labels" / s / f"{stem}.txt")
            assignment.append({**it, "split": s})
    assignment.sort(key=lambda a: a["name"])

    with open(ASSIGNMENT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "split", "cluster", "source", "boxes"])
        w.writeheader()
        w.writerows(assignment)

    # data.yaml (the 'path' line is replaced in the Colab notebook)
    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASSES)}\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    # --- Integrity checks ---
    problems = []
    cluster_splits = defaultdict(set)
    for a in assignment:
        if a["cluster"]:
            cluster_splits[a["cluster"]].add(a["split"])
    leaks = [c for c, s in cluster_splits.items() if len(s) > 1]
    if leaks:
        problems.append(f"clusters in more than one split: {leaks[:5]}")
    for s in FRACTIONS:
        imgs = {p.stem for p in (OUTPUT_DIR / "images" / s).glob("*.jpg")}
        lbls = {p.stem for p in (OUTPUT_DIR / "labels" / s).glob("*.txt")}
        if imgs != lbls:
            problems.append(f"{s}: images and labels do not match")
    if len(assignment) != len(items):
        problems.append("not every image was assigned")

    # --- Summary ---
    n_img, n_box = len(assignment), sum(a["boxes"] for a in assignment)
    print("\n================= SPLIT SUMMARY =================")
    print(f"{'split':<7}{'images':>8}{'%img':>7}{'boxes':>8}{'%box':>7}{'neg':>5}   sources")
    for s in FRACTIONS:
        part = [a for a in assignment if a["split"] == s]
        b = sum(a["boxes"] for a in part)
        print(f"{s:<7}{len(part):>8}{100 * len(part) / n_img:>6.1f}%{b:>8}"
              f"{100 * b / n_box:>6.1f}%{sum(a['boxes'] == 0 for a in part):>5}   "
              f"{dict(Counter(a['source'] for a in part))}")
    print(f"{'total':<7}{n_img:>8}{'':>7}{n_box:>8}")
    print(f"\nRepeat clusters kept together: {sum(len(v) == 1 for v in cluster_splits.values())} "
          f"of {len(cluster_splits)} clusters")
    print(f"Integrity problems:            {len(problems)} {problems if problems else ''}")
    print(f"\nDataset:    {OUTPUT_DIR}\nAssignment: {ASSIGNMENT_CSV}")


if __name__ == "__main__":
    main()
