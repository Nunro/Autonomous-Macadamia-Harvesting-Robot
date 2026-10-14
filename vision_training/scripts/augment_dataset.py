"""
augment_dataset.py
MCTR 474 Final Year Project, Step 8: Offline augmentation of the TRAINING split only.

Reads (never modifies):  data/dataset_v1/              (frozen split from Step 7)
Writes:                  data/dataset_v1_aug/          (train = originals + augmented copies,
                                                        val and test = unchanged copies)
                         reports/augmentation_log.csv  (one row per augmented image)
                         reports/augment_check/        (boxes drawn on a sample, for visual checking)

Augmentations (approved list):
  Pi camera simulation : downscale/upscale, Gaussian or motion blur, sensor noise, JPEG compression
  Lighting             : brightness/contrast, hue/saturation (small)
  Geometry (small)     : horizontal flip, rotation +-5 deg with the "ellipse" box method

Install once:  pip install albumentations==2.0.8
"""

from pathlib import Path
from collections import Counter
import csv
import random
import shutil
import sys

import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageDraw

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
SOURCE_DIR = PROJECT_DIR / "data" / "dataset_v1"
OUTPUT_DIR = PROJECT_DIR / "data" / "dataset_v1_aug"
LOG_CSV = PROJECT_DIR / "reports" / "augmentation_log.csv"
CHECK_DIR = PROJECT_DIR / "reports" / "augment_check"

COPIES_PER_IMAGE = 2
SEED = 474
SAVE_QUALITY = 95          # JPEG quality used to SAVE (the JPEG artefacts from the augmentation stay)
CHECK_SAMPLES = 30         # augmented images drawn with boxes for visual checking


def build_pipeline():
    return A.Compose(
        [
            # --- Pi camera simulation ---
            A.Downscale(scale_range=(0.25, 0.5),
                        interpolation_pair={"downscale": cv2.INTER_AREA,
                                            "upscale": cv2.INTER_LINEAR}, p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)),
                     A.MotionBlur(blur_limit=(3, 7))], p=0.4),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.3),
            # --- Lighting ---
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=0, p=0.4),
            # --- Geometry (small, the robot viewpoint is fixed) ---
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-5, 5), rotate_method="ellipse", border_mode=cv2.BORDER_CONSTANT,
                     fill=0, p=0.3),
            # --- JPEG compression last, like the Pi's own stream encoding ---
            A.ImageCompression(compression_type="jpeg", quality_range=(40, 80), p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                 min_visibility=0.3, clip=True),
        seed=SEED,
        save_applied_params=True,
    )


def read_labels(path):
    boxes, classes = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *b = line.split()
            classes.append(int(c))
            boxes.append([float(v) for v in b])
    return boxes, classes


def write_labels(path, boxes, classes):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                            for c, (x, y, w, h) in zip(classes, boxes)))


def imread_rgb(path):
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)       # albumentations expects RGB


def imwrite_rgb(path, rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY])
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def draw_check(img_path, label_path, out_path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    boxes, _ = read_labels(label_path)
    for x, y, w, h in boxes:
        d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                    outline=(255, 0, 255), width=max(2, round(max(W, H) / 400)))
    canvas = Image.new("RGB", (W, H + 30), "black")
    canvas.paste(im, (0, 30))
    ImageDraw.Draw(canvas).text((6, 8), f"{img_path.name}  boxes: {len(boxes)}", fill="white")
    canvas.save(out_path, quality=90)
    return canvas


def main():
    if not (SOURCE_DIR / "images" / "train").exists():
        sys.exit(f"{SOURCE_DIR} not found. Run split_dataset.py first.")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*")):
        sys.exit(f"{OUTPUT_DIR} is not empty. Delete it first if you want to re-run.")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy everything from dataset_v1 unchanged (train originals, val, test)
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            dst = OUTPUT_DIR / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            for f in (SOURCE_DIR / kind / split).iterdir():
                shutil.copy2(f, dst / f.name)

    # 2. Augment the training images
    pipeline = build_pipeline()
    train_imgs = sorted((SOURCE_DIR / "images" / "train").glob("*.jpg"))
    log, applied_counter = [], Counter()
    boxes_in = boxes_out = 0
    print(f"Augmenting {len(train_imgs)} training images x {COPIES_PER_IMAGE} copies...")

    for i, img_path in enumerate(train_imgs, start=1):
        image = imread_rgb(img_path)
        boxes, classes = read_labels(SOURCE_DIR / "labels" / "train" / f"{img_path.stem}.txt")
        for k in range(1, COPIES_PER_IMAGE + 1):
            out = pipeline(image=image, bboxes=boxes, class_labels=classes)
            name = f"{img_path.stem}_aug{k}"
            imwrite_rgb(OUTPUT_DIR / "images" / "train" / f"{name}.jpg", out["image"])
            write_labels(OUTPUT_DIR / "labels" / "train" / f"{name}.txt",
                         [list(b) for b in out["bboxes"]], [int(c) for c in out["class_labels"]])
            applied = [t["__class_fullname__"] if isinstance(t, dict) else str(t[0])
                       for t in out.get("applied_transforms", [])]
            applied = [a.split(".")[-1] for a in applied]
            applied_counter.update(set(applied))
            boxes_in += len(boxes)
            boxes_out += len(out["bboxes"])
            log.append({"augmented_image": f"{name}.jpg", "source_image": img_path.name,
                        "copy": k, "boxes_before": len(boxes), "boxes_after": len(out["bboxes"]),
                        "transforms": ";".join(applied)})
        if i % 50 == 0:
            print(f"  {i}/{len(train_imgs)}")

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)

    # 3. Visual check sample (fixed seed, so always the same sample)
    sample = random.Random(SEED).sample(log, min(CHECK_SAMPLES, len(log)))
    thumbs = []
    for row in sample:
        stem = Path(row["augmented_image"]).stem
        thumbs.append(draw_check(OUTPUT_DIR / "images" / "train" / f"{stem}.jpg",
                                 OUTPUT_DIR / "labels" / "train" / f"{stem}.txt",
                                 CHECK_DIR / f"{stem}_check.jpg"))
    cell, cols = 400, 5
    sheet = Image.new("RGB", (cols * cell, ((len(thumbs) + cols - 1) // cols) * cell), "white")
    for k, t in enumerate(thumbs):
        t.thumbnail((cell - 6, cell - 6))
        sheet.paste(t, ((k % cols) * cell + 3, (k // cols) * cell + 3))
    sheet.save(CHECK_DIR / "00_overview.jpg", quality=90)

    # 4. data.yaml (the 'path' line is replaced in the Colab notebook)
    shutil.copy2(SOURCE_DIR / "data.yaml", OUTPUT_DIR / "data.yaml")
    yaml = (OUTPUT_DIR / "data.yaml").read_text()
    (OUTPUT_DIR / "data.yaml").write_text(
        yaml.replace(SOURCE_DIR.as_posix(), OUTPUT_DIR.as_posix()))

    # --- Integrity checks and summary ---
    problems = []
    counts = {}
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (OUTPUT_DIR / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (OUTPUT_DIR / "labels" / split).glob("*.txt")}
        if imgs != lbls:
            problems.append(f"{split}: images and labels do not match")
        counts[split] = len(imgs)
    for split in ("val", "test"):
        if counts[split] != len(list((SOURCE_DIR / "images" / split).glob("*.jpg"))):
            problems.append(f"{split}: count changed")
    neg_aug = sum(1 for r in log if r["boxes_before"] == 0)
    lost_all = sum(1 for r in log if r["boxes_before"] > 0 and r["boxes_after"] == 0)

    print("\n=============== AUGMENTATION SUMMARY ===============")
    print(f"Train originals:            {len(train_imgs)}")
    print(f"Train augmented copies:     {len(log)}")
    print(f"Train total:                {counts['train']}")
    print(f"Val (unchanged):            {counts['val']}")
    print(f"Test (unchanged):           {counts['test']}")
    print(f"DATASET TOTAL:              {sum(counts.values())}")
    print(f"Boxes in augmented copies:  {boxes_out} of {boxes_in} kept "
          f"({boxes_in - boxes_out} dropped as mostly outside the image)")
    print(f"Negative copies:            {neg_aug}")
    print(f"Copies that lost ALL boxes: {lost_all}")
    print("How often each augmentation was applied (out of "
          f"{len(log)} copies):")
    for name, n in applied_counter.most_common():
        print(f"  {name:<26}{n:>5}  ({100 * n / len(log):.0f}%)")
    print(f"Integrity problems:         {len(problems)} {problems if problems else ''}")
    print(f"\nDataset: {OUTPUT_DIR}\nLog:     {LOG_CSV}\nChecks:  {CHECK_DIR}")


if __name__ == "__main__":
    main()
