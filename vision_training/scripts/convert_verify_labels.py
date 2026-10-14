"""
convert_verify_labels.py
MCTR 474 Final Year Project, Step 6: Convert a Label Studio JSON export to YOLO
format and verify the boxes visually.

Reads:   data/labelstudio_exports/<EXPORT_FILE>        (full JSON export, not JSON-MIN)
         data/standardised/dataset_XXXX.jpg            (only read)
Writes:  data/yolo_labels/<export name>/dataset_XXXX.txt   (one YOLO label file per labelled image)
         data/yolo_labels/<export name>/classes.txt
         reports/label_check/<export name>/*_check.jpg     (boxes drawn back onto each image)
         reports/label_check/<export name>/00_overview.jpg (all checks on one sheet)

The overlays are drawn from the YOLO .txt files, NOT from the JSON. If a box lands
exactly on its ball in the overlay, the whole chain (Label Studio -> JSON -> YOLO) is correct.

YOLO format, one line per box:  class x_center y_center width height  (all 0 to 1)
An image submitted with zero boxes gets an EMPTY .txt file (a negative/background image).
"""

from pathlib import Path
from collections import Counter
import json
import math
import statistics
import sys

from PIL import Image, ImageDraw

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
EXPORT_FILE = "full_export_v1.json"
CLASSES = ["golf_ball"]                 # class id 0
TRAIN_IMGSZ = 640                       # YOLO training size, used to predict box size in training

EXPORT_PATH = PROJECT_DIR / "data" / "labelstudio_exports" / EXPORT_FILE
IMAGE_DIR = PROJECT_DIR / "data" / "standardised"
LABEL_DIR = PROJECT_DIR / "data" / "yolo_labels" / Path(EXPORT_FILE).stem
CHECK_DIR = PROJECT_DIR / "reports" / "label_check" / Path(EXPORT_FILE).stem


def original_name(task):
    """Remove Label Studio's random upload prefix: 'a1b2c3d4-dataset_0001.jpg' -> 'dataset_0001.jpg'."""
    name = task.get("file_upload") or Path(task["data"]["image"]).name
    return name.split("-", 1)[1] if "-" in name else name


def draw_overlay(img_path, label_path, out_path):
    """Draw the boxes from the YOLO .txt file back onto the image."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    thickness = max(2, round(max(W, H) / 400))
    lines = label_path.read_text().split("\n") if label_path.stat().st_size else []
    for k, line in enumerate(l for l in lines if l.strip()):
        _, xc, yc, w, h = map(float, line.split())
        x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
        x1, y1 = (xc + w / 2) * W, (yc + h / 2) * H
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=thickness)
        draw.text((x0 + 3, max(0, y0 - 14)), str(k + 1), fill=(255, 0, 255))
    # Filename banner on an extra strip ABOVE the image, so it never hides a box
    text = f"{img_path.name}  boxes: {len([l for l in lines if l.strip()])}"
    canvas = Image.new("RGB", (W, H + 30), "black")
    canvas.paste(img, (0, 30))
    ImageDraw.Draw(canvas).text((6, 8), text, fill="white")
    canvas.save(out_path, quality=90)
    return canvas


def overview_sheet(images, out_path, cell=400, columns=5):
    if not images:
        return
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
    for k, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((cell - 6, cell - 6))
        sheet.paste(thumb, ((k % columns) * cell + 3, (k // columns) * cell + 3))
    sheet.save(out_path, quality=90)


def main():
    if not EXPORT_PATH.exists():
        sys.exit(f"Export not found:\n{EXPORT_PATH}")
    with open(EXPORT_PATH, encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or (tasks and "annotations" not in tasks[0]):
        sys.exit("This does not look like a full JSON export. Export as 'JSON', not 'JSON-MIN'.")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    (LABEL_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    warnings = []
    seen = Counter()
    labelled, negatives, unlabelled = [], [], 0
    box_w_px, box_h_px, box_w_train, boxes_per_image = [], [], [], []

    for task in tasks:
        name = original_name(task)
        seen[name] += 1
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            unlabelled += 1
            continue
        if len(annotations) > 1:
            warnings.append(f"{name}: {len(annotations)} annotations, using the latest")
        annotation = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")

        img_path = IMAGE_DIR / name
        if not img_path.exists():
            warnings.append(f"{name}: image not found in {IMAGE_DIR}")
            continue
        with Image.open(img_path) as im:
            W, H = im.size

        lines = []
        for r in annotation.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            label = v["rectanglelabels"][0]
            if label not in CLASSES:
                warnings.append(f"{name}: unknown label '{label}', box skipped")
                continue
            if (r.get("original_width"), r.get("original_height")) != (W, H):
                warnings.append(f"{name}: Label Studio saw {r.get('original_width')}x"
                                f"{r.get('original_height')}, file is {W}x{H}")
            if abs(v.get("rotation", 0)) > 0.01:
                warnings.append(f"{name}: box rotated {v['rotation']:.1f} deg, "
                                f"redraw it without rotation")

            # Label Studio: top-left corner and size in PERCENT of the image
            x0, y0 = v["x"] / 100, v["y"] / 100
            x1, y1 = x0 + v["width"] / 100, y0 + v["height"] / 100
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(1, x1), min(1, y1)
            if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                warnings.append(f"{name}: box extended past the image edge, clipped")
            w, h = cx1 - cx0, cy1 - cy0
            if w * W < 2 or h * H < 2:
                warnings.append(f"{name}: box smaller than 2 px ({w * W:.1f}x{h * H:.1f}), "
                                f"probably an accidental click, skipped")
                continue
            lines.append(f"{CLASSES.index(label)} {cx0 + w / 2:.6f} {cy0 + h / 2:.6f} "
                         f"{w:.6f} {h:.6f}")
            box_w_px.append(w * W)
            box_h_px.append(h * H)
            box_w_train.append(w * W * TRAIN_IMGSZ / max(W, H))

        label_path = LABEL_DIR / f"{Path(name).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        boxes_per_image.append(len(lines))
        (labelled if lines else negatives).append(name)

    for name, count in seen.items():
        if count > 1:
            warnings.append(f"{name}: appears in {count} tasks (imported twice?)")

    # --- Overlays drawn FROM the YOLO files ---
    overlays = []
    for name in sorted(labelled + negatives):
        stem = Path(name).stem
        overlays.append(draw_overlay(IMAGE_DIR / name, LABEL_DIR / f"{stem}.txt",
                                     CHECK_DIR / f"{stem}_check.jpg"))
    overview_sheet(overlays, CHECK_DIR / "00_overview.jpg")

    # --- Summary ---
    print("\n============ CONVERT AND VERIFY SUMMARY ============")
    print(f"Tasks in export:            {len(tasks)}")
    print(f"Labelled images (balls):    {len(labelled)}")
    print(f"Negative images (0 boxes):  {len(negatives)}  {sorted(negatives)}")
    print(f"Not yet labelled (skipped): {unlabelled}")
    if box_w_px:
        print(f"Total boxes:                {len(box_w_px)}")
        print(f"Boxes per image:            min {min(boxes_per_image)}, "
              f"mean {statistics.mean(boxes_per_image):.1f}, max {max(boxes_per_image)}")
        print(f"Box width (px, 1920 image): min {min(box_w_px):.0f}, "
              f"median {statistics.median(box_w_px):.0f}, max {max(box_w_px):.0f}")
        print(f"Box width at imgsz {TRAIN_IMGSZ}:    min {min(box_w_train):.1f}, "
              f"median {statistics.median(box_w_train):.1f}, max {max(box_w_train):.1f}")
        print(f"Boxes under 10 px at {TRAIN_IMGSZ}:  {sum(b < 10 for b in box_w_train)}")
    print(f"Warnings:                   {len(warnings)}")
    for w in warnings:
        print(f"  - {w}")
    print(f"\nYOLO labels: {LABEL_DIR}\nOverlays:    {CHECK_DIR}")


if __name__ == "__main__":
    main()
