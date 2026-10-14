"""
evaluate_clean_test.py
MCTR 474 Final Year Project: Evaluate the SAME model (best_v2.pt) on
  1. the full test set (should reproduce the Colab numbers, a sanity check of this setup)
  2. the leakage-free test subset (contaminated images removed)
  3. the removed images on their own (shows how much the leakage inflated the scores)

The model is NOT retrained or changed. Only the contaminated part of the measurement is removed.

Reads:   models/v2/best_v2.pt
         data/dataset_v1/  (test images and labels, unchanged)
         reports/leakage_review/contaminated_test_images.txt  (your list)
Writes:  reports/clean_test/  (Ultralytics plots per subset, metrics_clean_test.json,
                               recall_by_ball_size_clean.png)

Install once:  pip install ultralytics==8.4.177
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import beta
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\MCTR474_Vision")
MODEL_PATH = PROJECT_DIR / "models" / "v2" / "best_v2.pt"
DATASET_DIR = PROJECT_DIR / "data" / "dataset_v1"
CONTAMINATED_TXT = PROJECT_DIR / "reports" / "leakage_review" / "contaminated_test_images.txt"
OUT_DIR = PROJECT_DIR / "reports" / "clean_test"

IMGSZ = 640
CONF_DEPLOY = 0.5
CLASS_NAME = "golf-ball"
BUCKETS = [(0, 10, "<10 px"), (10, 20, "10-20 px"), (20, 40, "20-40 px"), (40, 1e9, ">40 px")]


def lower_bound(k, n):
    """95 % two-sided Clopper-Pearson lower bound for a proportion k/n."""
    if n == 0:
        return None
    return round(float(beta.ppf(0.025, k, n - k + 1)), 4) if k > 0 else 0.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def write_yaml(name, image_paths):
    """Ultralytics accepts a .txt list of image paths as the test set."""
    list_file = OUT_DIR / f"{name}_images.txt"
    list_file.write_text("\n".join(p.as_posix() for p in image_paths) + "\n")
    yaml_file = OUT_DIR / f"{name}.yaml"
    yaml_file.write_text(
        f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: {list_file.as_posix()}\n\nnc: 1\nnames:\n  0: {CLASS_NAME}\n")
    return yaml_file


def deploy_metrics(model, image_paths):
    """Precision/recall at the robot's confidence threshold, recall per ball size."""
    found, total = [0] * len(BUCKETS), [0] * len(BUCKETS)
    tp = fp = 0
    for p in image_paths:
        with Image.open(p) as im:
            W, H = im.size
        gt = []
        for line in (DATASET_DIR / "labels" / "test" / f"{p.stem}.txt").read_text().splitlines():
            if line.strip():
                _, x, y, w, h = map(float, line.split())
                gt.append(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        r = model.predict(str(p), imgsz=IMGSZ, conf=CONF_DEPLOY, device="cpu", verbose=False)[0]
        matched = set()
        for b in r.boxes.xyxy.cpu().numpy().tolist():
            scores = [(iou(b, g), j) for j, g in enumerate(gt) if j not in matched]
            if scores and max(scores)[0] >= 0.5:
                matched.add(max(scores)[1])
                tp += 1
            else:
                fp += 1
        for j, g in enumerate(gt):
            width = (g[2] - g[0]) * IMGSZ / max(W, H)
            k = next(i for i, (lo, hi, _) in enumerate(BUCKETS) if lo <= width < hi)
            total[k] += 1
            found[k] += j in matched
    n_found, n_total = sum(found), sum(total)
    return {
        "images": len(image_paths), "balls": n_total, "true_positives": tp,
        "false_positives": fp, "missed": n_total - n_found,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(n_found / n_total, 4) if n_total else None,
        "recall_95_lower_bound": lower_bound(n_found, n_total),
        "by_size": [{"size": lab, "balls": t, "found": f,
                     "recall": round(f / t, 4) if t else None,
                     "recall_95_lower_bound": lower_bound(f, t)}
                    for (_, _, lab), f, t in zip(BUCKETS, found, total)],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_test = sorted((DATASET_DIR / "images" / "test").glob("*.jpg"))
    listed = {l.strip() for l in CONTAMINATED_TXT.read_text().splitlines()
              if l.strip() and not l.strip().startswith("#")}
    unknown = listed - {p.name for p in all_test}
    if unknown:
        raise SystemExit(f"These names are not test images, fix the list first: {sorted(unknown)}")
    if not listed:
        raise SystemExit("Your contaminated list is empty. Fill it in (or confirm none) first.")

    subsets = {
        "full_test": all_test,
        "clean_test": [p for p in all_test if p.name not in listed],
        "removed_test": [p for p in all_test if p.name in listed],
    }
    model = YOLO(str(MODEL_PATH))
    results = {"contaminated_images": sorted(listed)}

    for name, paths in subsets.items():
        print(f"\n--- Evaluating {name}: {len(paths)} images ---")
        m = model.val(data=str(write_yaml(name, paths)), split="test", imgsz=IMGSZ, batch=16,
                      device="cpu", project=str(OUT_DIR), name=name, exist_ok=True, plots=True,
                      verbose=False)
        results[name] = {
            "ultralytics": {"precision": round(float(m.box.mp), 4),
                            "recall": round(float(m.box.mr), 4),
                            "mAP50": round(float(m.box.map50), 4),
                            "mAP50_95": round(float(m.box.map), 4)},
            "at_conf_0.5": deploy_metrics(model, paths),
        }

    with open(OUT_DIR / "metrics_clean_test.json", "w") as f:
        json.dump(results, f, indent=2)

    # Recall per ball size on the CLEAN subset, with 95 % lower bounds as error bars
    by_size = results["clean_test"]["at_conf_0.5"]["by_size"]
    labels = [b["size"] for b in by_size]
    rec = [b["recall"] or 0 for b in by_size]
    low = [(b["recall"] or 0) - (b["recall_95_lower_bound"] or 0) for b in by_size]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rec, yerr=[low, [0] * len(low)], capsize=6, color="#2b7a3d")
    for b, s in zip(bars, by_size):
        ax.text(b.get_x() + b.get_width() / 2, 0.05, f"{s['found']}/{s['balls']}",
                ha="center", color="white")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Ball width at 640 px input")
    ax.set_ylabel(f"Recall (conf >= {CONF_DEPLOY})")
    ax.set_title("Leakage-free test set: recall by ball size (error bar = 95 % lower bound)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recall_by_ball_size_clean.png", dpi=200)

    # --- Summary ---
    print("\n======================= CLEAN TEST SUMMARY =======================")
    print(f"Contaminated test images removed: {len(listed)}")
    print(f"{'subset':<14}{'imgs':>5}{'balls':>6}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}"
          f"   | conf 0.5: {'P':>6}{'R':>7}{'R 95% low':>11}{'FP':>4}{'missed':>8}")
    for name in subsets:
        u, d = results[name]["ultralytics"], results[name]["at_conf_0.5"]
        print(f"{name:<14}{d['images']:>5}{d['balls']:>6}{u['precision']:>7.3f}{u['recall']:>7.3f}"
              f"{u['mAP50']:>8.3f}{u['mAP50_95']:>10.3f}   |           "
              f"{d['precision'] or 0:>6.3f}{d['recall'] or 0:>7.3f}"
              f"{d['recall_95_lower_bound'] or 0:>11.3f}{d['false_positives']:>4}{d['missed']:>8}")
    print("\nClean test, recall by ball size at conf 0.5:")
    for b in by_size:
        print(f"  {b['size']:<9} {b['found']:>3}/{b['balls']:<4} recall {b['recall']}  "
              f"(95% lower bound {b['recall_95_lower_bound']})")
    print(f"\nResults: {OUT_DIR}")


if __name__ == "__main__":
    main()
