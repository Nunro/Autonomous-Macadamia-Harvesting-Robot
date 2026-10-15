"""Verify a model on the Pi: class names, scores on dataset_0399, speed at 640x480."""
import sys
import time

import cv2
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/nunro/harvest/models/v2/best_v2_ncnn_model"
IMG = "/home/nunro/harvest/test_images/dataset_0399.jpg"

model = YOLO(MODEL, task="detect")
print("Model:      ", MODEL)
print("Class names:", model.names)

img = cv2.imread(IMG)
r = model(img, imgsz=640, conf=0.10, verbose=False)[0]
confs = sorted(r.boxes.conf.tolist(), reverse=True)
print(f"Detections (conf >= 0.10): {len(confs)}")
print("Scores:     ", " ".join(f"{c:.3f}" for c in confs))
print("Laptop .pt: ", "10 detections: 0.959 0.923 0.920 0.918 0.911 0.907 0.898 0.896 0.886 0.881")

# Speed on a REAL image at the pipeline's frame size (640x480), not a black dummy frame
frame = cv2.resize(img, (640, 480))
for _ in range(3):
    model(frame, verbose=False)          # warm-up
times = []
for _ in range(30):
    t = time.monotonic()
    model(frame, verbose=False)
    times.append(time.monotonic() - t)
avg = sum(times) / len(times)
print(f"Speed (640x480 real image, 30 runs): {avg * 1000:.1f} ms per frame, {1 / avg:.1f} FPS")
