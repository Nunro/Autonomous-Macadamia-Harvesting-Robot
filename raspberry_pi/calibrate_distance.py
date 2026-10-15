#!/usr/bin/env python3
"""
calibrate_distance.py
MCTR 474: Painless in-situ distance calibration of the mounted Pi camera.
No chessboard, no angle gauge. Needs ONE tape measure on the floor and ONE golf ball.

Procedure (the script guides you):
  1. Lay a tape measure on the floor straight ahead of the robot, with 0 at the robot's front face.
  2. For each distance the script asks for, put the ball on the tape at that mark and press Enter.
     The script captures 15 frames with EXACTLY the camera settings of harvest_vision.py
     and records where the ball appears in the image.
  3. From all points it fits the two numbers that decide the distance calculation:
        DEPRESSION_DEG   (effective camera angle below horizontal)
        FOCAL_LENGTH_PX  (focal length in pixels at 640x480)
     with the camera height fixed at CAMERA_HEIGHT_MM.
  4. It reports the distance error at every point (old placeholders vs new values), the
     closest visible distance (basis for CAP_DIST) and the maximum reliable detection range.

STOP THE SERVICE FIRST (the camera can only be used by one program):
    sudo systemctl stop harvest_vision.service

Outputs (in ~/harvest/calibration/):
    calibration_results.json, calibration_points.csv, point_XXXcm.jpg (one image per point)
"""

import csv
import json
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# CONFIGURATION (camera settings identical to harvest_vision.py)
# ------------------------------------------------------------------------------
MODEL_PATH       = "/home/nunro/harvest/models/v2/best_v2_ncnn_model"
WIDTH, HEIGHT    = 640, 480
CAMERA_HEIGHT_MM = 220.0     # lens centre height, kept fixed in the fit
CAMERA_OFFSET_M  = 0.010     # lens is this far BEHIND the robot's front face
OLD_FOCAL_PX     = 259.1     # current placeholder in harvest_vision.py
OLD_DEPRESSION   = 25.0      # current design value in harvest_vision.py
ROBOT_CONF       = 0.50      # CONFIDENCE_THRESHOLD in harvest_vision.py

# Distances from the robot's FRONT FACE to the ball, in metres
DISTANCES_M      = [0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
FRAMES_PER_POINT = 15
OUT_DIR          = os.path.expanduser("~/harvest/calibration")


def predict_distance_m(cy, focal_px, depression_deg):
    """Same formula as harvest_vision.py: distance from the point below the lens."""
    angle = np.radians(depression_deg) + np.arctan2(cy - HEIGHT / 2.0, focal_px)
    return np.where(angle > 0, CAMERA_HEIGHT_MM / np.tan(np.maximum(angle, 1e-6)) / 1000.0, np.nan)


def fit(d_cam_m, cy):
    """Least-squares fit of depression angle and focal length in the angle domain.
    For a given focal length, the best depression angle is the mean of the per-point angles,
    so only the focal length needs a 1-D search (coarse, then fine)."""
    alpha = np.arctan2(CAMERA_HEIGHT_MM, np.asarray(d_cam_m) * 1000.0)   # true ray angle below horizontal
    v = np.asarray(cy) - HEIGHT / 2.0

    def score(f):
        theta_i = alpha - np.arctan2(v, f)
        return np.mean((theta_i - theta_i.mean()) ** 2), theta_i.mean()

    grid = np.arange(100.0, 1000.0, 1.0)
    f0 = grid[np.argmin([score(f)[0] for f in grid])]
    fine = np.arange(max(50.0, f0 - 2.0), f0 + 2.0, 0.01)
    f_best = fine[np.argmin([score(f)[0] for f in fine])]
    rms, theta = score(f_best)
    return float(f_best), float(np.degrees(theta)), float(np.degrees(np.sqrt(rms)))


def measure_point(picam2, model):
    """Capture FRAMES_PER_POINT frames, keep the strongest ball detection per frame."""
    cxs, cys, ws, confs, last = [], [], [], [], None
    for _ in range(FRAMES_PER_POINT):
        frame = picam2.capture_array()                      # RGB888 = BGR order (OpenCV order)
        r = model(frame, conf=0.10, verbose=False)[0]
        last = (frame, r)
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = r.boxes.xyxy[k].tolist()
            cxs.append((x1 + x2) / 2); cys.append((y1 + y2) / 2)
            ws.append(x2 - x1); confs.append(float(r.boxes.conf[k]))
        else:
            confs.append(0.0)
    hits = [c for c in confs if c >= ROBOT_CONF]
    return {
        "frames": FRAMES_PER_POINT,
        "frames_detected_at_robot_conf": len(hits),
        "median_conf": float(np.median(confs)),
        "cx": float(np.median(cxs)) if cxs else None,
        "cy": float(np.median(cys)) if cys else None,
        "width_px": float(np.median(ws)) if ws else None,
    }, last


def save_image(frame, result, path, text):
    img = frame.copy()                                      # already BGR, no conversion needed
    for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        colour = (0, 200, 0) if c >= ROBOT_CONF else (160, 160, 160)
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{c:.2f}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(path, img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading model...")
    model = YOLO(MODEL_PATH, task="detect")
    print("Starting camera (same settings as harvest_vision.py)...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}))
    picam2.set_controls({"AfMode": 0, "LensPosition": 1.25})
    picam2.start()
    time.sleep(1.5)

    print("\nLay the tape measure straight ahead with 0 at the robot's FRONT FACE.")
    print("Use ONE ball, centred on the tape. Keep other balls out of view.\n")
    points = []
    for d in DISTANCES_M:
        ans = input(f"Put the ball at {d:.2f} m and press Enter (or type s + Enter to skip): ")
        if ans.strip().lower() == "s":
            continue
        m, (frame, r) = measure_point(picam2, model)
        m["distance_front_m"] = d
        m["distance_cam_m"] = d + CAMERA_OFFSET_M
        points.append(m)
        save_image(frame, r, os.path.join(OUT_DIR, f"point_{int(round(d * 100)):03d}cm.jpg"),
                   f"{d:.2f} m  detected {m['frames_detected_at_robot_conf']}/{FRAMES_PER_POINT}")
        cy = f"{m['cy']:.1f}" if m["cy"] is not None else "-"
        print(f"   detected {m['frames_detected_at_robot_conf']:>2}/{FRAMES_PER_POINT} frames "
              f"at conf >= {ROBOT_CONF}, median conf {m['median_conf']:.2f}, cy {cy} px\n")
    picam2.stop()

    # ---- Fit on points that were detected reliably ----
    good = [p for p in points if p["frames_detected_at_robot_conf"] >= 0.8 * FRAMES_PER_POINT
            and p["cy"] is not None]
    if len(good) < 4:
        raise SystemExit(f"Only {len(good)} reliable points, at least 4 are needed. Re-run.")
    f_new, theta_new, rms_deg = fit([p["distance_cam_m"] for p in good], [p["cy"] for p in good])

    print("\n========================= CALIBRATION RESULT =========================")
    print(f"Reliable points used in the fit: {len(good)} of {len(points)}")
    print(f"FOCAL_LENGTH_PX = {f_new:.1f}   (placeholder was {OLD_FOCAL_PX})")
    print(f"DEPRESSION_DEG  = {theta_new:.2f}   (design value was {OLD_DEPRESSION})")
    print(f"Fit residual (RMS angle): {rms_deg:.3f} deg\n")

    print(f"{'tape':>6}{'cam':>7}{'cy':>7}{'conf':>6}{'det':>6}  {'OLD dist':>9}{'err':>7}"
          f"   {'NEW dist':>9}{'err':>7}")
    rows = []
    for p in points:
        if p["cy"] is None:
            print(f"{p['distance_front_m']:>6.2f}{p['distance_cam_m']:>7.2f}{'-':>7}{0:>6.2f}"
                  f"{0:>3}/{FRAMES_PER_POINT}  not detected")
            continue
        old = float(predict_distance_m(p["cy"], OLD_FOCAL_PX, OLD_DEPRESSION))
        new = float(predict_distance_m(p["cy"], f_new, theta_new))
        e_old = 100 * (old - p["distance_cam_m"]) / p["distance_cam_m"]
        e_new = 100 * (new - p["distance_cam_m"]) / p["distance_cam_m"]
        used = "" if p in good else "  (not used in fit)"
        print(f"{p['distance_front_m']:>6.2f}{p['distance_cam_m']:>7.2f}{p['cy']:>7.1f}"
              f"{p['median_conf']:>6.2f}{p['frames_detected_at_robot_conf']:>3}/{FRAMES_PER_POINT}"
              f"  {old:>8.3f}m{e_old:>+6.1f}%   {new:>8.3f}m{e_new:>+6.1f}%{used}")
        rows.append({**p, "pred_old_m": old, "pred_new_m": new,
                     "err_old_pct": e_old, "err_new_pct": e_new, "used_in_fit": p in good})

    # Closest visible ground distance: ball centre at the bottom edge of the frame
    near_cam = float(predict_distance_m(HEIGHT, f_new, theta_new))
    reliable = [p["distance_front_m"] for p in points
                if p["frames_detected_at_robot_conf"] >= 0.8 * FRAMES_PER_POINT]
    print(f"\nClosest visible distance (ball centre at bottom edge): {near_cam:.3f} m from the "
          f"lens ground point = {near_cam - CAMERA_OFFSET_M:.3f} m from the front face")
    print(f"Maximum reliable detection range (>= 80% of frames at conf >= {ROBOT_CONF}): "
          f"{max(reliable):.2f} m from the front face")

    with open(os.path.join(OUT_DIR, "calibration_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "calibration_results.json"), "w") as f:
        json.dump({"FOCAL_LENGTH_PX": round(f_new, 1), "DEPRESSION_DEG": round(theta_new, 2),
                   "CAMERA_HEIGHT_MM": CAMERA_HEIGHT_MM, "fit_rms_deg": round(rms_deg, 3),
                   "closest_visible_from_front_m": round(near_cam - CAMERA_OFFSET_M, 3),
                   "max_reliable_range_from_front_m": max(reliable),
                   "points": rows}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
