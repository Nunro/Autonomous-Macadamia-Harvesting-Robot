#!/usr/bin/env python3
"""
MCTR 474 - Macadamia Harvesting Robot - Vision Pipeline
Raspberry Pi 5, Camera Module 3 Wide
Nunro Brenkman 37540505 - NWU Potchefstroom

Includes integrated MJPEG streaming server on port 8080.
View live feed on any device on the same network:
  http://192.168.0.29:8080  (home WiFi)
  http://10.42.0.1:8080     (Pi hotspot in field)
"""

import time
import struct
import serial
import threading
import io
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================

UART_PORT            = "/dev/ttyAMA0"
UART_BAUD            = 115200
MODEL_PATH           = "/home/nunro/harvest/models/v2/best_v2_ncnn_model"   # v2 (v1 backup: models/v1_roboflow/best_ncnn_model)
INFERENCE_WIDTH      = 640
INFERENCE_HEIGHT     = 480
CAMERA_HEIGHT_MM     = 220.0
DEPRESSION_DEG       = 25.0
CAMERA_OFFSET_MM     = 10.0
FOCAL_LENGTH_PX      = 259.1   # placeholder, replace with calibrated value
PACKET_RATE_HZ       = 10
CONFIDENCE_THRESHOLD = 0.5
SPORTS_BALL_CLASS_ID = 0
STREAM_PORT          = 8080    # MJPEG stream port, view at http://<pi-ip>:8080
STATUS_BAR_HEIGHT    = 26      # status bar added BELOW the camera image on the stream
DISPLAY_MIN_CONF     = 0.25    # weakest detection drawn on the stream (grey box), display only

# ==============================================================================
# SHARED FRAME BUFFER (main thread writes, stream thread reads)
# ==============================================================================

latest_frame_jpg = None
frame_lock = threading.Lock()


# ==============================================================================
# MJPEG STREAMING SERVER
# ==============================================================================

class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request log noise in journal

    def do_GET(self):
        if self.path == '/':
            # Redirect to /stream for convenience
            self.send_response(302)
            self.send_header('Location', '/stream')
            self.end_headers()

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpg = latest_frame_jpg
                    if jpg is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.1)  # 10fps stream, independent of inference rate
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected, exit gracefully

        else:
            self.send_response(404)
            self.end_headers()


def stream_server_thread():
    server = HTTPServer(('0.0.0.0', STREAM_PORT), MjpegHandler)
    print(f"MJPEG stream started on port {STREAM_PORT}")
    print(f"  View at http://192.168.0.29:{STREAM_PORT}/stream  (home WiFi)")
    print(f"  View at http://10.42.0.1:{STREAM_PORT}/stream     (Pi hotspot)")
    server.serve_forever()


# ==============================================================================
# CAMERA GEOMETRY
# ==============================================================================

def pixel_to_bearing_mrad(pixel_x, frame_width, focal_length_px):
    pixel_offset = pixel_x - (frame_width / 2.0)
    bearing_rad = np.arctan2(pixel_offset, focal_length_px)
    bearing_mrad = int(bearing_rad * 1000.0)
    return max(-32768, min(32767, bearing_mrad))


def pixel_to_ground_distance(pixel_y, frame_height, focal_length_px,
                              camera_height_mm, depression_deg):
    pixel_offset_from_centre = pixel_y - (frame_height / 2.0)
    angle_from_axis_rad = np.arctan2(pixel_offset_from_centre, focal_length_px)
    total_depression_rad = np.radians(depression_deg) + angle_from_axis_rad
    if total_depression_rad <= 0.0:
        return 0.0
    return (camera_height_mm / np.tan(total_depression_rad)) / 1000.0


# ==============================================================================
# PACKET BUILDING
# ==============================================================================

def build_packet(bearing_mrad, ground_distance_m, mode_flag):
    payload = struct.pack(">hfB", int(bearing_mrad),
                          float(ground_distance_m), int(mode_flag))
    checksum = 0
    for byte in payload:
        checksum ^= byte
    return payload + struct.pack("B", checksum)


# ==============================================================================
# STREAM STATUS BAR
# ==============================================================================

def add_status_bar(display, state, best_conf, ground_dist_m, bearing_mrad,
                   n_balls, loop_hz):
    """Append a coloured status bar BELOW the image (it never hides any of the
    camera view). Colours are BGR because the frame is in OpenCV (BGR) order."""
    h, w = display.shape[:2]
    if state == "TRACKING":
        colour = (0, 110, 0)          # dark green
        text = (f"TRACKING   conf {best_conf:.2f}   dist {ground_dist_m:.2f} m   "
                f"bearing {bearing_mrad:+d} mrad   balls {n_balls}   {loop_hz:.1f} Hz")
    elif state == "OUT OF RANGE":
        colour = (0, 120, 200)        # orange
        text = (f"BALL OUT OF RANGE   conf {best_conf:.2f}   dist {ground_dist_m:.2f} m   "
                f"balls {n_balls}   {loop_hz:.1f} Hz")
    else:
        colour = (0, 0, 140)          # dark red
        text = (f"SEARCHING   no ball with conf >= {CONFIDENCE_THRESHOLD:.2f}   "
                f"{loop_hz:.1f} Hz")
    bar = np.full((STATUS_BAR_HEIGHT, w, 3), colour, dtype=np.uint8)
    bar[0:2, :] = 255                 # white separator line between image and bar
    cv2.putText(bar, text, (8, STATUS_BAR_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([display, bar])


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    global latest_frame_jpg

    print("MCTR 474 Vision Pipeline starting...")

    # Start MJPEG stream server in background thread (daemon so it dies with main)
    t = threading.Thread(target=stream_server_thread, daemon=True)
    t.start()

    # UART
    print(f"Opening UART on {UART_PORT} at {UART_BAUD} baud...")
    try:
        uart = serial.Serial(UART_PORT, UART_BAUD, timeout=1)
        print("UART open.")
    except serial.SerialException as e:
        print(f"ERROR: Could not open UART: {e}")
        raise

    # Model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        raise

    # Camera
    print("Initialising camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (INFERENCE_WIDTH, INFERENCE_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.set_controls({"AfMode": 0, "LensPosition": 1.25})
    picam2.start()
    print("Camera started with fixed focus at 0.8m.")
    time.sleep(1.0)

    print(f"Entering main inference loop at {PACKET_RATE_HZ}Hz.")
    period = 1.0 / PACKET_RATE_HZ
    loop_count = 0
    loop_hz = 0.0                     # measured loop rate, shown on the status bar
    last_loop_start = None

    while True:
        loop_start = time.monotonic()
        if last_loop_start is not None:
            instant_hz = 1.0 / max(loop_start - last_loop_start, 1e-6)
            loop_hz = instant_hz if loop_hz == 0.0 else 0.9 * loop_hz + 0.1 * instant_hz
        last_loop_start = loop_start

        # Capture frame
        frame = picam2.capture_array()

        # Run inference
        # conf=DISPLAY_MIN_CONF is the Ultralytics default (0.25), stated explicitly here.
        # Only detections >= CONFIDENCE_THRESHOLD are used for steering, see below.
        results = model(frame, conf=DISPLAY_MIN_CONF, verbose=False)

        bearing_mrad  = 0
        ground_dist_m = 0.0
        mode_flag     = 0
        best_conf     = 0.0
        best_box      = None
        n_balls       = 0             # detections at or above the threshold
        all_boxes     = []            # every golf ball detection, for the stream overlay only

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != SPORTS_BALL_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                all_boxes.append(box)
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                n_balls += 1
                if conf > best_conf:
                    best_conf = conf
                    best_box = box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bearing_mrad = pixel_to_bearing_mrad(
                        cx, INFERENCE_WIDTH, FOCAL_LENGTH_PX)
                    ground_dist_m = pixel_to_ground_distance(
                        cy, INFERENCE_HEIGHT, FOCAL_LENGTH_PX,
                        CAMERA_HEIGHT_MM, DEPRESSION_DEG)
                    if 0.05 < ground_dist_m < 3.0:
                        mode_flag = 1

        # Build MJPEG frame with detection overlay for the stream
        # Frame is already BGR (Picamera2 RGB888), draw bounding box if detected
        display = frame.copy()  # Picamera2 RGB888 arrays are already BGR (OpenCV order)

        # All other detections first (thin boxes), the target is drawn on top afterwards:
        #   yellow = other balls >= CONFIDENCE_THRESHOLD (seen, not chosen as target)
        #   grey   = near misses between DISPLAY_MIN_CONF and CONFIDENCE_THRESHOLD (ignored)
        for box in all_boxes:
            if box is best_box:
                continue
            conf = float(box.conf[0])
            colour = (0, 255, 255) if conf >= CONFIDENCE_THRESHOLD else (160, 160, 160)
            bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(display, (bx1, by1), (bx2, by2), colour, 1)
            cv2.putText(display, f"{conf:.2f}", (bx1, max(by1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
        if best_box is not None and mode_flag == 1:
            x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0].tolist()]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display,
                        f"{best_conf:.2f} | {ground_dist_m:.2f}m | {bearing_mrad:+d}mrad",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        elif best_box is not None:
            # Ball detected but its distance is outside 0.05-3.0 m (e.g. near the horizon)
            x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0].tolist()]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

        if mode_flag == 1:
            state = "TRACKING"
        elif best_box is not None:
            state = "OUT OF RANGE"
        else:
            state = "SEARCHING"
        display = add_status_bar(display, state, best_conf, ground_dist_m,
                                 bearing_mrad, n_balls, loop_hz)

        # Encode to JPEG and push to shared buffer
        _, jpg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame_jpg = jpg.tobytes()

        # Send UART packet
        packet = build_packet(bearing_mrad, ground_dist_m, mode_flag)
        uart.write(packet)

        # Print status every 10 frames
        if loop_count % 10 == 0:
            if mode_flag == 1:
                print(f"[{loop_count:06d}] TRACKING  - "
                      f"bearing={bearing_mrad:+5d}mrad, "
                      f"dist={ground_dist_m:.3f}m, conf={best_conf:.2f}")
            else:
                print(f"[{loop_count:06d}] SEARCHING - no ball detected")

        loop_count += 1

        elapsed = time.monotonic() - loop_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
