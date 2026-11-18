#!/usr/bin/env python3
"""
RPi 4 Smart AI Vision Suite (4GB RAM / 5V 2A Eco-Power Edition)
Works seamlessly on:
  - Laptop Webcam (/dev/video0) for local testing
  - Raspberry Pi 4 + Camera Module V2 (IMX219 via Picamera2 / libcamera)

Controls in Live Window:
  [1] Mode 1: AI Face, Eyes & Smile Detector (ResNet-10 SSD DNN)
  [2] Mode 2: Smart Owner Face Recognition (Press [R] to register your face!)
  [3] Mode 3: AI 20-Class Object & Person Counter (MobileNet-SSD)
  [4] Mode 4: Smart Security Guard Zone & Intruder Auto-Snapshot
  [R] Register your face (in Mode 2)
  [S] Save a snapshot to snapshots/
  [E] Cycle Eco Power Frame-Skip (1 -> 2 -> 3 -> 4) for 5V 2A protection
  [Q] or [ESC] Quit
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from power_telemetry import apply_eco_power_limits, SystemMonitor, TelemetrySnapshot

# Enforce 2-core thread cap immediately for 5V 2A Raspberry Pi 4 safety
apply_eco_power_limits(max_threads=2)

import download_models  # noqa: E402
from camera import SmartCamera  # noqa: E402
from vision_engine import SmartVisionEngine, SNAPSHOTS_DIR, MODELS_DIR  # noqa: E402


def draw_hud(
    frame: np.ndarray,
    engine: SmartVisionEngine,
    telemetry: TelemetrySnapshot,
    camera_backend: str,
) -> None:
    h, w = frame.shape[:2]

    # Top translucent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (18, 18, 22), -1)
    cv2.rectangle(overlay, (0, h - 32), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Line 1: Active Mode Title
    mode_title = engine.MODE_NAMES.get(engine.mode, "AI Mode")
    cv2.putText(
        frame,
        mode_title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 240, 180),
        2,
        cv2.LINE_AA,
    )

    # Line 2: Telemetry (FPS, App RAM, CPU, Temp, Eco Skip)
    temp_str = f" | Temp: {telemetry.cpu_temp_c:.1f}C" if telemetry.cpu_temp_c > 0 else ""
    metrics_text = (
        f"FPS: {telemetry.fps:.1f} | App RAM: {telemetry.app_ram_mb:.0f}MB "
        f"| CPU: {telemetry.cpu_percent:.0f}%{temp_str} | EcoSkip: 1/{telemetry.eco_frame_skip}"
    )
    cv2.putText(
        frame,
        metrics_text,
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 225, 235),
        1,
        cv2.LINE_AA,
    )

    # Warning banner if Raspberry Pi 5V 2A under-voltage occurs
    if telemetry.under_voltage_now:
        cv2.rectangle(frame, (0, 64), (w, 92), (0, 0, 210), -1)
        cv2.putText(
            frame,
            "WARNING: 5V 2A Under-Voltage Detected! Auto-Eco Mode Engaged",
            (14, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Bottom Bar: Status + Keyboard Controls
    cv2.putText(
        frame,
        f"{engine.status_banner}   |   Keys: [1-4] Modes  [R] Reg Face  [S] Snap  [E] Eco  [Q] Quit",
        (12, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def ensure_models_ready() -> None:
    face_model = MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    obj_model = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
    if not face_model.exists() or not obj_model.exists():
        print("[Setup] Downloading lightweight AI models first...")
        download_models.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPi 4 Smart AI Vision Suite (4GB RAM / 5V 2A Eco Edition)"
    )
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3, 4], help="Initial AI mode (1-4)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default 640 for low RAM/Power)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default 480)")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS cap (default 20 for 5V 2A safety)")
    parser.add_argument("--frame-skip", type=int, default=2, help="Run DNN inference every Nth frame (default 2)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for V4L2/Laptop webcam")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo video stream")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (saves periodic frames)")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run indefinitely)")
    args = parser.parse_args()

    ensure_models_ready()

    monitor = SystemMonitor(base_frame_skip=max(1, args.frame_skip))
    cam = SmartCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_index=args.camera,
        force_demo=args.demo,
    )
    engine = SmartVisionEngine()
    engine.set_mode(args.mode)

    window_name = "RPi 4 Smart AI Vision Suite (Eco 5V/2A Mode)"
    frame_idx = 0
    target_frame_time = 1.0 / max(5, args.fps)

    print("\n============================================================")
    print("  RPi 4 Smart AI Vision Suite - Running!")
    print(f"  Camera Backend : {cam.backend_name}")
    print(f"  Active Mode    : {engine.MODE_NAMES[engine.mode]}")
    print("  Controls       : [1][2][3][4] Switch Mode | [R] Register Face")
    print("                   [S] Save Snapshot | [E] Eco Power | [Q] Quit")
    print("============================================================\n")

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_idx += 1
            telemetry = monitor.tick_frame()

            # Run AI neural network inference every Nth frame to keep CPU cool & <1.3A on 5V 2A adapter
            run_ai = (frame_idx % telemetry.eco_frame_skip == 1) or (telemetry.eco_frame_skip == 1)
            processed = engine.process_frame(frame, run_inference=run_ai)
            draw_hud(processed, engine, telemetry, cam.backend_name)

            if not args.headless:
                cv2.imshow(window_name, processed)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[Exit] User pressed Quit.")
                    break
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    new_m = int(chr(key))
                    engine.set_mode(new_m)
                    print(f"[Mode] -> {engine.MODE_NAMES[new_m]}")
                elif key in (ord("r"), ord("R")):
                    engine.set_mode(2)
                    engine.trigger_face_registration()
                    print("[FaceID] Requested owner face registration...")
                elif key in (ord("s"), ord("S")):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_p = SNAPSHOTS_DIR / f"manual_{ts}.jpg"
                    cv2.imwrite(str(out_p), processed)
                    engine.status_banner = f"Saved snapshot: {out_p.name}"
                    print(f"[Snapshot] Saved to {out_p}")
                elif key in (ord("e"), ord("E")):
                    monitor.base_frame_skip = (monitor.base_frame_skip % 4) + 1
                    monitor.snapshot.eco_frame_skip = monitor.base_frame_skip
                    engine.status_banner = f"Eco Frame-Skip set to 1/{monitor.base_frame_skip}"
                    print(f"[Power] Eco Frame-Skip -> 1/{monitor.base_frame_skip}")

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                out_preview = SNAPSHOTS_DIR / "last_test_preview.jpg"
                cv2.imwrite(str(out_preview), processed)
                print(
                    f"[Test Complete] Processed {frame_idx} frames | "
                    f"FPS: {telemetry.fps:.1f} | App RAM: {telemetry.app_ram_mb:.1f} MB | "
                    f"Saved preview -> {out_preview}"
                )
                break

            # Cap FPS to prevent unnecessary CPU wattage on 5V 2A adapter
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
    finally:
        cam.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
