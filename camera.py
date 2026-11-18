#!/usr/bin/env python3
"""
Hybrid Camera Interface for Laptop Webcam & Raspberry Pi Camera Module V2 (IMX219).
Automatically detects Picamera2 (libcamera on Raspberry Pi OS) or falls back to OpenCV VideoCapture.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class SmartCamera:
    """
    Unified camera wrapper that supports:
    1. Raspberry Pi Camera Module V2 via Picamera2 (libcamera stack on RPi OS Bullseye/Bookworm)
    2. Laptop Webcam / USB Camera via OpenCV V4L2
    3. Synthetic Demo Camera mode (for testing without hardware camera)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 20,
        camera_index: int = 0,
        force_demo: bool = False,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.force_demo = force_demo

        self.backend_name = "Uninitialized"
        self.picam2 = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._demo_tick = 0

        self._initialize()

    def _is_raspberry_pi(self) -> bool:
        try:
            with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as f:
                return "raspberry pi" in f.read().lower()
        except FileNotFoundError:
            return False

    def _initialize(self) -> None:
        if self.force_demo:
            self.backend_name = "Synthetic Demo Stream"
            return

        # 1. Try Picamera2 first if on Raspberry Pi (for Camera Module V2 CSI)
        if self._is_raspberry_pi():
            try:
                from picamera2 import Picamera2  # type: ignore

                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (self.width, self.height)}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.5)  # Warm up IMX219 sensor
                self.backend_name = "Picamera2 (RPi Camera V2 CSI)"
                print(f"[Camera] Started {self.backend_name} at {self.width}x{self.height}")
                return
            except Exception as exc:
                print(f"[Camera] Picamera2 not active ({exc}), trying V4L2/OpenCV...")
                self.picam2 = None

        # 2. Try OpenCV VideoCapture with explicit CAP_V4L2 first (avoids Ubuntu GStreamer v4l2src bug)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency & RAM usage
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                self.backend_name = f"OpenCV V4L2 (/dev/video{self.camera_index})"
                print(f"[Camera] Started {self.backend_name} at {self.width}x{self.height}")
                return
            cap.release()

        # 3. Fallback to Synthetic Demo Stream
        print("[Camera] No hardware camera accessible; falling back to Synthetic Demo Stream.")
        self.backend_name = "Synthetic Demo Stream"

    def read(self) -> Tuple[bool, np.ndarray]:
        """Reads a BGR frame of shape (height, width, 3)."""
        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    # Picamera2 RGB888 in recent versions outputs BGR directly for OpenCV,
                    # or 3-channel array; ensure standard dimensions
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    return True, frame
            except Exception:
                return False, self._generate_demo_frame()

        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame

        return True, self._generate_demo_frame()

    def _generate_demo_frame(self) -> np.ndarray:
        """Generates an animated synthetic test frame when no physical camera is attached."""
        self._demo_tick += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)

        # Moving synthetic target
        cx = int(self.width * 0.5 + 140 * np.sin(self._demo_tick * 0.05))
        cy = int(self.height * 0.5 + 60 * np.cos(self._demo_tick * 0.03))

        # Draw a stylized face/person silhouette for demo mode
        cv2.circle(frame, (cx, cy), 55, (180, 200, 230), -1)
        cv2.circle(frame, (cx - 18, cy - 12), 7, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 18, cy - 12), 7, (40, 40, 40), -1)
        cv2.ellipse(frame, (cx, cy + 15), (22, 12), 0, 0, 180, (40, 40, 120), 3)
        cv2.putText(
            frame,
            "DEMO STREAM (Connect Camera or omit --demo)",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
