#!/usr/bin/env python3
"""
Multi-Mode Lightweight AI & Computer Vision Engine for Raspberry Pi 4 (4GB RAM / 5V 2A Power Budget).

Modes:
  [1] AI Face, Eyes, Smile & Distance/Attention Detector (ResNet-10 SSD DNN + Haar Sub-ROI)
  [2] Smart Owner Face Recognition ("Press 'R' to Register Your Face" -> Authorized vs Unknown)
  [3] AI Everyday Object & Person Detector (MobileNet-SSD 20 Classes)
  [4] Smart Security Zone Guard (Motion + AI Person/Face Intrusion Alert with Auto-Snapshot)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    color: Tuple[int, int, int] = (0, 255, 128)
    sub_boxes: List[Tuple[int, int, int, int, str]] = field(default_factory=list)


class SmartVisionEngine:
    MODE_NAMES = {
        1: "1: AI Face + Eyes + Smile Detector (DNN)",
        2: "2: Smart Face Recognition (Press 'R' to Register)",
        3: "3: AI Object & Person Counter (MobileNet-SSD)",
        4: "4: Smart Security Zone Guard & Auto-Snapshot",
    }

    def __init__(self, conf_threshold: float = 0.55):
        self.conf_threshold = conf_threshold
        self.mode = 1
        self.cached_detections: List[DetectionBox] = []
        self.status_banner: str = "Ready"
        self.alert_active: bool = False

        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

        # Models
        self.face_net = None
        self.obj_net = None
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None

        # Face Recognizer (Mode 2)
        self.recognizer = None
        self.owner_registered = False
        self.owner_name = "Ahmed (Owner)"
        self._request_register = False

        # Security Guard (Mode 4)
        self.prev_gray: Optional[np.ndarray] = None
        self.last_snapshot_time = 0.0
        self.total_alerts = 0

        self._load_models()
        self._init_face_recognizer()

    def _load_models(self) -> None:
        # 1. Load Deep Learning Face Detector (ResNet-10 SSD FP16)
        proto = MODELS_DIR / "face_deploy.prototxt"
        model = MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        if proto.exists() and model.exists():
            try:
                self.face_net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
                self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[Engine] Loaded ResNet-10 SSD Face DNN (FP16)")
            except Exception as exc:
                print(f"[Engine] Could not load Face DNN: {exc}")

        # 2. Load MobileNet-SSD Object Detector
        obj_proto = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
        obj_model = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
        if obj_proto.exists() and obj_model.exists():
            try:
                self.obj_net = cv2.dnn.readNetFromCaffe(str(obj_proto), str(obj_model))
                self.obj_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.obj_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[Engine] Loaded MobileNet-SSD 20-Class Object DNN")
            except Exception as exc:
                print(f"[Engine] Could not load MobileNet-SSD: {exc}")

        # 3. Load Haar Cascades for fallback & ultra-fast eye/smile ROI analysis
        def _load_haar(fname: str) -> Optional[cv2.CascadeClassifier]:
            candidates = [
                MODELS_DIR / fname,
                Path("/usr/share/opencv4/haarcascades") / fname,
                Path("/usr/local/share/opencv4/haarcascades") / fname,
            ]
            for p in candidates:
                if p.exists():
                    cc = cv2.CascadeClassifier(str(p))
                    if not cc.empty():
                        return cc
            return None

        self.face_cascade = _load_haar("haarcascade_frontalface_default.xml")
        self.eye_cascade = _load_haar("haarcascade_eye.xml")
        self.smile_cascade = _load_haar("haarcascade_smile.xml")

    def _init_face_recognizer(self) -> None:
        if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            model_file = KNOWN_FACES_DIR / "owner_lbph.yml"
            if model_file.exists():
                try:
                    self.recognizer.read(str(model_file))
                    self.owner_registered = True
                    print("[Engine] Loaded registered owner face profile from known_faces/owner_lbph.yml")
                except Exception:
                    pass

    def set_mode(self, new_mode: int) -> None:
        if new_mode in self.MODE_NAMES:
            self.mode = new_mode
            self.cached_detections.clear()
            self.alert_active = False
            self.status_banner = f"Switched to {self.MODE_NAMES[new_mode]}"

    def trigger_face_registration(self) -> None:
        self._request_register = True

    def detect_faces_raw(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Returns list of (x1, y1, x2, y2, confidence) using DNN or Haar fallback."""
        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int, float]] = []

        if self.face_net is not None:
            # 300x300 blob for ResNet-10 SSD
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf >= self.conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 - x1 > 20 and y2 - y1 > 20:
                        boxes.append((x1, y1, x2, y2, conf))
            return boxes

        # Fallback to Haar Cascade if DNN not present
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, fw, fh) in rects:
                boxes.append((int(x), int(y), int(x + fw), int(y + fh), 0.85))
        return boxes

    def process_frame(self, frame: np.ndarray, run_inference: bool = True) -> np.ndarray:
        """
        Runs AI inference on active frames and draws lightweight HUD overlays on every frame.
        Frame-skipping (`run_inference=False`) reuses cached bounding boxes to save ~50% CPU & Power!
        """
        output = frame.copy()

        if run_inference:
            if self.mode == 1:
                self._run_mode1_face_analysis(frame)
            elif self.mode == 2:
                self._run_mode2_face_recognition(frame)
            elif self.mode == 3:
                self._run_mode3_object_detection(frame)
            elif self.mode == 4:
                self._run_mode4_security_guard(frame)

        # Render detections & mode-specific overlays
        self._draw_overlays(output)
        return output

    def _run_mode1_face_analysis(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = self.detect_faces_raw(frame)
        results: List[DetectionBox] = []

        for (x1, y1, x2, y2, conf) in raw_faces:
            roi_gray = gray[y1:y2, x1:x2]
            sub_boxes: List[Tuple[int, int, int, int, str]] = []
            eyes_count = 0
            smiling = False

            if roi_gray.size > 0:
                if self.eye_cascade is not None:
                    # Search upper 60% of face for eyes
                    upper_face = roi_gray[0 : int((y2 - y1) * 0.65), :]
                    eyes = self.eye_cascade.detectMultiScale(
                        upper_face, scaleFactor=1.15, minNeighbors=4, minSize=(14, 14)
                    )
                    eyes_count = len(eyes)
                    for (ex, ey, ew, eh) in eyes[:2]:
                        sub_boxes.append((x1 + int(ex), y1 + int(ey), int(ew), int(eh), "Eye"))

                if self.smile_cascade is not None:
                    # Search lower 50% of face for smile
                    lower_y = int((y2 - y1) * 0.5)
                    lower_face = roi_gray[lower_y:, :]
                    smiles = self.smile_cascade.detectMultiScale(
                        lower_face, scaleFactor=1.7, minNeighbors=18, minSize=(22, 12)
                    )
                    if len(smiles) > 0:
                        smiling = True
                        sx, sy, sw, sh = smiles[0]
                        sub_boxes.append(
                            (x1 + int(sx), y1 + lower_y + int(sy), int(sw), int(sh), "Smile")
                        )

            mood = "Smiling" if smiling else ("Focused" if eyes_count >= 1 else "Face")
            label = f"{mood} ({conf * 100:.0f}%)"
            color = (0, 255, 140) if smiling else (255, 200, 0)
            results.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                    confidence=conf,
                    color=color,
                    sub_boxes=sub_boxes,
                )
            )

        self.cached_detections = results
        self.status_banner = f"Faces Detected: {len(results)}"

    def _run_mode2_face_recognition(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = self.detect_faces_raw(frame)
        results: List[DetectionBox] = []

        for (x1, y1, x2, y2, conf) in raw_faces:
            face_crop = gray[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, (120, 120))
            face_eq = cv2.equalizeHist(face_resized)

            # Register face if user pressed 'R'
            if self._request_register and self.recognizer is not None:
                samples = [face_eq]
                # Create slight augmentation samples for robust instant registration
                samples.append(cv2.flip(face_eq, 1))
                M1 = cv2.getRotationMatrix2D((60, 60), 5, 1.0)
                M2 = cv2.getRotationMatrix2D((60, 60), -5, 1.0)
                samples.append(cv2.warpAffine(face_eq, M1, (120, 120)))
                samples.append(cv2.warpAffine(face_eq, M2, (120, 120)))
                labels = np.array([1, 1, 1, 1], dtype=np.int32)
                self.recognizer.train(samples, labels)
                self.recognizer.write(str(KNOWN_FACES_DIR / "owner_lbph.yml"))
                cv2.imwrite(str(KNOWN_FACES_DIR / "owner_reference.jpg"), frame[y1:y2, x1:x2])
                self.owner_registered = True
                self._request_register = False
                self.status_banner = "Registered Owner Face Successfully!"

            if self.owner_registered and self.recognizer is not None:
                label_id, dist = self.recognizer.predict(face_eq)
                # In LBPH, lower distance = better match (< 78 is strong match)
                if label_id == 1 and dist < 78.0:
                    match_score = max(50.0, min(99.0, 100.0 - dist * 0.6))
                    label = f"{self.owner_name} ({match_score:.0f}%)"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown Person (dist:{dist:.0f})"
                    color = (0, 70, 255)
            else:
                label = "Unregistered (Press 'R' to save face)"
                color = (0, 215, 255)

            results.append(
                DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=conf, color=color)
            )

        if self._request_register and not raw_faces:
            self.status_banner = "Look at the camera and press 'R' again to register!"
            self._request_register = False
        elif not self.owner_registered:
            self.status_banner = "Press 'R' while looking at camera to register your face"
        else:
            self.status_banner = f"Face ID Active | Faces: {len(results)}"

        self.cached_detections = results

    def _run_mode3_object_detection(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        results: List[DetectionBox] = []
        counts: Dict[str, int] = {}

        if self.obj_net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self.obj_net.setInput(blob)
            detections = self.obj_net.forward()

            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf >= 0.45:
                    idx = int(detections[0, 0, i, 1])
                    if 0 <= idx < len(VOC_CLASSES):
                        cls_name = VOC_CLASSES[idx]
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype("int")
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                        color = (0, 255, 100) if cls_name == "person" else (255, 180, 0)
                        results.append(
                            DetectionBox(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                label=f"{cls_name.upper()} {conf * 100:.0f}%",
                                confidence=conf,
                                color=color,
                            )
                        )
        else:
            # Fallback to face detection if MobileNet-SSD not present
            for (x1, y1, x2, y2, conf) in self.detect_faces_raw(frame):
                counts["face"] = counts.get("face", 0) + 1
                results.append(
                    DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, label="FACE", confidence=conf)
                )

        summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "Scanning room..."
        self.status_banner = f"Detected: {summary}"
        self.cached_detections = results

    def _run_mode4_security_guard(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        # Restricted Guard Zone (central 60% of frame)
        zx1, zy1 = int(w * 0.2), int(h * 0.18)
        zx2, zy2 = int(w * 0.8), int(h * 0.85)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)

        results: List[DetectionBox] = []
        intruder_in_zone = False

        # 1. Check AI Face Detection inside or near Guard Zone
        for (x1, y1, x2, y2, conf) in self.detect_faces_raw(frame):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_zone = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            if in_zone:
                intruder_in_zone = True
            results.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label="INTRUDER FACE!" if in_zone else "Face Outside Zone",
                    confidence=conf,
                    color=(0, 0, 255) if in_zone else (0, 220, 255),
                )
            )

        # 2. Check Motion Contours inside Guard Zone
        if self.prev_gray is not None and self.prev_gray.shape == blur.shape:
            delta = cv2.absdiff(self.prev_gray, blur)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 1200:
                    continue
                x, y, cw, ch = cv2.boundingRect(c)
                cx, cy = x + cw // 2, y + ch // 2
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    intruder_in_zone = True
                    results.append(
                        DetectionBox(
                            x1=x,
                            y1=y,
                            x2=x + cw,
                            y2=y + ch,
                            label="MOTION IN ZONE",
                            confidence=0.9,
                            color=(0, 100, 255),
                        )
                    )
        self.prev_gray = blur

        self.alert_active = intruder_in_zone
        now = time.time()
        if intruder_in_zone and (now - self.last_snapshot_time >= 5.0):
            self.last_snapshot_time = now
            self.total_alerts += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = SNAPSHOTS_DIR / f"alert_{ts}.jpg"
            cv2.imwrite(str(snap_path), frame)
            self.status_banner = f"ALERT #{self.total_alerts}! Saved {snap_path.name}"
        elif intruder_in_zone:
            self.status_banner = f"INTRUDER IN GUARD ZONE! (Total Alerts: {self.total_alerts})"
        else:
            self.status_banner = f"Zone Armed & Clear (Alerts Saved: {self.total_alerts})"

        self.cached_detections = results

    def _draw_overlays(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        # If in Security Guard Mode (4), draw the guarded zone polygon
        if self.mode == 4:
            zx1, zy1 = int(w * 0.2), int(h * 0.18)
            zx2, zy2 = int(w * 0.8), int(h * 0.85)
            zone_color = (0, 0, 255) if self.alert_active else (0, 220, 120)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_color, 2)
            cv2.putText(
                frame,
                "GUARDED SECURITY ZONE",
                (zx1 + 8, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone_color,
                2,
                cv2.LINE_AA,
            )

        # Draw bounding boxes
        for det in self.cached_detections:
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), det.color, 2)

            # Corner accents
            line_len = min(16, (det.x2 - det.x1) // 4)
            cv2.line(frame, (det.x1, det.y1), (det.x1 + line_len, det.y1), det.color, 3)
            cv2.line(frame, (det.x1, det.y1), (det.x1, det.y1 + line_len), det.color, 3)

            # Label pill
            (tw, th), _ = cv2.getTextSize(det.label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ly = max(det.y1 - 6, th + 10)
            cv2.rectangle(
                frame,
                (det.x1, ly - th - 6),
                (det.x1 + tw + 10, ly + 4),
                det.color,
                -1,
            )
            cv2.putText(
                frame,
                det.label,
                (det.x1 + 5, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

            # Sub-boxes (Eyes / Smile)
            for (sx, sy, sw, sh, slabel) in det.sub_boxes:
                scolor = (255, 255, 0) if slabel == "Eye" else (0, 255, 255)
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), scolor, 1)
                cv2.putText(
                    frame,
                    slabel,
                    (sx, max(12, sy - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    scolor,
                    1,
                    cv2.LINE_AA,
                )
