#!/usr/bin/env python3
"""
Downloads ultra-lightweight OpenCV DNN models optimized for Raspberry Pi 4 (4GB RAM / 5V 2A power budget).
No PyTorch or TensorFlow required — runs purely on OpenCV's C++ DNN engine!
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    # 1. OpenCV ResNet-10 SSD Face Detector (FP16 - only 5.1 MB, high accuracy, low CPU)
    "face_deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/"
        "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    ),
    # 2. MobileNet-SSD Object Detector (20 everyday classes: person, bottle, chair, tvmonitor, cat, dog...)
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
    ),
}

HAAR_FILES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
    "haarcascade_smile.xml",
]


def download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [OK] Already exists: {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    print(f"  [..] Downloading {dest.name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"  [OK] Downloaded {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as exc:
        print(f"  [!!] Failed to download {dest.name}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def copy_or_download_haar():
    system_haar_dirs = [
        Path("/usr/share/opencv4/haarcascades"),
        Path("/usr/local/share/opencv4/haarcascades"),
    ]
    try:
        import cv2
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            system_haar_dirs.insert(0, Path(cv2.data.haarcascades))
    except Exception:
        pass

    for fname in HAAR_FILES:
        dest = MODELS_DIR / fname
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  [OK] Haar cascade ready: {fname}")
            continue

        copied = False
        for d in system_haar_dirs:
            src = d / fname
            if src.exists():
                shutil.copy2(src, dest)
                print(f"  [OK] Copied {fname} from {d}")
                copied = True
                break

        if not copied:
            url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{fname}"
            download_file(url, dest)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing AI Models in {MODELS_DIR} ===")

    all_ok = True
    for filename, url in MODELS.items():
        if not download_file(url, MODELS_DIR / filename):
            all_ok = False

    copy_or_download_haar()

    if all_ok:
        print("\n[SUCCESS] All AI models are ready for Laptop & Raspberry Pi 4!")
    else:
        print("\n[WARN] Some models could not be downloaded. Fallback Haar Cascades will still work.")
        sys.exit(0)


if __name__ == "__main__":
    main()
