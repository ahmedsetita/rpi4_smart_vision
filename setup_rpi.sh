#!/usr/bin/env bash
# ==============================================================================
# One-Step Setup Script for Raspberry Pi 4 (4GB RAM / Camera Module V2)
# Works on Raspberry Pi OS (Bookworm & Bullseye) without heavy pip compilation!
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== [1/3] Installing lightweight native packages on Raspberry Pi OS ==="
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-numpy libcamera-apps

echo "=== [2/3] Checking / Downloading AI Models ==="
python3 "$DIR/download_models.py"

echo "=== [3/3] Verifying Raspberry Pi Camera Module V2 ==="
if command -v rpicam-hello >/dev/null 2>&1; then
    echo "[INFO] Detected modern rpicam stack (Raspberry Pi OS Bookworm)."
elif command -v libcamera-hello >/dev/null 2>&1; then
    echo "[INFO] Detected libcamera stack (Raspberry Pi OS Bullseye)."
fi

echo ""
echo "============================================================"
echo " Setup Complete! Run the AI Vision Suite on Raspberry Pi with:"
echo "   python3 app.py"
echo "============================================================"
