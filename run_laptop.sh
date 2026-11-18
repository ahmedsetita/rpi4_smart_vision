#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Use system Python 3.10 which has OpenCV 4.5.4 + DNN + GUI support pre-installed
PYTHON_BIN="/usr/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "Starting RPi 4 Smart AI Vision Suite on Laptop..."
exec "$PYTHON_BIN" "$DIR/app.py" "$@"
