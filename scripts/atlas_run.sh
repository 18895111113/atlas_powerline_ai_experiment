#!/bin/bash
set -e

MODEL_PATH=${1:-./models/powerline_yolov5s.om}
SOURCE=${2:-0}
NAMES=${3:-./configs/classes.txt}
SAVE_PATH=${4:-}

CMD="python3 src/atlas/video_infer.py --model ${MODEL_PATH} --source ${SOURCE} --names ${NAMES}"

if [ -n "${SAVE_PATH}" ]; then
  CMD="${CMD} --save-path ${SAVE_PATH}"
fi

echo "[INFO] ${CMD}"
eval "${CMD}"
