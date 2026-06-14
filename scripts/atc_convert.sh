#!/bin/bash
set -e

MODEL_PATH=${1:-./best.onnx}
SOC_VERSION=${2:-Ascend310}
OUTPUT_DIR=${3:-./models}
OUTPUT_NAME=${4:-powerline_yolov5s}
INPUT_NAME=${5:-images}
INPUT_SHAPE=${6:-1,3,640,640}

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] model      : ${MODEL_PATH}"
echo "[INFO] soc        : ${SOC_VERSION}"
echo "[INFO] output dir : ${OUTPUT_DIR}"
echo "[INFO] input      : ${INPUT_NAME}:${INPUT_SHAPE}"

atc \
  --model="${MODEL_PATH}" \
  --framework=5 \
  --output="${OUTPUT_DIR}/${OUTPUT_NAME}" \
  --input_shape="${INPUT_NAME}:${INPUT_SHAPE}" \
  --soc_version="${SOC_VERSION}" \
  --log=error

echo "[INFO] OM model generated at ${OUTPUT_DIR}/${OUTPUT_NAME}.om"
