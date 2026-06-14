#!/bin/bash
set -e

echo "[INFO] checking npu-smi"
npu-smi info

echo "[INFO] checking ATC"
if atc --version >/tmp/atc_version.log 2>&1; then
  cat /tmp/atc_version.log
elif atc -v >/tmp/atc_version.log 2>&1; then
  cat /tmp/atc_version.log
else
  echo "[WARN] atc version flag is not supported by this toolkit; checking atc --help instead."
  atc --help | head -n 20
fi

echo "[INFO] checking pyACL"
python3 - <<'PY'
import acl
print("acl ok")
PY

echo "[INFO] Atlas environment looks available."
