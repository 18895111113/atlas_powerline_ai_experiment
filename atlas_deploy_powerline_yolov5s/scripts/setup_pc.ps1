$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements\pc.txt

if (-not (Test-Path ".\yolov5")) {
  git clone https://github.com/ultralytics/yolov5.git
}

if (-not (Test-Path ".\yolov5\train.py")) {
  throw "YOLOv5 repository is incomplete. Delete .\yolov5 and rerun this script when GitHub is reachable."
}

.\.venv\Scripts\pip.exe install -r .\yolov5\requirements.txt

Write-Host "[INFO] PC environment is ready."
Write-Host "[INFO] Activate with: .\.venv\Scripts\activate"
