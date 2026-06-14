$YoloV5Dir = ".\yolov5"
$Weights = ".\yolov5\runs_powerline\yolov5s_powerline\weights\best.pt"

python .\src\pc\export_onnx.py `
  --yolov5-dir $YoloV5Dir `
  --weights $Weights `
  --imgsz 640

$Onnx = ".\yolov5\runs_powerline\yolov5s_powerline\weights\best.onnx"
if (Test-Path $Onnx) {
  New-Item -ItemType Directory -Force -Path ".\models\onnx" | Out-Null
  Copy-Item $Onnx ".\models\onnx\best.onnx" -Force
  Write-Host "[INFO] copied ONNX to .\models\onnx\best.onnx"
}
