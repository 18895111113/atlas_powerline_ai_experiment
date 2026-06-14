$YoloV5Dir = ".\yolov5"
$DataYaml = ".\configs\powerline_data.yaml"

python .\src\pc\train_yolov5.py `
  --yolov5-dir $YoloV5Dir `
  --data $DataYaml `
  --epochs 150 `
  --batch 16 `
  --imgsz 640 `
  --weights yolov5s.pt `
  --project runs_powerline `
  --name yolov5s_powerline
