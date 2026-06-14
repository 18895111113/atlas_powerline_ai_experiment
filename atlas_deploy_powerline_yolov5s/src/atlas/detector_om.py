from src.atlas.acl_runtime import AclModelRunner
from src.atlas.yolo_postprocess import postprocess_yolov5, preprocess_bgr


class PowerlineDetector:
    def __init__(self, model_path, imgsz=640, conf_thres=0.25, iou_thres=0.45, device_id=0):
        self.runner = AclModelRunner(model_path=model_path, device_id=device_id)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame):
        blob, ratio, dwdh = preprocess_bgr(frame, self.imgsz)
        outputs = self.runner.infer(blob)
        prediction = outputs[0]
        return postprocess_yolov5(
            prediction=prediction,
            original_shape=frame.shape,
            ratio=ratio,
            dwdh=dwdh,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )

    def release(self):
        self.runner.release()
