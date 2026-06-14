from pathlib import Path

import cv2
import numpy as np


def load_class_names(path):
    class_path = Path(path)
    return [line.strip() for line in class_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def preprocess_bgr(image, imgsz=640):
    padded, ratio, dwdh = letterbox(image, (imgsz, imgsz))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(np.ascontiguousarray(blob), axis=0)
    return blob, ratio, dwdh


def xywh2xyxy(boxes):
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms(boxes, scores, iou_thres):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        remain = np.where(ious < iou_thres)[0]
        order = order[remain + 1]
    return keep


def scale_boxes(boxes, original_shape, ratio, dwdh):
    dw, dh = dwdh
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio
    h, w = original_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes


def postprocess_yolov5(prediction, original_shape, ratio, dwdh, conf_thres=0.25, iou_thres=0.45):
    pred = prediction[0]
    if pred.ndim != 2 or pred.shape[1] < 6:
        raise ValueError(f"unexpected prediction shape: {prediction.shape}")

    obj_conf = pred[:, 4:5]
    cls_scores = pred[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
    scores = obj_conf[:, 0] * cls_conf

    mask = scores >= conf_thres
    if not np.any(mask):
        return []

    pred = pred[mask]
    cls_ids = cls_ids[mask]
    scores = scores[mask]

    boxes = xywh2xyxy(pred[:, :4])
    boxes = scale_boxes(boxes, original_shape, ratio, dwdh)

    detections = []
    for class_id in np.unique(cls_ids):
        class_mask = cls_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        keep = nms(class_boxes, class_scores, iou_thres)
        for idx in keep:
            detections.append(
                {
                    "class_id": int(class_id),
                    "score": float(class_scores[idx]),
                    "box": class_boxes[idx].astype(int).tolist(),
                }
            )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def draw_detections(image, detections, class_names):
    canvas = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        class_id = item["class_id"]
        score = item["score"]
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} {score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas
