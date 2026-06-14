import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return bordered


def preprocess(image_path: Path, imgsz: int):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, imgsz)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(np.ascontiguousarray(image), axis=0)
    return image


def main():
    parser = argparse.ArgumentParser(description="Quick ONNX validation.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]

    blob = preprocess(args.image, args.imgsz)
    outputs = session.run(output_names, {input_name: blob})

    print(f"[INFO] input  name : {input_name}")
    print(f"[INFO] output name : {output_names}")
    for idx, output in enumerate(outputs):
        print(f"[INFO] output[{idx}] shape={output.shape}, dtype={output.dtype}")
        print(f"[INFO] output[{idx}] min={output.min():.6f}, max={output.max():.6f}")


if __name__ == "__main__":
    main()
