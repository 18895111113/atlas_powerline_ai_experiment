import argparse
import time
from pathlib import Path

import cv2

from src.atlas.detector_om import PowerlineDetector
from src.atlas.yolo_postprocess import draw_detections, load_class_names


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Real-time OM inference on Atlas.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--names", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--no-view", action="store_true", help="Disable cv2.imshow for SSH/headless runs.")
    parser.add_argument("--view-width", type=int, default=1280)
    parser.add_argument("--view-height", type=int, default=720)
    args = parser.parse_args()

    class_names = load_class_names(args.names)
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")

    writer = None
    detector = PowerlineDetector(
        model_path=str(args.model),
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device_id=args.device_id,
    )

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.view_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.view_height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save_path), fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            infer_start = time.time()
            detections = detector.detect(frame)
            infer_cost = time.time() - infer_start

            canvas = draw_detections(frame, detections, class_names)
            frame_count += 1
            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_count / elapsed

            cv2.putText(
                canvas,
                f"FPS: {fps:.2f} | Infer: {infer_cost * 1000:.1f} ms",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(canvas)

            if not args.no_view:
                cv2.imshow("Atlas Powerline Detection", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
            elif frame_count % 30 == 0:
                print(f"[INFO] frames={frame_count} fps={fps:.2f} infer_ms={infer_cost * 1000:.1f}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        detector.release()
        if not args.no_view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
