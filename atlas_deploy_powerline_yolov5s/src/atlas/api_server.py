import argparse
import base64
import json
import tempfile
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from src.atlas.yolo_postprocess import draw_detections, load_class_names


RISK_BY_LABEL = {
    "crane": "high",
    "excavator": "high",
    "foreign_object": "medium",
    "person": "low",
}

ZH_BY_LABEL = {
    "crane": "crane",
    "excavator": "excavator",
    "foreign_object": "foreign_object",
    "person": "person",
}


def parse_source(value: str):
    return int(value) if str(value).isdigit() else value


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_model_registry(args):
    registry_path = resolve_path(args.model_registry)
    if registry_path.exists():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        raw_models = data.get("models", [])
        default_model_id = data.get("default_model_id")
    else:
        raw_models = [
            {
                "id": "default",
                "name": "Default powerline model",
                "om": str(args.model),
                "classes": str(args.names),
                "input_size": args.imgsz,
            }
        ]
        default_model_id = "default"

    models = {}
    for entry in raw_models:
        model_id = str(entry["id"])
        model_path = resolve_path(entry.get("om") or entry.get("model") or args.model)
        names_path = resolve_path(entry.get("classes") or entry.get("names") or args.names)
        models[model_id] = {
            "id": model_id,
            "name": entry.get("name", model_id),
            "model_path": model_path,
            "names_path": names_path,
            "imgsz": int(entry.get("input_size", entry.get("imgsz", args.imgsz))),
            "conf_thres": float(entry.get("conf_thres", args.conf_thres)),
            "iou_thres": float(entry.get("iou_thres", args.iou_thres)),
            "description": entry.get("description", ""),
        }

    if not models:
        raise RuntimeError(f"no models configured in registry: {registry_path}")
    if default_model_id not in models:
        default_model_id = next(iter(models))
    return default_model_id, models


def public_model_info(spec, state, default_model_id):
    model_path = spec["model_path"]
    names_path = spec["names_path"]
    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": spec["description"],
        "default": spec["id"] == default_model_id,
        "available": model_path.exists() and names_path.exists(),
        "loaded": state["detector"] is not None,
        "error": state["error"],
        "model": str(model_path),
        "names": str(names_path),
        "class_names": state["class_names"],
        "input_size": spec["imgsz"],
    }


def request_model_id():
    if request.method == "GET":
        return request.args.get("model_id")
    payload = request.get_json(silent=True) if request.is_json else None
    return request.form.get("model_id") or request.args.get("model_id") or (payload or {}).get("model_id")


def configure_capture(cap, args):
    # Keep only conservative camera hints; some Atlas drivers dislike forced FOURCC/FPS.
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)


def serialize_detections(detections, class_names, width, height):
    result = []
    for item in detections:
        class_id = int(item["class_id"])
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        result.append(
            {
                "class_id": class_id,
                "label": label,
                "zh": ZH_BY_LABEL.get(label, label),
                "risk": RISK_BY_LABEL.get(label, "medium"),
                "score": float(item["score"]),
                "box": [x1, y1, x2, y2],
                "box_norm": [
                    x1 / max(width, 1),
                    y1 / max(height, 1),
                    x2 / max(width, 1),
                    y2 / max(height, 1),
                ],
            }
        )
    return result


def raw_detections_from_serialized(detections):
    return [
        {"class_id": item["class_id"], "score": item["score"], "box": item["box"]}
        for item in detections
    ]


def encode_annotated_image(frame, detections, class_names, jpeg_quality):
    canvas = draw_detections(frame, raw_detections_from_serialized(detections), class_names)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("annotated image jpeg encode failed")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def encode_annotated_jpeg(frame, detections, class_names, jpeg_quality):
    canvas = draw_detections(frame, raw_detections_from_serialized(detections), class_names)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("annotated jpeg encode failed")
    return encoded.tobytes()


def encode_result_header(result):
    payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def make_mjpeg_part(jpeg_bytes, result):
    result_header = encode_result_header(result)
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"X-Atlas-Result: {result_header}\r\n".encode("ascii")
        + b"\r\n"
        + jpeg_bytes
        + b"\r\n"
    )


class AtlasDetectionService:
    def __init__(self, args):
        self.args = args
        self.default_model_id, self.models = load_model_registry(args)
        self.model_states = {}
        self.latest_result = None
        self.latest_lock = Lock()
        self.detector_lock = Lock()
        self.infer_lock = Lock()

        for model_id, spec in self.models.items():
            class_names = []
            error = ""
            if spec["names_path"].exists():
                class_names = load_class_names(spec["names_path"])
            else:
                error = f"class names not found: {spec['names_path']}"
            self.model_states[model_id] = {
                "detector": None,
                "error": error,
                "class_names": class_names,
            }

    def resolve_model_id(self, model_id=None):
        selected = model_id or self.default_model_id
        if selected not in self.models:
            raise KeyError(f"unknown model_id: {selected}")
        return selected

    def model_info(self, model_id):
        selected = self.resolve_model_id(model_id)
        return public_model_info(
            self.models[selected],
            self.model_states[selected],
            self.default_model_id,
        )

    def list_models(self, selected_model_id=None):
        selected = self.resolve_model_id(selected_model_id)
        return {
            "default_model_id": self.default_model_id,
            "selected_model_id": selected,
            "models": [
                public_model_info(self.models[model_id], self.model_states[model_id], self.default_model_id)
                for model_id in self.models
            ],
        }

    def class_names_for(self, model_id):
        selected = self.resolve_model_id(model_id)
        return self.model_states[selected]["class_names"]

    def load_detector(self, model_id=None):
        selected = self.resolve_model_id(model_id)
        spec = self.models[selected]
        state = self.model_states[selected]

        if state["detector"] is not None:
            return state["detector"]

        with self.detector_lock:
            if state["detector"] is not None:
                return state["detector"]

            if not spec["names_path"].exists():
                state["error"] = f"class names not found: {spec['names_path']}"
                return None
            if not spec["model_path"].exists():
                state["error"] = f"model not found: {spec['model_path']}"
                return None

            try:
                from src.atlas.detector_om import PowerlineDetector

                state["detector"] = PowerlineDetector(
                    model_path=str(spec["model_path"]),
                    imgsz=spec["imgsz"],
                    conf_thres=spec["conf_thres"],
                    iou_thres=spec["iou_thres"],
                    device_id=self.args.device_id,
                )
                state["error"] = ""
            except Exception as error:
                state["error"] = str(error)
                state["detector"] = None

            return state["detector"]

    def health(self, model_id=None):
        selected = self.resolve_model_id(model_id)
        detector = self.load_detector(selected)
        spec = self.models[selected]
        state = self.model_states[selected]
        return {
            "status": "ok" if detector is not None else "degraded",
            "model_loaded": detector is not None,
            "model_id": selected,
            "model_name": spec["name"],
            "model": str(spec["model_path"]),
            "names": str(spec["names_path"]),
            "class_names": state["class_names"],
            "camera_source": str(self.args.camera_source),
            "device_id": self.args.device_id,
            "error": state["error"],
            "default_model_id": self.default_model_id,
            "models": self.list_models(selected)["models"],
        }

    def detect_frame(self, frame, model_id=None):
        selected = self.resolve_model_id(model_id)
        detector = self.load_detector(selected)
        if detector is None:
            raise RuntimeError(self.model_states[selected]["error"] or "detector is not ready")

        with self.infer_lock:
            started = time.time()
            raw_detections = detector.detect(frame)
        infer_ms = (time.time() - started) * 1000
        height, width = frame.shape[:2]
        class_names = self.model_states[selected]["class_names"]
        detections = serialize_detections(raw_detections, class_names, width, height)
        return {
            "time": time.strftime("%H:%M:%S"),
            "model_id": selected,
            "model_name": self.models[selected]["name"],
            "frame_width": width,
            "frame_height": height,
            "infer_ms": infer_ms,
            "detections": detections,
        }

    def set_latest(self, result):
        with self.latest_lock:
            self.latest_result = result

    def get_latest(self):
        with self.latest_lock:
            return self.latest_result


def create_app(args):
    app = Flask(__name__)
    service = AtlasDetectionService(args)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/models", methods=["GET"])
    def models():
        try:
            return jsonify(service.list_models(request.args.get("model_id")))
        except KeyError as error:
            return jsonify({"error": str(error)}), 400

    @app.route("/api/health", methods=["GET"])
    def health():
        try:
            status = service.health(request.args.get("model_id"))
        except KeyError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(status)

    @app.route("/api/detect/image", methods=["POST", "OPTIONS"])
    def detect_image():
        if request.method == "OPTIONS":
            return "", 204
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "missing multipart field: file"}), 400

        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "failed to decode image"}), 400

        try:
            result = service.detect_frame(frame, request_model_id())
            result["annotated_image"] = encode_annotated_image(
                frame,
                result["detections"],
                service.class_names_for(result["model_id"]),
                args.jpeg_quality,
            )
        except KeyError as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            return jsonify({"error": str(error)}), 503

        service.set_latest(result)
        return jsonify(result)

    @app.route("/api/detect/video", methods=["POST", "OPTIONS"])
    def detect_video():
        if request.method == "OPTIONS":
            return "", 204
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "missing multipart field: file"}), 400

        model_id = request_model_id()
        suffix = Path(uploaded.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            uploaded.save(temp_file.name)
            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                return jsonify({"error": "failed to open uploaded video"}), 400

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stride = max(1, int(native_fps / max(args.video_sample_fps, 0.1)))
            events = []
            frame_index = 0
            processed = 0
            latest = None
            started = time.time()

            try:
                while processed < args.video_max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % stride != 0:
                        frame_index += 1
                        continue

                    result = service.detect_frame(frame, model_id)
                    result["annotated_image"] = encode_annotated_image(
                        frame,
                        result["detections"],
                        service.class_names_for(result["model_id"]),
                        args.jpeg_quality,
                    )
                    result["frame_index"] = frame_index
                    result["frame_time"] = frame_index / native_fps
                    events.append(result)
                    latest = result
                    processed += 1
                    frame_index += 1
            except KeyError as error:
                cap.release()
                return jsonify({"error": str(error)}), 400
            except Exception as error:
                cap.release()
                return jsonify({"error": str(error)}), 503
            finally:
                cap.release()

        summary = {
            "time": time.strftime("%H:%M:%S"),
            "model_id": latest["model_id"] if latest else service.resolve_model_id(model_id),
            "model_name": latest["model_name"] if latest else service.models[service.resolve_model_id(model_id)]["name"],
            "fps": processed / max(time.time() - started, 1e-6),
            "processed_frames": processed,
            "events": events,
            "detections": latest["detections"] if latest else [],
            "frame_width": latest["frame_width"] if latest else 0,
            "frame_height": latest["frame_height"] if latest else 0,
            "infer_ms_avg": float(np.mean([event["infer_ms"] for event in events])) if events else 0.0,
        }
        service.set_latest(summary)
        return jsonify(summary)

    @app.route("/api/detect/video/stream", methods=["POST", "OPTIONS"])
    def detect_video_stream():
        if request.method == "OPTIONS":
            return "", 204
        try:
            model_id = service.resolve_model_id(request_model_id())
        except KeyError as error:
            return jsonify({"error": str(error)}), 400
        if service.load_detector(model_id) is None:
            return jsonify({"error": service.model_states[model_id]["error"] or "detector is not ready"}), 503

        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "missing multipart field: file"}), 400

        suffix = Path(uploaded.filename or "upload.mp4").suffix or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        try:
            with temp_file:
                uploaded.save(temp_file.name)

            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                cap.release()
                temp_path.unlink(missing_ok=True)
                return jsonify({"error": "failed to open uploaded video"}), 400
            cap.release()
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        def generate():
            cap = cv2.VideoCapture(str(temp_path))
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stride = max(1, int(native_fps / max(args.video_sample_fps, 0.1)))
            frame_index = 0
            processed = 0
            started = time.time()

            try:
                while processed < args.video_max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % stride != 0:
                        frame_index += 1
                        continue

                    result = service.detect_frame(frame, model_id)
                    result["frame_index"] = frame_index
                    result["frame_time"] = frame_index / native_fps
                    processed += 1
                    result["fps"] = processed / max(time.time() - started, 1e-6)
                    service.set_latest(result)

                    jpeg_bytes = encode_annotated_jpeg(
                        frame,
                        result["detections"],
                        service.class_names_for(result["model_id"]),
                        args.jpeg_quality,
                    )
                    yield make_mjpeg_part(jpeg_bytes, result)
                    frame_index += 1
            except Exception as error:
                service.set_latest(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "model_id": model_id,
                        "error": f"video stream infer failed: {error}",
                        "detections": [],
                    }
                )
                print(f"[ERROR] video stream infer failed: {error}", flush=True)
            finally:
                cap.release()
                temp_path.unlink(missing_ok=True)

        headers = {
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers=headers,
        )

    @app.route("/api/stream", methods=["GET"])
    def stream():
        try:
            model_id = service.resolve_model_id(request.args.get("model_id"))
        except KeyError as error:
            return jsonify({"error": str(error)}), 400
        if service.load_detector(model_id) is None:
            return jsonify({"error": service.model_states[model_id]["error"] or "detector is not ready"}), 503

        def generate():
            cap = cv2.VideoCapture(parse_source(args.camera_source))
            if not cap.isOpened():
                service.set_latest(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "model_id": model_id,
                        "error": f"failed to open camera source: {args.camera_source}",
                        "detections": [],
                    }
                )
                return

            configure_capture(cap, args)
            frame_count = 0
            read_failures = 0
            started = time.time()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        read_failures += 1
                        if read_failures < args.camera_read_retry:
                            time.sleep(0.03)
                            continue
                        service.set_latest(
                            {
                                "time": time.strftime("%H:%M:%S"),
                                "model_id": model_id,
                                "error": f"camera frame read failed: {args.camera_source}",
                                "detections": [],
                            }
                        )
                        print(f"[WARN] camera frame read failed: {args.camera_source}", flush=True)
                        break

                    read_failures = 0

                    try:
                        result = service.detect_frame(frame, model_id)
                    except Exception as error:
                        service.set_latest(
                            {
                                "time": time.strftime("%H:%M:%S"),
                                "model_id": model_id,
                                "error": f"stream infer failed: {error}",
                                "detections": [],
                            }
                        )
                        print(f"[ERROR] stream infer failed: {error}", flush=True)
                        break
                    frame_count += 1
                    result["fps"] = frame_count / max(time.time() - started, 1e-6)
                    service.set_latest(result)

                    raw_detections = [
                        {"class_id": item["class_id"], "score": item["score"], "box": item["box"]}
                        for item in result["detections"]
                    ]
                    canvas = draw_detections(frame, raw_detections, service.class_names_for(result["model_id"]))
                    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    if not ok:
                        service.set_latest(
                            {
                                "time": time.strftime("%H:%M:%S"),
                                "model_id": model_id,
                                "error": "stream jpeg encode failed",
                                "detections": [],
                            }
                        )
                        print("[ERROR] stream jpeg encode failed", flush=True)
                        continue
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            finally:
                cap.release()

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/events", methods=["GET"])
    def events():
        def generate():
            while True:
                latest = service.get_latest()
                if latest is None:
                    payload = json.dumps({"message": "waiting_for_camera_frame"}, ensure_ascii=False)
                    yield f"event: status\ndata: {payload}\n\n"
                elif latest.get("error"):
                    payload = json.dumps({"message": latest["error"], "model_id": latest.get("model_id")}, ensure_ascii=False)
                    yield f"event: status\ndata: {payload}\n\n"
                else:
                    payload = json.dumps(latest, ensure_ascii=False)
                    yield f"event: result\ndata: {payload}\n\n"
                time.sleep(args.event_interval)

        return Response(generate(), mimetype="text/event-stream")

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="HTTP bridge for Atlas powerline detection.")
    parser.add_argument("--model", type=Path, default=Path("models/om/powerline_yolov5s.om"))
    parser.add_argument("--names", type=Path, default=Path("configs/classes.txt"))
    parser.add_argument("--model-registry", type=Path, default=Path("configs/model_registry.json"))
    parser.add_argument("--camera-source", type=str, default="0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--event-interval", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--video-sample-fps", type=float, default=8.0)
    parser.add_argument("--video-max-frames", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")
    parser.add_argument("--camera-read-retry", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
