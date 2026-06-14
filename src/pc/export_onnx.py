import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv5 weights to ONNX.")
    parser.add_argument("--yolov5-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    yolov5_dir = args.yolov5_dir.resolve()
    weights = args.weights.resolve()
    export_py = yolov5_dir / "export.py"
    if not export_py.exists():
        raise FileNotFoundError(f"export.py not found: {export_py}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    cmd = [
        sys.executable,
        str(export_py),
        "--weights",
        str(weights),
        "--img",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--include",
        "onnx",
        "--opset",
        str(args.opset),
    ]
    if not args.no_simplify:
        cmd.append("--simplify")

    print("[INFO] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=yolov5_dir)


if __name__ == "__main__":
    main()
