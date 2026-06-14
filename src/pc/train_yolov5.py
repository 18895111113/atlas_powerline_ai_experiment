import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Wrapper to train YOLOv5.")
    parser.add_argument("--yolov5-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--project", default="runs_powerline")
    parser.add_argument("--name", default="yolov5s_powerline")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    yolov5_dir = args.yolov5_dir.resolve()
    data_yaml = args.data.resolve()
    train_py = yolov5_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found: {train_py}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")

    cmd = [
        sys.executable,
        str(train_py),
        "--img",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--data",
        str(data_yaml),
        "--weights",
        args.weights,
        "--project",
        args.project,
        "--name",
        args.name,
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[INFO] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=yolov5_dir)


if __name__ == "__main__":
    main()
