import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def validate_label_file(label_path: Path, num_classes: int):
    errors = []
    if not label_path.exists():
        errors.append(f"missing label file: {label_path}")
        return errors, 0

    count = 0
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no} expects 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path}:{line_no} has non-numeric content")
            continue
        if cls < 0 or cls >= num_classes:
            errors.append(f"{label_path}:{line_no} class id {cls} out of range [0, {num_classes - 1}]")
        for name, value in zip(("x", "y", "w", "h"), (x, y, w, h)):
            if not (0.0 <= value <= 1.0):
                errors.append(f"{label_path}:{line_no} {name}={value} not in [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{label_path}:{line_no} width/height must be > 0")
        count += 1
    return errors, count


def main():
    parser = argparse.ArgumentParser(description="Check YOLO-format dataset integrity.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--num-classes", required=True, type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"images directory not found: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_root}")

    all_images = list(iter_images(images_root))
    if not all_images:
        raise RuntimeError(f"no images found under {images_root}")

    total_boxes = 0
    errors = []

    for image_path in all_images:
        rel = image_path.relative_to(images_root)
        label_path = (labels_root / rel).with_suffix(".txt")
        label_errors, box_count = validate_label_file(label_path, args.num_classes)
        errors.extend(label_errors)
        total_boxes += box_count

    print(f"[INFO] dataset root : {dataset_root}")
    print(f"[INFO] total images : {len(all_images)}")
    print(f"[INFO] total boxes  : {total_boxes}")

    if errors:
        print(f"[ERROR] found {len(errors)} issue(s):")
        for item in errors[:100]:
            print(f"  - {item}")
        if len(errors) > 100:
            print(f"  - ... and {len(errors) - 100} more")
        raise SystemExit(1)

    print("[INFO] dataset check passed")


if __name__ == "__main__":
    main()
