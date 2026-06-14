import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TARGET_CLASSES = {0, 1}


def read_labels(label_path: Path):
    labels = []
    if not label_path.exists():
        return labels
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[1:]]
        labels.append((cls, x, y, w, h))
    return labels


def yolo_to_xyxy(label, image_w, image_h):
    cls, x, y, w, h = label
    x1 = (x - w / 2) * image_w
    y1 = (y - h / 2) * image_h
    x2 = (x + w / 2) * image_w
    y2 = (y + h / 2) * image_h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, crop_w, crop_h):
    x1 = max(0, min(crop_w, x1))
    y1 = max(0, min(crop_h, y1))
    x2 = max(0, min(crop_w, x2))
    y2 = max(0, min(crop_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bw = (x2 - x1) / crop_w
    bh = (y2 - y1) / crop_h
    bx = ((x1 + x2) / 2) / crop_w
    by = ((y1 + y2) / 2) / crop_h
    return cls, bx, by, bw, bh


def clamp_crop(left, top, crop_w, crop_h, image_w, image_h):
    left = max(0, min(image_w - crop_w, left))
    top = max(0, min(image_h - crop_h, top))
    return int(round(left)), int(round(top)), int(round(left + crop_w)), int(round(top + crop_h))


def crop_labels(labels, crop_box, image_w, image_h, min_keep=0.35):
    left, top, right, bottom = crop_box
    crop_w = right - left
    crop_h = bottom - top
    output = []

    for label in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)
        inter_x1 = max(x1, left)
        inter_y1 = max(y1, top)
        inter_x2 = min(x2, right)
        inter_y2 = min(y2, bottom)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        if inter_area / box_area < min_keep:
            continue

        yolo = xyxy_to_yolo(
            cls,
            inter_x1 - left,
            inter_y1 - top,
            inter_x2 - left,
            inter_y2 - top,
            crop_w,
            crop_h,
        )
        if yolo is not None:
            output.append(yolo)

    return output


def format_labels(labels):
    return "\n".join(
        f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in labels
    ) + ("\n" if labels else "")


def augment_split(input_root: Path, output_root: Path, split: str, variants_per_image: int, seed: int):
    rng = random.Random(seed + hash(split) % 10000)
    image_dir = input_root / "images" / split
    label_dir = input_root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return 0

    dst_image_dir = output_root / "images" / split
    dst_label_dir = output_root / "labels" / split
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        labels = read_labels(label_dir / f"{image_path.stem}.txt")
        targets = [label for label in labels if label[0] in TARGET_CLASSES]
        if not targets:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_w, image_h = image.size
            pixel_targets = [yolo_to_xyxy(label, image_w, image_h) for label in targets]

            for variant in range(variants_per_image):
                cls, x1, y1, x2, y2 = rng.choice(pixel_targets)
                box_w = max(8, x2 - x1)
                box_h = max(8, y2 - y1)
                scale = rng.uniform(1.15, 1.85)
                crop_w = min(image_w, max(box_w * scale, image_w * rng.uniform(0.28, 0.55)))
                crop_h = min(image_h, max(box_h * scale, image_h * rng.uniform(0.28, 0.55)))
                cx = (x1 + x2) / 2 + rng.uniform(-0.15, 0.15) * box_w
                cy = (y1 + y2) / 2 + rng.uniform(-0.15, 0.15) * box_h
                crop_box = clamp_crop(cx - crop_w / 2, cy - crop_h / 2, crop_w, crop_h, image_w, image_h)
                new_labels = crop_labels(labels, crop_box, image_w, image_h)
                if not any(label[0] in TARGET_CLASSES for label in new_labels):
                    continue

                left, top, right, bottom = crop_box
                crop = image.crop((left, top, right, bottom))
                stem = f"closeup_{image_path.stem}_{variant}"
                crop.save(dst_image_dir / f"{stem}.jpg", quality=92)
                (dst_label_dir / f"{stem}.txt").write_text(format_labels(new_labels), encoding="utf-8")
                generated += 1

    return generated


def write_data_yaml(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "path: .",
                "train:",
                "  - datasets/powerline/images/train",
                "  - datasets/openimages_powerline_extra/images/train",
                "  - datasets/construction_closeup_extra/images/train",
                "val:",
                "  - datasets/powerline/images/val",
                "  - datasets/openimages_powerline_extra/images/val",
                "  - datasets/construction_closeup_extra/images/val",
                "test:",
                "  - datasets/powerline/images/test",
                "  - datasets/openimages_powerline_extra/images/test",
                "  - datasets/construction_closeup_extra/images/test",
                "",
                "nc: 4",
                "names:",
                "  0: crane",
                "  1: excavator",
                "  2: foreign_object",
                "  3: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Create close-up crane/excavator crops from existing YOLO labels.")
    parser.add_argument("--input-root", type=Path, default=Path("datasets/powerline"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/construction_closeup_extra"))
    parser.add_argument("--data-yaml", type=Path, default=Path("configs/powerline_v2_closeup_data.yaml"))
    parser.add_argument("--variants-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)

    total = 0
    for split in ("train", "val", "test"):
        count = augment_split(args.input_root, args.output_root, split, args.variants_per_image, args.seed)
        total += count
        print(f"[INFO] {split}: generated {count} close-up images")

    write_data_yaml(args.data_yaml)
    print(f"[INFO] total generated images: {total}")
    print(f"[INFO] data yaml: {args.data_yaml}")


if __name__ == "__main__":
    main()
