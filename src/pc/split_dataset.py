import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()])


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, out_root: Path, split: str):
    image_dst = out_root / "images" / split / image_path.name
    label_dst = out_root / "labels" / split / label_path.name
    ensure_dir(image_dst.parent)
    ensure_dir(label_dst.parent)
    shutil.copy2(image_path, image_dst)
    shutil.copy2(label_path, label_dst)


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test.")
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--source-labels", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    images = collect_images(args.source_images)
    if not images:
        raise RuntimeError("no source images found")

    random.seed(args.seed)
    random.shuffle(images)

    train_end = int(len(images) * args.train_ratio)
    val_end = train_end + int(len(images) * args.val_ratio)

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, items in split_map.items():
        for image_path in items:
            label_path = (args.source_labels / image_path.stem).with_suffix(".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"missing label file for {image_path.name}: {label_path}")
            copy_pair(image_path, label_path, args.output_root, split)

    for split, items in split_map.items():
        print(f"[INFO] {split}: {len(items)}")


if __name__ == "__main__":
    main()
