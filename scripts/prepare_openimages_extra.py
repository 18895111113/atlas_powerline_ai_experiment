import argparse
import random
import shutil
from pathlib import Path


SOURCES = {
    "person": {
        "root": Path("datasets/sources/openimages_extra/person/person"),
        "target_class": 3,
    },
    "kite": {
        "root": Path("datasets/sources/openimages_extra/kite/kite"),
        "target_class": 2,
    },
    "balloon": {
        "root": Path("datasets/sources/openimages_extra/balloon/balloon"),
        "target_class": 2,
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def remap_label(src_label: Path, dst_label: Path, target_class: int):
    lines = []
    for raw in src_label.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        parts[0] = str(target_class)
        lines.append(" ".join(parts))
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_name(index: int, total: int):
    ratio = index / max(total, 1)
    if ratio < 0.85:
        return "train"
    if ratio < 0.95:
        return "val"
    return "test"


def prepare_source(name: str, spec: dict, output_root: Path, seed: int):
    image_dir = spec["root"] / "images"
    label_dir = spec["root"] / "darknet"
    if not image_dir.exists() or not label_dir.exists():
        print(f"[WARN] skip {name}: missing images or darknet labels")
        return 0

    items = []
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            items.append((image_path, label_path))

    random.Random(seed).shuffle(items)

    for index, (image_path, label_path) in enumerate(items):
        split = split_name(index, len(items))
        dst_image_dir = output_root / "images" / split
        dst_label_dir = output_root / "labels" / split
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        stem = f"openimages_{name}_{image_path.stem}"
        dst_image = dst_image_dir / f"{stem}{image_path.suffix.lower()}"
        dst_label = dst_label_dir / f"{stem}.txt"
        shutil.copy2(image_path, dst_image)
        remap_label(label_path, dst_label, spec["target_class"])

    print(f"[INFO] {name}: prepared {len(items)} labeled images")
    return len(items)


def write_data_yaml(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "path: .",
                "train:",
                "  - datasets/powerline/images/train",
                "  - datasets/openimages_powerline_extra/images/train",
                "val:",
                "  - datasets/powerline/images/val",
                "  - datasets/openimages_powerline_extra/images/val",
                "test:",
                "  - datasets/powerline/images/test",
                "  - datasets/openimages_powerline_extra/images/test",
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
    parser = argparse.ArgumentParser(description="Prepare downloaded Open Images extras for YOLOv5 training.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/openimages_powerline_extra"))
    parser.add_argument("--data-yaml", type=Path, default=Path("configs/powerline_v2_data.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = 0
    for name, spec in SOURCES.items():
        total += prepare_source(name, spec, args.output_root, args.seed)
    write_data_yaml(args.data_yaml)
    print(f"[INFO] total prepared images: {total}")
    print(f"[INFO] data yaml: {args.data_yaml}")


if __name__ == "__main__":
    main()
