import argparse
import ast
import random
import shutil
from pathlib import Path
from zipfile import ZipFile


TARGET_CLASSES = {
    "crane": 0,
    "excavator": 1,
    "foreign_object": 2,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_source_classes(source_root: Path) -> list[str]:
    classes_path = source_root / "classes.txt"
    if classes_path.exists():
        return read_classes(classes_path)

    data_yaml = source_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"missing classes.txt or data.yaml: {source_root}")

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("names:"):
            continue
        names_text = line.split(":", 1)[1].strip()
        names = ast.literal_eval(names_text)
        if isinstance(names, dict):
            return [names[index] for index in sorted(names)]
        if isinstance(names, list):
            return [str(name) for name in names]
    raise ValueError(f"could not parse names from: {data_yaml}")


def reset_yolo_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            target = output_root / kind / split
            if target.exists():
                for child in target.iterdir():
                    if child.name == ".gitkeep":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            target.mkdir(parents=True, exist_ok=True)
            (target / ".gitkeep").touch(exist_ok=True)


def split_name(index: int, total: int) -> str:
    ratio = index / max(total, 1)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_yolo_source(
    *,
    source_root: Path,
    output_root: Path,
    source_name: str,
    class_map: dict[str, str],
    seed: int,
) -> dict[str, int]:
    classes = read_source_classes(source_root)
    source_to_target = {}
    for source_class, target_class in class_map.items():
        if source_class not in classes:
            continue
        source_to_target[classes.index(source_class)] = TARGET_CLASSES[target_class]

    if not source_to_target:
        raise ValueError(f"no mapped classes found for {source_root}")

    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    if images_dir.exists() and labels_dir.exists():
        return convert_flat_yolo_source(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_root=output_root,
            source_name=source_name,
            source_to_target=source_to_target,
            seed=seed,
        )

    split_roots = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "test",
    }
    if any((source_root / split / "images").exists() for split in split_roots):
        return convert_split_yolo_source(
            source_root=source_root,
            split_roots=split_roots,
            output_root=output_root,
            source_name=source_name,
            source_to_target=source_to_target,
        )

    raise FileNotFoundError(f"missing YOLO images/labels structure under: {source_root}")


def convert_flat_yolo_source(
    *,
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    source_name: str,
    source_to_target: dict[int, int],
    seed: int,
) -> dict[str, int]:
    label_files = sorted(labels_dir.glob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(label_files)

    stats = {"images": 0, "objects": 0}
    for index, label_path in enumerate(label_files):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue

        converted_lines = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue
            try:
                source_class_id = int(float(parts[0]))
            except ValueError:
                continue
            if source_class_id not in source_to_target:
                continue
            target_class_id = source_to_target[source_class_id]
            converted_lines.append(" ".join([str(target_class_id), *parts[1:]]))

        if not converted_lines:
            continue

        split = split_name(index, len(label_files))
        safe_stem = f"{source_name}_{label_path.stem}"
        out_image = output_root / "images" / split / f"{safe_stem}{image_path.suffix.lower()}"
        out_label = output_root / "labels" / split / f"{safe_stem}.txt"
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
        stats["images"] += 1
        stats["objects"] += len(converted_lines)

    return stats


def convert_split_yolo_source(
    *,
    source_root: Path,
    split_roots: dict[str, str],
    output_root: Path,
    source_name: str,
    source_to_target: dict[int, int],
) -> dict[str, int]:
    stats = {"images": 0, "objects": 0}
    for source_split, target_split in split_roots.items():
        images_dir = source_root / source_split / "images"
        labels_dir = source_root / source_split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue

            converted_lines = []
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    source_class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if source_class_id not in source_to_target:
                    continue
                target_class_id = source_to_target[source_class_id]
                converted_lines.append(" ".join([str(target_class_id), *parts[1:]]))

            if not converted_lines:
                continue

            safe_stem = f"{source_name}_{label_path.stem}"
            out_image = output_root / "images" / target_split / f"{safe_stem}{image_path.suffix.lower()}"
            out_label = output_root / "labels" / target_split / f"{safe_stem}.txt"
            shutil.copy2(image_path, out_image)
            out_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["objects"] += len(converted_lines)

    return stats


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    if (output_dir / "classes.txt").exists() or (output_dir / "data.yaml").exists():
        return output_dir

    candidates = [
        path
        for path in [*output_dir.rglob("classes.txt"), *output_dir.rglob("data.yaml")]
        if (path.parent / "images").exists() or (path.parent / "train" / "images").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"could not find YOLO classes.txt or data.yaml in {zip_path}")
    return candidates[0].parent


def write_data_yaml(output_root: Path, yaml_path: Path) -> None:
    yaml_path.write_text(
        """path: ./datasets/powerline
train: images/train
val: images/val
test: images/test

names:
  0: crane
  1: excavator
  2: foreign_object
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge construction equipment and foreign-object datasets into YOLOv5 layout.")
    parser.add_argument("--construction-root", type=Path, default=Path("datasets/sources/construction_equipment"))
    parser.add_argument("--roboflow-zip", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/powerline"))
    parser.add_argument("--data-yaml", type=Path, default=Path("configs/powerline_data.yaml"))
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    reset_yolo_dirs(args.output_root)

    summaries = []
    construction_stats = convert_yolo_source(
        source_root=args.construction_root,
        output_root=args.output_root,
        source_name="construction",
        class_map={"crane": "crane", "excavator": "excavator"},
        seed=args.seed,
    )
    summaries.append(("construction", construction_stats))

    if args.roboflow_zip is not None:
        roboflow_root = extract_zip(args.roboflow_zip, Path("datasets/sources/roboflow_foreign_objects"))
        roboflow_classes = read_source_classes(roboflow_root)
        foreign_map = {name: "foreign_object" for name in roboflow_classes}
        roboflow_stats = convert_yolo_source(
            source_root=roboflow_root,
            output_root=args.output_root,
            source_name="foreign",
            class_map=foreign_map,
            seed=args.seed + 1,
        )
        summaries.append(("roboflow_foreign_objects", roboflow_stats))

    write_data_yaml(args.output_root, args.data_yaml)

    for name, stats in summaries:
        print(f"[INFO] {name}: {stats['images']} images, {stats['objects']} objects")
    print(f"[INFO] wrote dataset: {args.output_root}")
    print(f"[INFO] wrote yaml: {args.data_yaml}")


if __name__ == "__main__":
    main()
