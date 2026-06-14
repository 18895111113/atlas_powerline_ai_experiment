import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "atlas-powerline-ai-dataset-builder/1.0 (local research dataset preparation)",
}
COMMONS_CATEGORIES = {
    "crane": [
        "Category:Mobile cranes",
        "Category:Crawler cranes",
        "Category:Tower cranes",
        "Category:Truck-mounted cranes",
        "Category:Cranes at construction sites",
    ],
    "excavator": [
        "Category:Excavators",
        "Category:Hydraulic excavators",
        "Category:Crawler excavators",
        "Category:Excavators at construction sites",
    ],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def safe_name(value: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def iter_category_files(category: str, limit: int):
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 1024,
        "format": "json",
    }
    seen = 0
    while seen < limit:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue
            url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
            title = page.get("title", "image")
            if not url:
                continue
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            yield title, url
            seen += 1
            if seen >= limit:
                return
        if "continue" not in data:
            return
        params.update(data["continue"])


def download_file(url: str, output_path: Path):
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def download_category(kind: str, output_root: Path, max_per_category: int):
    kind_root = output_root / kind
    kind_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for category in COMMONS_CATEGORIES[kind]:
        category_name = safe_name(category.replace("Category:", ""))
        category_root = kind_root / category_name
        category_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {kind}: {category}")
        for title, url in iter_category_files(category, max_per_category):
            suffix = Path(urlparse(url).path).suffix.lower()
            filename = f"{safe_name(title.replace('File:', ''))}{suffix}"
            output_path = category_root / filename
            if output_path.exists():
                continue
            try:
                download_file(url, output_path)
                downloaded += 1
            except Exception as error:
                print(f"[WARN] failed: {url} ({error})")
    print(f"[INFO] {kind}: downloaded {downloaded} files")


def main():
    parser = argparse.ArgumentParser(description="Download crane/excavator images from Wikimedia Commons categories.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/sources/commons_equipment_unlabeled"))
    parser.add_argument("--max-per-category", type=int, default=35)
    parser.add_argument("--category", choices=["crane", "excavator", "all"], default="all")
    args = parser.parse_args()

    categories = ["crane", "excavator"] if args.category == "all" else [args.category]
    for category in categories:
        download_category(category, args.output_root, args.max_per_category)


if __name__ == "__main__":
    main()
