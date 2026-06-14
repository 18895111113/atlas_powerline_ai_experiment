import argparse
from pathlib import Path

from icrawler.builtin import BingImageCrawler


QUERIES = {
    "crane": [
        "mobile crane close up construction site",
        "crawler crane boom close up construction site",
        "truck crane near power line construction",
        "tower crane construction site close up",
        "construction crane boom detail",
        "mobile crane urban construction site crowded background",
        "truck crane near overhead power lines",
        "crane boom among buildings wires",
        "crawler crane construction site workers vehicles",
        "crane close up cluttered construction site",
        "吊车 工地 高压线",
        "汽车吊 工地 近景",
        "履带吊 施工现场 复杂背景",
        "塔吊 城市工地 高压线",
        "吊车 电力施工 现场",
    ],
    "excavator": [
        "excavator close up construction site",
        "hydraulic excavator arm close up",
        "excavator digging close view construction",
        "excavator near power line construction",
        "crawler excavator close up site",
        "excavator cluttered construction site workers vehicles",
        "excavator near overhead power lines",
        "excavator forest roadside construction background",
        "excavator close up busy construction site",
        "excavator demolition site complex background",
        "挖掘机 工地 近景",
        "挖掘机 高压线 附近",
        "挖掘机 施工现场 复杂背景",
        "挖掘机 土方 现场",
        "挖掘机 电力施工",
    ],
}


def download_category(category: str, output_root: Path, max_per_query: int, start_query: int):
    category_root = output_root / category
    category_root.mkdir(parents=True, exist_ok=True)

    for index, query in enumerate(QUERIES[category], start=1):
        if index < start_query:
            continue
        query_root = category_root / f"q{index:02d}"
        query_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {category}: {query}")
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(query_root)},
        )
        crawler.crawl(keyword=query, max_num=max_per_query, file_idx_offset=0)


def main():
    parser = argparse.ArgumentParser(description="Download unlabeled crane/excavator reference images.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/sources/web_extra_unlabeled"))
    parser.add_argument("--max-per-query", type=int, default=60)
    parser.add_argument("--start-query", type=int, default=1)
    parser.add_argument("--category", choices=["crane", "excavator", "all"], default="all")
    args = parser.parse_args()

    categories = ["crane", "excavator"] if args.category == "all" else [args.category]
    for category in categories:
        download_category(category, args.output_root, args.max_per_query, args.start_query)


if __name__ == "__main__":
    main()
