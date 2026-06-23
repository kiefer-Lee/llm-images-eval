from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from ..coco_io import write_json


VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian", "supercategory": "person"},
    {"id": 2, "name": "people", "supercategory": "person"},
    {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 4, "name": "car", "supercategory": "vehicle"},
    {"id": 5, "name": "van", "supercategory": "vehicle"},
    {"id": 6, "name": "truck", "supercategory": "vehicle"},
    {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
    {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
    {"id": 9, "name": "bus", "supercategory": "vehicle"},
    {"id": 10, "name": "motor", "supercategory": "vehicle"},
]


def main() -> None:
    args = parse_args()
    convert_visdrone_to_coco(args.image_dir, args.ann_dir, args.out, args.include_ignored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET txt annotations to COCO JSON.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Include VisDrone ignored regions/category 0 as annotations.",
    )
    return parser.parse_args()


def convert_visdrone_to_coco(
    image_dir: str | Path,
    ann_dir: str | Path,
    out: str | Path,
    include_ignored: bool = False,
) -> None:
    image_dir = Path(image_dir)
    ann_dir = Path(ann_dir)
    valid_category_ids = {cat["id"] for cat in VISDRONE_CATEGORIES}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    for image_id, image_path in enumerate(image_files, start=1):
        with Image.open(image_path) as img:
            width, height = img.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        ann_path = ann_dir / f"{image_path.stem}.txt"
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = [float(v) for v in line.split(",")]
                if len(fields) < 8:
                    continue
                x, y, w, h, score, category_id, truncation, occlusion = fields[:8]
                category_id = int(category_id)
                is_ignored = category_id == 0 or float(score) == 0.0
                if category_id == 0:
                    if not include_ignored:
                        continue
                    target_category_ids = sorted(valid_category_ids)
                elif category_id in valid_category_ids:
                    target_category_ids = [category_id]
                else:
                    continue

                w = max(0.0, min(w, width - x))
                h = max(0.0, min(h, height - y))
                if w <= 0 or h <= 0:
                    continue
                for target_category_id in target_category_ids:
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": target_category_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": int(is_ignored),
                            "ignore": int(is_ignored),
                            "visdrone_category_id": category_id,
                            "truncation": int(truncation),
                            "occlusion": int(occlusion),
                        }
                    )
                    ann_id += 1

    write_json(
        out,
        {
            "info": {"description": "VisDrone DET converted to COCO format"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": VISDRONE_CATEGORIES,
        },
    )


if __name__ == "__main__":
    main()
