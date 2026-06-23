from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .coco_io import CocoDataset


def save_detection_visualization(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    dataset: CocoDataset,
    output_path: str | Path,
    score_thr: float = 0.0,
) -> None:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()

    for det in detections:
        score = float(det.get("score", 0.0))
        if score < score_thr:
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue

        category_id = int(det["category_id"])
        category = dataset.category_by_id.get(category_id)
        label = category.name if category else str(category_id)
        color = _category_color(category_id)
        line_width = max(2, round(min(image.size) / 360))
        xyxy = [x, y, x + w, y + h]

        for offset in range(line_width):
            draw.rectangle(
                [xyxy[0] - offset, xyxy[1] - offset, xyxy[2] + offset, xyxy[3] + offset],
                outline=color,
            )

        caption = f"{label} {score:.2f}"
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        text_x = max(0, min(x, image.width - text_w - 4))
        text_y = max(0, y - text_h - 6)
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 4, text_y + text_h + 4],
            fill=color,
        )
        draw.text((text_x + 2, text_y + 2), caption, fill=(255, 255, 255), font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def visualization_path(vis_dir: str | Path, file_name: str) -> Path:
    source = Path(file_name)
    suffix = source.suffix if source.suffix else ".jpg"
    return Path(vis_dir) / f"{source.stem}_pred{suffix}"


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def _category_color(category_id: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (69, 123, 157),
        (131, 56, 236),
        (255, 183, 3),
        (2, 48, 71),
        (0, 150, 199),
        (214, 40, 40),
    ]
    return palette[(category_id - 1) % len(palette)]
