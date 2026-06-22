from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageTransform:
    original_width: int
    original_height: int
    sent_width: int
    sent_height: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    letterbox: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedImage:
    data_url: str
    transform: ImageTransform
    mime_type: str


def prepare_image(
    image_path: str | Path,
    max_side: int | None = 1280,
    letterbox: bool = False,
    image_format: str = "JPEG",
    quality: int = 90,
) -> PreparedImage:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size

    if max_side is None or max(original_width, original_height) <= max_side:
        sent = image.copy()
        scale_x = 1.0
        scale_y = 1.0
        pad_x = 0.0
        pad_y = 0.0
    elif letterbox:
        scale = max_side / max(original_width, original_height)
        resized_w = max(1, round(original_width * scale))
        resized_h = max(1, round(original_height * scale))
        resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        sent = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        pad_x = float((max_side - resized_w) // 2)
        pad_y = float((max_side - resized_h) // 2)
        sent.paste(resized, (int(pad_x), int(pad_y)))
        scale_x = resized_w / original_width
        scale_y = resized_h / original_height
    else:
        scale = max_side / max(original_width, original_height)
        resized_w = max(1, round(original_width * scale))
        resized_h = max(1, round(original_height * scale))
        sent = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        pad_x = 0.0
        pad_y = 0.0
        scale_x = resized_w / original_width
        scale_y = resized_h / original_height

    sent_width, sent_height = sent.size
    transform = ImageTransform(
        original_width=original_width,
        original_height=original_height,
        sent_width=sent_width,
        sent_height=sent_height,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_x=pad_x,
        pad_y=pad_y,
        letterbox=letterbox,
    )

    buffer = BytesIO()
    fmt = image_format.upper()
    save_kwargs = {"quality": quality} if fmt in {"JPEG", "JPG", "WEBP"} else {}
    sent.save(buffer, format=fmt, **save_kwargs)
    mime = "image/jpeg" if fmt in {"JPEG", "JPG"} else f"image/{fmt.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return PreparedImage(
        data_url=f"data:{mime};base64,{encoded}",
        transform=transform,
        mime_type=mime,
    )


def sent_xyxy_to_original_xyxy(
    bbox: list[float],
    transform: ImageTransform,
    coord_mode: str,
) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]

    if coord_mode == "normalized":
        x1 *= transform.sent_width
        x2 *= transform.sent_width
        y1 *= transform.sent_height
        y2 *= transform.sent_height
    elif coord_mode == "original":
        pass
    elif coord_mode != "sent":
        raise ValueError(f"Unsupported coord mode: {coord_mode}")

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if coord_mode != "original":
        x1 = (x1 - transform.pad_x) / transform.scale_x
        x2 = (x2 - transform.pad_x) / transform.scale_x
        y1 = (y1 - transform.pad_y) / transform.scale_y
        y2 = (y2 - transform.pad_y) / transform.scale_y

    x1 = max(0.0, min(float(transform.original_width), x1))
    x2 = max(0.0, min(float(transform.original_width), x2))
    y1 = max(0.0, min(float(transform.original_height), y1))
    y2 = max(0.0, min(float(transform.original_height), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def xyxy_to_coco_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
