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


# Coordinate frames a model may actually use, regardless of what the prompt
# requested. ``normalized_1000`` is the 0-1000 convention used by Qwen-VL and
# friends. These are the modes ``infer_coord_mode`` / ``sent_xyxy_to_original_xyxy``
# understand; the CLI ``--coord-mode`` (the requested frame) is a subset.
COORD_MODES = ("sent", "normalized", "normalized_1000", "original")

# How far coordinates may exceed the *requested* frame before we stop trusting
# it. A model that returns a slightly un-clipped box is still treated as obeying
# the prompt; the box is clamped downstream.
COORD_TRUST_TOL = 0.15
# How tightly coordinates must fit a *different* frame before we conclude the
# model ignored the prompt and switch to that frame instead.
COORD_SWITCH_TOL = 0.02
# At/below this max coordinate, output is assumed to be normalized in [0, 1].
NORMALIZED_MAX = 1.5


def frame_bounds(transform: ImageTransform, coord_mode: str) -> tuple[float, float]:
    if coord_mode == "sent":
        return float(transform.sent_width), float(transform.sent_height)
    if coord_mode == "original":
        return float(transform.original_width), float(transform.original_height)
    if coord_mode == "normalized":
        return 1.0, 1.0
    if coord_mode == "normalized_1000":
        return 1000.0, 1000.0
    raise ValueError(f"Unsupported coord mode: {coord_mode}")


def _bbox_max_coords(bboxes: list[list[float]]) -> tuple[float, float] | None:
    max_x = 0.0
    max_y = 0.0
    found = False
    for bbox in bboxes:
        if bbox is None or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        found = True
    return (max_x, max_y) if found else None


def _frame_fits(max_x: float, max_y: float, bound_x: float, bound_y: float, tol: float) -> bool:
    return max_x <= bound_x * (1.0 + tol) and max_y <= bound_y * (1.0 + tol)


def infer_coord_mode(
    bboxes: list[list[float]],
    transform: ImageTransform,
    preferred: str,
    trust_tol: float = COORD_TRUST_TOL,
    switch_tol: float = COORD_SWITCH_TOL,
) -> str:
    """Guess which coordinate frame the model actually used for ``bboxes``.

    The prompt asks for ``preferred`` (the CLI ``--coord-mode``), but models
    frequently ignore it -- e.g. returning original-resolution pixels for a
    downscaled image, or 0-1000 values when [0, 1] was requested. We trust the
    requested frame when the coordinates plausibly fit it, and only override
    when they clearly belong to a different frame, so a single mislabeled box
    no longer derails the whole image.
    """
    coords = _bbox_max_coords(bboxes)
    if coords is None:
        return preferred
    max_x, max_y = coords

    # Clearly fractional output is normalized [0, 1] no matter what was asked.
    if max(max_x, max_y) <= NORMALIZED_MAX:
        return "normalized"

    # Trust the requested frame when the values plausibly fit it. A requested
    # ``normalized`` frame with non-fractional values is really 0-1000 output.
    if preferred == "normalized":
        if _frame_fits(max_x, max_y, *frame_bounds(transform, "normalized_1000"), trust_tol):
            return "normalized_1000"
    elif preferred in ("sent", "original"):
        if _frame_fits(max_x, max_y, *frame_bounds(transform, preferred), trust_tol):
            return preferred

    # The model clearly broke the requested frame: adopt the first other frame
    # the coordinates fit tightly. Order matters for ambiguous overlaps.
    for mode in ("sent", "normalized_1000", "original"):
        if mode == preferred:
            continue
        if _frame_fits(max_x, max_y, *frame_bounds(transform, mode), switch_tol):
            return mode

    # Nothing fits; keep the requested frame and let validation/clamping decide.
    return preferred


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
    elif coord_mode == "normalized_1000":
        x1 = x1 / 1000.0 * transform.sent_width
        x2 = x2 / 1000.0 * transform.sent_width
        y1 = y1 / 1000.0 * transform.sent_height
        y2 = y2 / 1000.0 * transform.sent_height
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
