from __future__ import annotations

from .coco_io import CocoDataset, CocoImage
from .image_ops import ImageTransform


def build_detection_prompt(
    dataset: CocoDataset,
    image: CocoImage,
    transform: ImageTransform,
    coord_mode: str,
    extra_instruction: str | None = None,
) -> str:
    if coord_mode == "normalized":
        coord_text = (
            "Return bbox_2d as normalized [x1, y1, x2, y2] values in [0, 1], "
            "relative to the image shown to you."
        )
    elif coord_mode == "original":
        coord_text = (
            f"Return bbox_2d as [x1, y1, x2, y2] pixel coordinates in the "
            f"original dataset image coordinate frame: width={image.width}, "
            f"height={image.height}, with origin at the top-left corner."
        )
    else:
        coord_text = (
            f"Return bbox_2d as [x1, y1, x2, y2] pixel coordinates in the "
            f"image shown to you. The shown image coordinate frame is "
            f"width={transform.sent_width}, height={transform.sent_height}, "
            "with origin at the top-left corner."
        )

    extra = f"\nAdditional instruction: {extra_instruction.strip()}" if extra_instruction else ""
    return f"""You are an object detection model.

Detect every visible object that belongs to one of the allowed COCO categories below.
Do not detect categories outside this list. Use category_id exactly as listed.

Allowed categories:
{dataset.category_prompt()}

Original dataset image: file_name={image.file_name}, width={image.width}, height={image.height}.
Image sent to you: width={transform.sent_width}, height={transform.sent_height}.
{coord_text}
Important: follow the coordinate frame requested above exactly.
If an object is partly outside the image, clip the box to the visible region.
If no allowed object is visible, return an empty detections list.
Use confidence score in [0, 1]. If uncertain, use a lower score rather than omitting the object.
{extra}

Return only valid JSON, with this exact top-level shape:
{{
  "detections": [
    {{
      "label": "category name",
      "category_id": 1,
      "bbox_2d": [x1, y1, x2, y2],
      "score": 0.0,
      "attributes": {{}}
    }}
  ]
}}
"""
