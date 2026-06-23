from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDetection:
    label: str | None
    category_id: int | None
    bbox_2d: list[float]
    score: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


def parse_model_json(text: str) -> tuple[list[ParsedDetection], Any]:
    payload = _load_json_object(text)
    detections = payload.get("detections", payload) if isinstance(payload, dict) else payload
    if not isinstance(detections, list):
        raise ValueError("Model response JSON must be a list or contain a detections list.")

    parsed: list[ParsedDetection] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        bbox = (
            item.get("bbox_2d")
            or item.get("bbox")
            or item.get("box")
            or item.get("xyxy")
        )
        if bbox is None and "x1" in item:
            bbox = [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
        if bbox is None or len(bbox) != 4:
            continue

        bbox_values = [float(v) for v in bbox]
        if item.get("bbox_format") == "xywh" or item.get("format") == "xywh":
            x, y, w, h = bbox_values
            bbox_values = [x, y, x + w, y + h]

        score = item.get("score", item.get("confidence", 1.0))
        category_id = item.get("category_id")
        parsed.append(
            ParsedDetection(
                label=item.get("label") or item.get("category") or item.get("name"),
                category_id=int(category_id) if category_id is not None else None,
                bbox_2d=bbox_values,
                score=float(score),
                attributes=item.get("attributes") or {},
            )
        )
    return parsed, payload


def _load_json_object(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            parsed = _decode_first_json(inner)
            if parsed is not None:
                return parsed

    # Find the first JSON object/array and decode just it, ignoring any
    # trailing prose or extra values the model appended (json's "Extra data").
    parsed = _decode_first_json(text)
    if parsed is not None:
        return parsed
    raise ValueError("No JSON object or array found in model response.")


def _decode_first_json(text: str) -> Any | None:
    """Decode the first complete JSON object/array, ignoring trailing data."""
    decoder = json.JSONDecoder()
    for idx in range(len(text)):
        if text[idx] not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
            return obj
        except json.JSONDecodeError:
            continue
    return None
