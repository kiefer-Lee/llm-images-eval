from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coco_io import CocoDataset, write_json
from src.image_ops import ImageTransform, xyxy_to_coco_xywh
from src.parsing import ParsedDetection, parse_model_json
from src.visdrone_paths import resolve_visdrone_val_paths, validate_visdrone_val_paths
from src.visualize import save_detection_visualization, visualization_path


KNOWN_MODES = ("sent", "normalized_1000", "original", "normalized")


@dataclass(frozen=True)
class ParsedRawEntry:
    entry: dict[str, Any]
    detections: list[ParsedDetection]
    parse_source: str


@dataclass(frozen=True)
class FrameInference:
    kind: str
    mode: str | None
    valid_modes: tuple[str, ...]
    max_x: float
    max_y: float
    reason: str


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(source_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}. "
            "Pass a new --output-dir."
        )
    output_dir.mkdir(parents=True)

    visdrone_paths = resolve_visdrone_val_paths(args.visdrone_root)
    validate_visdrone_val_paths(visdrone_paths)
    dataset = CocoDataset(visdrone_paths.ann_file, visdrone_paths.image_root)
    images_by_id = {image.id: image for image in dataset.images}

    raw_entries = load_raw_entries(source_dir / "raw_responses.jsonl")
    parsed_entries = [parse_raw_entry(entry) for entry in raw_entries]
    first_pass = {
        int(parsed.entry["image_id"]): infer_frame(
            parsed.detections,
            transform_from_entry(parsed.entry),
            normalized_1000_tolerance=args.normalized_1000_tolerance,
        )
        for parsed in parsed_entries
    }
    global_mode = choose_global_mode(first_pass.values(), args.global_mode)

    selected_file_names = selected_visualization_file_names(
        Path(args.visualization_source_dir)
        if args.visualization_source_dir
        else source_dir / "visualizations"
    )

    detections: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    raw_recovered_rows: list[dict[str, Any]] = []
    num_visualized = 0
    fallback_parse_count = 0
    mode_counts: Counter[str] = Counter()
    inference_kind_counts: Counter[str] = Counter()

    vis_dir = output_dir / "visualizations"
    for parsed in parsed_entries:
        entry = parsed.entry
        image_id = int(entry["image_id"])
        image = images_by_id.get(image_id)
        if image is None:
            continue

        transform = transform_from_entry(entry)
        inference = first_pass[image_id]
        mode = inference.mode or global_mode
        if mode not in KNOWN_MODES:
            mode = global_mode

        image_detections = convert_detections(
            parsed.detections,
            dataset=dataset,
            image_id=image_id,
            transform=transform,
            coord_mode=mode,
            min_score=args.min_score,
            bottom_edge_y_sent_fix=args.bottom_edge_y_sent_fix,
            bottom_edge_y_tolerance=args.bottom_edge_y_tolerance,
            bottom_edge_min_y_ratio=args.bottom_edge_min_y_ratio,
            bottom_edge_min_height_ratio=args.bottom_edge_min_height_ratio,
        )
        detections.extend(image_detections)
        mode_counts[mode] += 1
        inference_kind_counts[inference.kind] += 1
        if parsed.parse_source != "response_text":
            fallback_parse_count += 1

        row = {
            "image_id": image_id,
            "file_name": entry.get("file_name"),
            "parse_source": parsed.parse_source,
            "inference_kind": inference.kind,
            "inferred_mode": inference.mode,
            "mode_used": mode,
            "valid_modes": list(inference.valid_modes),
            "max_x": inference.max_x,
            "max_y": inference.max_y,
            "reason": inference.reason,
            "num_raw_detections": len(parsed.detections),
            "num_recovered_detections": len(image_detections),
        }
        inference_rows.append(row)
        raw_recovered_rows.append(
            {
                **row,
                "raw_detections": [det.__dict__ for det in parsed.detections],
                "recovered_coco_detections": image_detections,
            }
        )

        if image.file_name in selected_file_names:
            image_path = dataset.resolve_image_path(image)
            save_detection_visualization(
                image_path=image_path,
                detections=image_detections,
                dataset=dataset,
                output_path=visualization_path(vis_dir, image.file_name),
                score_thr=args.vis_score_thr,
            )
            num_visualized += 1

    write_json(output_dir / "detections.coco.json", detections)
    write_json(output_dir / "coordinate_inference.json", inference_rows)
    write_json(output_dir / "raw_recovered_detections.json", raw_recovered_rows)

    summary = {
        "source_dir": str(source_dir),
        "raw_responses": str(source_dir / "raw_responses.jsonl"),
        "output_dir": str(output_dir),
        "num_raw_entries": len(raw_entries),
        "num_images_recovered": len(parsed_entries),
        "num_recovered_detections": len(detections),
        "global_mode_used_for_ambiguous": global_mode,
        "mode_counts_by_image": dict(mode_counts),
        "inference_kind_counts": dict(inference_kind_counts),
        "fallback_parse_count": fallback_parse_count,
        "selected_visualization_source_count": len(selected_file_names),
        "num_visualized": num_visualized,
        "visualizations": str(vis_dir),
        "min_score": args.min_score,
        "vis_score_thr": args.vis_score_thr,
        "bottom_edge_y_sent_fix": args.bottom_edge_y_sent_fix,
        "bottom_edge_y_tolerance": args.bottom_edge_y_tolerance,
        "bottom_edge_min_y_ratio": args.bottom_edge_min_y_ratio,
        "bottom_edge_min_height_ratio": args.bottom_edge_min_height_ratio,
    }
    write_json(output_dir / "recovery_summary.json", summary)

    print(f"Recovered detections: {len(detections)} -> {output_dir / 'detections.coco.json'}")
    print(f"Coordinate inference: {output_dir / 'coordinate_inference.json'}")
    print(f"Recovered raw text result: {output_dir / 'raw_recovered_detections.json'}")
    print(f"Summary: {output_dir / 'recovery_summary.json'}")
    print(f"Visualized images: {num_visualized} -> {vis_dir}")
    print(f"Global mode for ambiguous images: {global_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover Qwen raw detection coordinates by inferring whether the model "
            "returned sent-image pixels, original-image pixels, or 0-1000 coordinates."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "outputs" / "qwen3-vl-30b-a3b-instruct"),
        help="Existing run directory containing raw_responses.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New output directory. Must not already exist.",
    )
    parser.add_argument("--visdrone-root", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--vis-score-thr", type=float, default=0.0)
    parser.add_argument(
        "--visualization-source-dir",
        default=None,
        help=(
            "Directory whose *_pred.jpg names define the selected images to redraw. "
            "Defaults to <source-dir>/visualizations."
        ),
    )
    parser.add_argument(
        "--global-mode",
        choices=KNOWN_MODES,
        default=None,
        help=(
            "Override the mode used for ambiguous images. By default this is chosen "
            "from clear coordinate-frame evidence in raw responses."
        ),
    )
    parser.add_argument(
        "--normalized-1000-tolerance",
        type=float,
        default=0.30,
        help=(
            "Treat 0-1000 coordinates with this relative overshoot as normalized_1000. "
            "Default 0.30 accepts Qwen-style overshoot values up to 1300 before clipping to the image."
        ),
    )
    parser.add_argument(
        "--bottom-edge-y-sent-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When recovering normalized_1000 boxes, treat bottom-edge y values "
            "near the sent-image height as sent-image pixels. This fixes Qwen "
            "responses that mix 0-1000 x coordinates with sent-image y coordinates."
        ),
    )
    parser.add_argument(
        "--bottom-edge-y-tolerance",
        type=float,
        default=2.0,
        help="Pixel tolerance for detecting y2 at the sent-image bottom edge.",
    )
    parser.add_argument(
        "--bottom-edge-min-y-ratio",
        type=float,
        default=0.90,
        help="Minimum raw y1 / sent_height ratio for the bottom-edge y-axis fix.",
    )
    parser.add_argument(
        "--bottom-edge-min-height-ratio",
        type=float,
        default=0.05,
        help="Minimum raw box height / sent_height ratio for the bottom-edge y-axis fix.",
    )
    return parser.parse_args()


def default_output_dir(source_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return source_dir.parent / f"{source_dir.name}-coord-recovered-{stamp}"


def load_raw_entries(raw_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {raw_path}:{line_no}: {exc}") from exc
    return entries


def parse_raw_entry(entry: dict[str, Any]) -> ParsedRawEntry:
    response_text = str(entry.get("response_text") or "")
    try:
        detections, _ = parse_model_json(response_text)
        return ParsedRawEntry(entry=entry, detections=detections, parse_source="response_text")
    except Exception:
        payload = entry.get("parsed_payload")
        detections = parsed_detections_from_payload(payload)
        return ParsedRawEntry(entry=entry, detections=detections, parse_source="parsed_payload")


def parsed_detections_from_payload(payload: Any) -> list[ParsedDetection]:
    detections = payload.get("detections", payload) if isinstance(payload, dict) else payload
    if not isinstance(detections, list):
        return []
    parsed: list[ParsedDetection] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box") or item.get("xyxy")
        if bbox is None and "x1" in item:
            bbox = [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
        if not bbox or len(bbox) != 4:
            continue
        parsed.append(
            ParsedDetection(
                label=item.get("label") or item.get("category") or item.get("name"),
                category_id=int(item["category_id"]) if item.get("category_id") is not None else None,
                bbox_2d=[float(v) for v in bbox],
                score=float(item.get("score", item.get("confidence", 1.0))),
                attributes=item.get("attributes") or {},
            )
        )
    return parsed


def transform_from_entry(entry: dict[str, Any]) -> ImageTransform:
    transform = dict(entry["transform"])
    return ImageTransform(
        original_width=int(transform["original_width"]),
        original_height=int(transform["original_height"]),
        sent_width=int(transform["sent_width"]),
        sent_height=int(transform["sent_height"]),
        scale_x=float(transform["scale_x"]),
        scale_y=float(transform["scale_y"]),
        pad_x=float(transform.get("pad_x", 0.0)),
        pad_y=float(transform.get("pad_y", 0.0)),
        letterbox=bool(transform.get("letterbox", False)),
    )


def infer_frame(
    detections: list[ParsedDetection],
    transform: ImageTransform,
    normalized_1000_tolerance: float,
    eps: float = 1e-6,
) -> FrameInference:
    coords = [value for det in detections for value in det.bbox_2d]
    if not coords:
        return FrameInference(
            kind="empty",
            mode=None,
            valid_modes=(),
            max_x=0.0,
            max_y=0.0,
            reason="No parsed boxes; coordinate frame cannot be inferred for this image.",
        )

    xs = [value for det in detections for value in (det.bbox_2d[0], det.bbox_2d[2])]
    ys = [value for det in detections for value in (det.bbox_2d[1], det.bbox_2d[3])]
    min_coord = min(coords)
    max_x = max(xs)
    max_y = max(ys)

    valid_modes: list[str] = []
    if min_coord >= -eps and max_x <= transform.sent_width + eps and max_y <= transform.sent_height + eps:
        valid_modes.append("sent")
    normalized_1000_limit = 1000.0 * (1.0 + normalized_1000_tolerance)
    if min_coord >= -eps and max_x <= normalized_1000_limit + eps and max_y <= normalized_1000_limit + eps:
        valid_modes.append("normalized_1000")
    if min_coord >= -eps and max_x <= transform.original_width + eps and max_y <= transform.original_height + eps:
        valid_modes.append("original")
    if min_coord >= -eps and max_x <= 1.0 + eps and max_y <= 1.0 + eps:
        valid_modes.append("normalized")

    valid_set = set(valid_modes)
    if "normalized" in valid_set:
        return FrameInference(
            kind="clear",
            mode="normalized",
            valid_modes=tuple(valid_modes),
            max_x=max_x,
            max_y=max_y,
            reason="All coordinates are in [0, 1].",
        )
    if "sent" in valid_set and "normalized_1000" not in valid_set:
        return FrameInference(
            kind="clear",
            mode="sent",
            valid_modes=tuple(valid_modes),
            max_x=max_x,
            max_y=max_y,
            reason="Coordinates fit the sent-image frame but exceed the 0-1000 frame.",
        )
    if "normalized_1000" in valid_set and "sent" not in valid_set:
        return FrameInference(
            kind="clear",
            mode="normalized_1000",
            valid_modes=tuple(valid_modes),
            max_x=max_x,
            max_y=max_y,
            reason=(
                "Coordinates fit the 0-1000 frame within tolerance but exceed "
                "the requested sent-image frame."
            ),
        )
    if "original" in valid_set and "normalized_1000" not in valid_set and "sent" not in valid_set:
        return FrameInference(
            kind="clear",
            mode="original",
            valid_modes=tuple(valid_modes),
            max_x=max_x,
            max_y=max_y,
            reason="Coordinates fit only the original image frame.",
        )
    if "sent" in valid_set and "normalized_1000" in valid_set:
        return FrameInference(
            kind="ambiguous",
            mode=None,
            valid_modes=tuple(valid_modes),
            max_x=max_x,
            max_y=max_y,
            reason=(
                "Coordinates fit both sent-image pixels and 0-1000 coordinates; "
                "the run-level clear evidence decides the mode."
            ),
        )
    return FrameInference(
        kind="unknown",
        mode=None,
        valid_modes=tuple(valid_modes),
        max_x=max_x,
        max_y=max_y,
        reason="Coordinates do not cleanly fit the supported frames.",
    )


def choose_global_mode(inferences: Any, override: str | None) -> str:
    if override:
        return override
    counts: Counter[str] = Counter(
        inference.mode for inference in inferences if inference.kind == "clear" and inference.mode
    )
    if not counts:
        return "sent"
    return counts.most_common(1)[0][0]


def selected_visualization_file_names(vis_dir: Path) -> set[str]:
    if not vis_dir.exists():
        return set()
    selected: set[str] = set()
    for path in vis_dir.glob("*_pred.*"):
        selected.add(f"{path.stem.removesuffix('_pred')}{path.suffix}")
    return selected


def convert_detections(
    detections: list[ParsedDetection],
    dataset: CocoDataset,
    image_id: int,
    transform: ImageTransform,
    coord_mode: str,
    min_score: float,
    bottom_edge_y_sent_fix: bool = True,
    bottom_edge_y_tolerance: float = 2.0,
    bottom_edge_min_y_ratio: float = 0.90,
    bottom_edge_min_height_ratio: float = 0.05,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for det in detections:
        category_id = dataset.resolve_category_id(det.label, det.category_id)
        if category_id is None:
            continue
        score = float(det.score)
        if score < min_score:
            continue
        xyxy = recover_xyxy(
            det.bbox_2d,
            transform,
            coord_mode,
            bottom_edge_y_sent_fix=bottom_edge_y_sent_fix,
            bottom_edge_y_tolerance=bottom_edge_y_tolerance,
            bottom_edge_min_y_ratio=bottom_edge_min_y_ratio,
            bottom_edge_min_height_ratio=bottom_edge_min_height_ratio,
        )
        if xyxy is None:
            continue
        converted.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": xyxy_to_coco_xywh(xyxy),
                "score": score,
            }
        )
    return converted


def recover_xyxy(
    bbox: list[float],
    transform: ImageTransform,
    coord_mode: str,
    bottom_edge_y_sent_fix: bool = True,
    bottom_edge_y_tolerance: float = 2.0,
    bottom_edge_min_y_ratio: float = 0.90,
    bottom_edge_min_height_ratio: float = 0.05,
) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if coord_mode == "normalized_1000":
        use_sent_y_for_bottom_edge = (
            bottom_edge_y_sent_fix
            and transform.sent_height > 0
            and transform.scale_y > 0
            and y2 >= transform.sent_height - bottom_edge_y_tolerance
            and y2 <= transform.sent_height + bottom_edge_y_tolerance
            and y1 >= transform.sent_height * bottom_edge_min_y_ratio
            and (y2 - y1) >= transform.sent_height * bottom_edge_min_height_ratio
        )
        x1 = x1 / 1000.0 * transform.original_width
        x2 = x2 / 1000.0 * transform.original_width
        if use_sent_y_for_bottom_edge:
            y1 = (y1 - transform.pad_y) / transform.scale_y
            y2 = (y2 - transform.pad_y) / transform.scale_y
        else:
            y1 = y1 / 1000.0 * transform.original_height
            y2 = y2 / 1000.0 * transform.original_height
    elif coord_mode == "normalized":
        x1 = x1 * transform.original_width
        x2 = x2 * transform.original_width
        y1 = y1 * transform.original_height
        y2 = y2 * transform.original_height
    elif coord_mode == "sent":
        x1 = (x1 - transform.pad_x) / transform.scale_x
        x2 = (x2 - transform.pad_x) / transform.scale_x
        y1 = (y1 - transform.pad_y) / transform.scale_y
        y2 = (y2 - transform.pad_y) / transform.scale_y
    elif coord_mode == "original":
        pass
    else:
        raise ValueError(f"Unsupported coord mode: {coord_mode}")

    x1 = max(0.0, min(float(transform.original_width), x1))
    x2 = max(0.0, min(float(transform.original_width), x2))
    y1 = max(0.0, min(float(transform.original_height), y1))
    y2 = max(0.0, min(float(transform.original_height), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    main()





