from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .coco_io import CocoDataset, append_jsonl, write_json
from .image_ops import prepare_image, sent_xyxy_to_original_xyxy, xyxy_to_coco_xywh
from .parsing import ParsedDetection, parse_model_json
from .prompts import build_detection_prompt
from .vlm_client import OpenAICompatibleVlmClient, VlmConfig


def main() -> None:
    args = parse_args()
    run_prediction(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM object detection on a COCO-style validation set."
    )
    parser.add_argument("--ann-file", required=True, help="COCO annotation JSON.")
    parser.add_argument("--image-root", required=True, help="Directory containing validation images.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to VLM_MODEL.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to VLM_API_KEY/OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to VLM_BASE_URL.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images for smoke tests.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the COCO images list.")
    parser.add_argument("--image-max-side", type=int, default=1280, help="Resize longest side before sending.")
    parser.add_argument("--no-resize", action="store_true", help="Send original image dimensions.")
    parser.add_argument("--letterbox", action="store_true", help="Letterbox to a square max-side image.")
    parser.add_argument(
        "--coord-mode",
        choices=["sent", "normalized", "original"],
        default="sent",
        help="Coordinate frame expected from the model.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--json-mode", choices=["none", "json_object"], default="none")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between images.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop detections below this score.")
    parser.add_argument("--extra-instruction", default=None, help="Extra prompt text.")
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Append raw/failure JSONL logs instead of replacing them at run start.",
    )
    parser.add_argument(
        "--mock-empty",
        action="store_true",
        help="Do not call API; produce empty detections for pipeline testing.",
    )
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    failure_path = output_dir / "failures.jsonl"
    pred_path = output_dir / "detections.coco.json"
    if not args.append_logs:
        for log_path in (raw_path, failure_path):
            if log_path.exists():
                log_path.unlink()

    dataset = CocoDataset(args.ann_file, args.image_root)
    images = dataset.images[args.start_index :]
    if args.max_images is not None:
        images = images[: args.max_images]

    client: OpenAICompatibleVlmClient | None = None
    if not args.mock_empty:
        config = VlmConfig.from_env(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=args.json_mode,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        client = OpenAICompatibleVlmClient(config)

    coco_detections: list[dict[str, Any]] = []
    for image in tqdm(images, desc="VLM detect"):
        image_path = dataset.resolve_image_path(image)
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            prepared = prepare_image(
                image_path=image_path,
                max_side=None if args.no_resize else args.image_max_side,
                letterbox=args.letterbox,
            )
            prompt = build_detection_prompt(
                dataset=dataset,
                image=image,
                transform=prepared.transform,
                coord_mode=args.coord_mode,
                extra_instruction=args.extra_instruction,
            )
            response_text = '{"detections": []}' if args.mock_empty else client.detect(prompt, prepared.data_url)  # type: ignore[union-attr]
            parsed, parsed_payload = parse_model_json(response_text)
            image_detections = convert_to_coco_detections(
                parsed=parsed,
                dataset=dataset,
                image_id=image.id,
                transform=prepared.transform,
                coord_mode=args.coord_mode,
                min_score=args.min_score,
            )
            coco_detections.extend(image_detections)
            append_jsonl(
                raw_path,
                {
                    "image_id": image.id,
                    "file_name": image.file_name,
                    "image_path": str(image_path),
                    "transform": prepared.transform.to_dict(),
                    "prompt": prompt,
                    "response_text": response_text,
                    "parsed_payload": parsed_payload,
                    "coco_detections": image_detections,
                },
            )
        except Exception as exc:  # noqa: BLE001 - keep batch inference running.
            append_jsonl(
                failure_path,
                {
                    "image_id": image.id,
                    "file_name": image.file_name,
                    "image_path": str(image_path),
                    "error": repr(exc),
                },
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_json(pred_path, coco_detections)
    write_json(
        output_dir / "run_config.json",
        {
            "ann_file": str(args.ann_file),
            "image_root": str(args.image_root),
            "num_images": len(images),
            "num_detections": len(coco_detections),
            "coord_mode": args.coord_mode,
            "image_max_side": None if args.no_resize else args.image_max_side,
            "letterbox": args.letterbox,
            "mock_empty": args.mock_empty,
            "append_logs": args.append_logs,
        },
    )


def convert_to_coco_detections(
    parsed: list[ParsedDetection],
    dataset: CocoDataset,
    image_id: int,
    transform: Any,
    coord_mode: str,
    min_score: float,
) -> list[dict[str, Any]]:
    coco_detections: list[dict[str, Any]] = []
    for det in parsed:
        score = max(0.0, min(1.0, float(det.score)))
        if score < min_score:
            continue
        category_id = dataset.resolve_category_id(det.label, det.category_id)
        if category_id is None:
            continue
        original_xyxy = sent_xyxy_to_original_xyxy(det.bbox_2d, transform, coord_mode)
        if original_xyxy is None:
            continue
        coco_detections.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": xyxy_to_coco_xywh(original_xyxy),
                "score": score,
            }
        )
    return coco_detections


if __name__ == "__main__":
    main()
