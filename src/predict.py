from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .coco_io import CocoDataset, append_jsonl, write_json
from .image_ops import prepare_image, sent_xyxy_to_original_xyxy, xyxy_to_coco_xywh
from .parsing import ParsedDetection, parse_model_json
from .prompts import build_detection_prompt
from .visualize import save_detection_visualization, visualization_path
from .visdrone_paths import resolve_visdrone_val_paths, validate_visdrone_val_paths
from .vlm_client import OpenAICompatibleVlmClient, VlmConfig


def main() -> None:
    args = parse_args()
    run_prediction(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM object detection on the VisDrone validation set."
    )
    parser.add_argument(
        "--visdrone-root",
        default=None,
        help=(
            "Path to Datasets/VisDrone. Defaults to project-local Datasets/VisDrone, "
            "falling back to ../Datasets/VisDrone."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Defaults to outputs/visdrone_val.",
    )
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
    parser.add_argument(
        "--format-retries",
        type=int,
        default=2,
        help="Retry an image when the response JSON or coordinates violate the prompt.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between images.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop detections below this score.")
    parser.add_argument("--extra-instruction", default=None, help="Extra prompt text.")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated result images after each image is parsed.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Visualization directory. Defaults to <output-dir>/visualizations.",
    )
    parser.add_argument(
        "--vis-score-thr",
        type=float,
        default=0.0,
        help="Only draw detections whose score is at least this value.",
    )
    parser.add_argument(
        "--vis-random-count",
        type=int,
        default=None,
        help=(
            "Randomly save visualizations for this many processed images. "
            "Only works with --save-vis. Default: visualize all processed images."
        ),
    )
    parser.add_argument(
        "--vis-random-seed",
        type=int,
        default=42,
        help="Random seed used by --vis-random-count.",
    )
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress messages. The tqdm progress bar is still shown.",
    )
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> None:
    visdrone_paths = resolve_visdrone_val_paths(args.visdrone_root)
    validate_visdrone_val_paths(visdrone_paths)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    failure_path = output_dir / "failures.jsonl"
    pred_path = output_dir / "detections.coco.json"
    vis_dir = Path(args.vis_dir) if args.vis_dir else output_dir / "visualizations"
    if not args.append_logs:
        for log_path in (raw_path, failure_path):
            if log_path.exists():
                log_path.unlink()

    dataset = CocoDataset(visdrone_paths.ann_file, visdrone_paths.image_root)
    images = dataset.images[args.start_index :]
    if args.max_images is not None:
        images = images[: args.max_images]
    vis_image_ids = _select_visualization_image_ids(
        image_ids=[image.id for image in images],
        save_vis=args.save_vis,
        random_count=args.vis_random_count,
        seed=args.vis_random_seed,
    )
    if not args.quiet:
        tqdm.write(f"Dataset: VisDrone2019-DET-val")
        tqdm.write(f"Images: {len(images)}")
        tqdm.write(f"Annotations: {visdrone_paths.ann_file}")
        tqdm.write(f"Image root: {visdrone_paths.image_root}")
        tqdm.write(f"Output dir: {output_dir}")
        if args.save_vis:
            tqdm.write(
                f"Visualizations: {len(vis_image_ids)} image(s) -> {vis_dir} "
                f"(score_thr={args.vis_score_thr})"
            )

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
    num_failures = 0
    num_visualized = 0
    num_prompt_violation_images = 0
    num_prompt_violation_attempts = 0
    num_model_response_attempts = 0
    progress = tqdm(images, desc="VLM detect", unit="img")
    for idx, image in enumerate(progress, start=1):
        progress.set_description(f"VLM detect {idx}/{len(images)}")
        image_path = dataset.resolve_image_path(image)
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            prepared = prepare_image(
                image_path=image_path,
                max_side=None if args.no_resize else args.image_max_side,
                letterbox=args.letterbox,
            )
            parsed, parsed_payload, response_text, final_prompt, retry_errors, response_attempts = request_valid_response(
                client=client,
                dataset=dataset,
                image=image,
                prepared_transform=prepared.transform,
                image_data_url=prepared.data_url,
                coord_mode=args.coord_mode,
                extra_instruction=args.extra_instruction,
                format_retries=args.format_retries,
                mock_empty=args.mock_empty,
            )
            num_model_response_attempts += response_attempts
            num_prompt_violation_attempts += len(retry_errors)
            if retry_errors:
                num_prompt_violation_images += 1
            image_detections = convert_to_coco_detections(
                parsed=parsed,
                dataset=dataset,
                image_id=image.id,
                transform=prepared.transform,
                coord_mode=args.coord_mode,
                min_score=args.min_score,
            )
            coco_detections.extend(image_detections)
            vis_path = None
            if image.id in vis_image_ids:
                vis_path = visualization_path(vis_dir, image.file_name)
                save_detection_visualization(
                    image_path=image_path,
                    detections=image_detections,
                    dataset=dataset,
                    output_path=vis_path,
                    score_thr=args.vis_score_thr,
                )
                num_visualized += 1
            append_jsonl(
                raw_path,
                {
                    "image_id": image.id,
                    "file_name": image.file_name,
                    "image_path": str(image_path),
                    "transform": prepared.transform.to_dict(),
                    "prompt": final_prompt,
                    "response_text": response_text,
                    "response_attempts": response_attempts,
                    "prompt_violation": bool(retry_errors),
                    "format_retry_errors": retry_errors,
                    "parsed_payload": parsed_payload,
                    "coco_detections": image_detections,
                    "visualization_path": str(vis_path) if vis_path else None,
                },
            )
        except Exception as exc:  # noqa: BLE001 - keep batch inference running.
            num_failures += 1
            failure_payload: dict[str, Any] = {
                "image_id": image.id,
                "file_name": image.file_name,
                "image_path": str(image_path),
                "error": repr(exc),
            }
            if isinstance(exc, PromptValidationError):
                num_model_response_attempts += exc.response_attempts
                num_prompt_violation_attempts += len(exc.retry_errors)
                num_prompt_violation_images += 1
                failure_payload.update(
                    {
                        "response_attempts": exc.response_attempts,
                        "prompt_violation": True,
                        "format_retry_errors": exc.retry_errors,
                        "last_response_text": exc.response_text,
                    }
                )
            append_jsonl(
                failure_path,
                failure_payload,
            )
            if not args.quiet:
                progress.write(f"[WARN] Failed image_id={image.id} file={image.file_name}: {exc!r}")
        if args.sleep > 0:
            time.sleep(args.sleep)
        progress.set_postfix(
            dets=len(coco_detections),
            fail=num_failures,
            vis=num_visualized,
            viol=num_prompt_violation_attempts,
            refresh=True,
        )
    progress.close()

    prompt_violation_stats = _build_prompt_violation_stats(
        num_images=len(images),
        num_model_response_attempts=num_model_response_attempts,
        num_prompt_violation_attempts=num_prompt_violation_attempts,
        num_prompt_violation_images=num_prompt_violation_images,
        num_failures=num_failures,
    )
    write_json(pred_path, coco_detections)
    write_json(output_dir / "prompt_violation_stats.json", prompt_violation_stats)
    write_json(
        output_dir / "run_config.json",
        {
            "dataset": "VisDrone2019-DET-val",
            "visdrone_root": str(visdrone_paths.root),
            "ann_file": str(visdrone_paths.ann_file),
            "image_root": str(visdrone_paths.image_root),
            "num_images": len(images),
            "num_detections": len(coco_detections),
            "coord_mode": args.coord_mode,
            "image_max_side": None if args.no_resize else args.image_max_side,
            "letterbox": args.letterbox,
            "mock_empty": args.mock_empty,
            "append_logs": args.append_logs,
            "num_failures": num_failures,
            "format_retries": args.format_retries,
            "prompt_violation_stats": prompt_violation_stats,
            "save_vis": args.save_vis,
            "vis_dir": str(vis_dir) if args.save_vis else None,
            "vis_score_thr": args.vis_score_thr,
            "vis_random_count": args.vis_random_count,
            "vis_random_seed": args.vis_random_seed,
            "num_visualized_images": len(vis_image_ids),
        },
    )
    if not args.quiet:
        tqdm.write("Done.")
        tqdm.write(f"Detections: {len(coco_detections)} -> {pred_path}")
        tqdm.write(f"Raw responses: {raw_path}")
        if num_failures:
            tqdm.write(f"Failures: {num_failures} -> {failure_path}")
        if args.save_vis:
            tqdm.write(f"Visualized images: {num_visualized} -> {vis_dir}")
        tqdm.write(
            "Prompt violations: "
            f"{prompt_violation_stats['violation_attempt_ratio']:.2%} by response attempt, "
            f"{prompt_violation_stats['violation_image_ratio']:.2%} by image "
            f"-> {output_dir / 'prompt_violation_stats.json'}"
        )


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "visdrone_val"


def _select_visualization_image_ids(
    image_ids: list[int],
    save_vis: bool,
    random_count: int | None,
    seed: int,
) -> set[int]:
    if not save_vis:
        return set()
    if random_count is None:
        return set(image_ids)
    if random_count < 0:
        raise ValueError("--vis-random-count must be >= 0.")
    if random_count >= len(image_ids):
        return set(image_ids)
    rng = random.Random(seed)
    return set(rng.sample(image_ids, random_count))


def request_valid_response(
    client: OpenAICompatibleVlmClient | None,
    dataset: CocoDataset,
    image: Any,
    prepared_transform: Any,
    image_data_url: str,
    coord_mode: str,
    extra_instruction: str | None,
    format_retries: int,
    mock_empty: bool,
) -> tuple[list[ParsedDetection], Any, str, str, list[str], int]:
    if format_retries < 0:
        raise ValueError("--format-retries must be >= 0.")

    retry_errors: list[str] = []
    response_text = ""
    final_prompt = ""
    parsed_payload: Any = None
    parsed: list[ParsedDetection] = []
    response_attempts = 0
    for attempt in range(format_retries + 1):
        retry_instruction = _build_retry_instruction(retry_errors[-1]) if retry_errors else None
        merged_extra = _merge_extra_instruction(extra_instruction, retry_instruction)
        final_prompt = build_detection_prompt(
            dataset=dataset,
            image=image,
            transform=prepared_transform,
            coord_mode=coord_mode,
            extra_instruction=merged_extra,
        )
        response_text = '{"detections": []}' if mock_empty else client.detect(final_prompt, image_data_url)  # type: ignore[union-attr]
        response_attempts += 1
        try:
            parsed, parsed_payload = parse_model_json(response_text)
            validate_model_detections(parsed, dataset, prepared_transform, coord_mode)
            return parsed, parsed_payload, response_text, final_prompt, retry_errors, response_attempts
        except Exception as exc:  # noqa: BLE001 - validation errors should trigger semantic retry.
            retry_errors.append(str(exc))
            if attempt >= format_retries:
                raise PromptValidationError(
                    message=f"Invalid model response after {response_attempts} attempt(s): {exc}",
                    retry_errors=retry_errors,
                    response_attempts=response_attempts,
                    response_text=response_text,
                    prompt=final_prompt,
                ) from exc
    return parsed, parsed_payload, response_text, final_prompt, retry_errors, response_attempts


class PromptValidationError(ValueError):
    def __init__(
        self,
        message: str,
        retry_errors: list[str],
        response_attempts: int,
        response_text: str,
        prompt: str,
    ) -> None:
        super().__init__(message)
        self.retry_errors = retry_errors
        self.response_attempts = response_attempts
        self.response_text = response_text
        self.prompt = prompt


def _build_prompt_violation_stats(
    num_images: int,
    num_model_response_attempts: int,
    num_prompt_violation_attempts: int,
    num_prompt_violation_images: int,
    num_failures: int,
) -> dict[str, float | int]:
    return {
        "num_images": num_images,
        "num_model_response_attempts": num_model_response_attempts,
        "num_prompt_violation_attempts": num_prompt_violation_attempts,
        "num_prompt_violation_images": num_prompt_violation_images,
        "num_failures": num_failures,
        "violation_attempt_ratio": _safe_ratio(
            num_prompt_violation_attempts,
            num_model_response_attempts,
        ),
        "violation_image_ratio": _safe_ratio(
            num_prompt_violation_images,
            num_images,
        ),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_model_detections(
    parsed: list[ParsedDetection],
    dataset: CocoDataset,
    transform: Any,
    coord_mode: str,
) -> None:
    for idx, det in enumerate(parsed):
        resolved_category_id = dataset.resolve_category_id(det.label, det.category_id)
        if det.category_id is not None and det.category_id not in dataset.category_by_id:
            raise ValueError(f"detections[{idx}].category_id is unknown: {det.category_id}")
        if det.label is None and det.category_id is None:
            raise ValueError(f"detections[{idx}] must include label or category_id")
        if resolved_category_id is None:
            raise ValueError(
                f"detections[{idx}] category could not be matched: "
                f"label={det.label!r}, category_id={det.category_id!r}"
            )
        if len(det.bbox_2d) != 4:
            raise ValueError(f"detections[{idx}].bbox_2d must have 4 numbers")
        x1, y1, x2, y2 = [float(v) for v in det.bbox_2d]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        _validate_bbox_range(idx, [x1, y1, x2, y2], transform, coord_mode)


def _validate_bbox_range(
    idx: int,
    bbox: list[float],
    transform: Any,
    coord_mode: str,
) -> None:
    x1, y1, x2, y2 = bbox
    if coord_mode == "sent":
        max_x = float(transform.sent_width)
        max_y = float(transform.sent_height)
        mode_desc = f"sent image pixels [0,{transform.sent_width}] x [0,{transform.sent_height}]"
    elif coord_mode == "normalized":
        max_x = 1.0
        max_y = 1.0
        mode_desc = "normalized coordinates [0,1]"
    elif coord_mode == "original":
        max_x = float(transform.original_width)
        max_y = float(transform.original_height)
        mode_desc = f"original image pixels [0,{transform.original_width}] x [0,{transform.original_height}]"
    else:
        raise ValueError(f"Unsupported coord mode: {coord_mode}")

    values = [x1, y1, x2, y2]
    if any(v < 0 for v in values) or x1 > max_x or x2 > max_x or y1 > max_y or y2 > max_y:
        raise ValueError(
            f"detections[{idx}].bbox_2d={bbox} is outside {mode_desc}. "
            "This violates the coordinate frame requested in the prompt."
        )


def _merge_extra_instruction(
    extra_instruction: str | None,
    retry_instruction: str | None,
) -> str | None:
    parts = [part for part in [extra_instruction, retry_instruction] if part]
    return "\n".join(parts) if parts else None


def _build_retry_instruction(error: str) -> str:
    return (
        "Your previous response was rejected by the validator. "
        f"Reason: {error}\n"
        "Retry the SAME detection task. Do not explain. Return only valid JSON. "
        "Every bbox_2d must strictly follow the coordinate frame requested above."
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
