from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .coco_io import CocoDataset, append_jsonl, write_json
from .image_ops import (
    COORD_TRUST_TOL,
    frame_bounds,
    infer_coord_mode,
    prepare_image,
    sent_xyxy_to_original_xyxy,
    xyxy_to_coco_xywh,
)
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
    parser.add_argument(
        "--coord-tolerance",
        type=float,
        default=COORD_TRUST_TOL,
        help=(
            "Relative slack when matching model coordinates to a frame. Boxes "
            "within this margin of the requested frame are trusted (and clamped); "
            "clearly out-of-frame boxes are auto-detected as another frame, and "
            "only coordinates that fit no frame trigger a format retry."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap on response tokens. Omit (default) to send no cap so reasoning models have full budget.",
    )
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

    resume_state = _load_append_log_resume_state(raw_path, pred_path) if args.append_logs else None
    existing_detection_count = len(resume_state.detections) if resume_state else 0
    skipped_image_count = 0
    if resume_state:
        original_image_count = len(images)
        images = [image for image in images if image.id not in resume_state.completed_image_ids]
        skipped_image_count = original_image_count - len(images)

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
        if resume_state:
            tqdm.write(
                "Resume: "
                f"{existing_detection_count} existing detection(s), "
                f"{skipped_image_count} completed image(s) skipped"
            )
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

    coco_detections: list[dict[str, Any]] = list(resume_state.detections) if resume_state else []
    num_failures = 0
    num_visualized = 0
    num_prompt_violation_images = 0
    num_prompt_violation_attempts = 0
    num_model_response_attempts = 0
    num_frame_mismatch_images = 0
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
            parsed, parsed_payload, response_text, final_prompt, retry_errors, response_attempts, effective_coord_mode = request_valid_response(
                client=client,
                dataset=dataset,
                image=image,
                prepared_transform=prepared.transform,
                image_data_url=prepared.data_url,
                coord_mode=args.coord_mode,
                coord_tolerance=args.coord_tolerance,
                extra_instruction=args.extra_instruction,
                format_retries=args.format_retries,
                mock_empty=args.mock_empty,
            )
            num_model_response_attempts += response_attempts
            num_prompt_violation_attempts += len(retry_errors)
            if retry_errors:
                num_prompt_violation_images += 1
            frame_mismatch = effective_coord_mode != args.coord_mode
            if frame_mismatch:
                num_frame_mismatch_images += 1
            image_detections = convert_to_coco_detections(
                parsed=parsed,
                dataset=dataset,
                image_id=image.id,
                transform=prepared.transform,
                coord_mode=effective_coord_mode,
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
                    "requested_coord_mode": args.coord_mode,
                    "effective_coord_mode": effective_coord_mode,
                    "frame_mismatch": frame_mismatch,
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
            frame=num_frame_mismatch_images,
            refresh=True,
        )
    progress.close()

    prompt_violation_stats = _build_prompt_violation_stats(
        num_images=len(images),
        num_model_response_attempts=num_model_response_attempts,
        num_prompt_violation_attempts=num_prompt_violation_attempts,
        num_prompt_violation_images=num_prompt_violation_images,
        num_frame_mismatch_images=num_frame_mismatch_images,
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
            "num_existing_detections": existing_detection_count,
            "num_resume_skipped_images": skipped_image_count,
            "coord_mode": args.coord_mode,
            "coord_tolerance": args.coord_tolerance,
            "num_frame_mismatch_images": num_frame_mismatch_images,
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
        tqdm.write(
            "Frame mismatches (model ignored requested coord frame, auto-corrected): "
            f"{prompt_violation_stats['frame_mismatch_image_ratio']:.2%} by image "
            f"({num_frame_mismatch_images}/{len(images)})"
        )


@dataclass(frozen=True)
class ResumeState:
    detections: list[dict[str, Any]]
    completed_image_ids: set[int]


def _load_append_log_resume_state(raw_path: Path, pred_path: Path) -> ResumeState:
    raw_detections_by_image = _load_detections_from_raw_log(raw_path)
    pred_detections_by_image = _load_detections_from_prediction_file(pred_path)

    detections: list[dict[str, Any]] = []
    for image_detections in raw_detections_by_image.values():
        detections.extend(image_detections)
    for image_id, image_detections in pred_detections_by_image.items():
        if image_id not in raw_detections_by_image:
            detections.extend(image_detections)

    return ResumeState(
        detections=detections,
        completed_image_ids=set(raw_detections_by_image) | set(pred_detections_by_image),
    )


def _load_detections_from_raw_log(path: Path) -> dict[int, list[dict[str, Any]]]:
    detections_by_image: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return detections_by_image

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            image_id = _coerce_int(payload.get("image_id"))
            image_detections = payload.get("coco_detections")
            if image_id is None or not isinstance(image_detections, list):
                continue
            detections_by_image[image_id] = [
                det for det in image_detections if isinstance(det, dict)
            ]
    return detections_by_image


def _load_detections_from_prediction_file(path: Path) -> dict[int, list[dict[str, Any]]]:
    detections_by_image: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return detections_by_image

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Existing prediction file must be a JSON list: {path}")

    for det in payload:
        if not isinstance(det, dict):
            continue
        image_id = _coerce_int(det.get("image_id"))
        if image_id is None:
            continue
        detections_by_image.setdefault(image_id, []).append(det)
    return detections_by_image


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    coord_tolerance: float,
    extra_instruction: str | None,
    format_retries: int,
    mock_empty: bool,
) -> tuple[list[ParsedDetection], Any, str, str, list[str], int, str]:
    if format_retries < 0:
        raise ValueError("--format-retries must be >= 0.")
    if coord_tolerance < 0:
        raise ValueError("--coord-tolerance must be >= 0.")

    retry_errors: list[str] = []
    response_text = ""
    final_prompt = ""
    parsed_payload: Any = None
    parsed: list[ParsedDetection] = []
    response_attempts = 0
    effective_coord_mode = coord_mode
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
            effective_coord_mode = infer_coord_mode(
                [det.bbox_2d for det in parsed],
                prepared_transform,
                coord_mode,
                trust_tol=coord_tolerance,
            )
            validate_model_detections(
                parsed, dataset, prepared_transform, effective_coord_mode, coord_tolerance
            )
            return (
                parsed,
                parsed_payload,
                response_text,
                final_prompt,
                retry_errors,
                response_attempts,
                effective_coord_mode,
            )
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
    return (
        parsed,
        parsed_payload,
        response_text,
        final_prompt,
        retry_errors,
        response_attempts,
        effective_coord_mode,
    )


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
    num_frame_mismatch_images: int,
    num_failures: int,
) -> dict[str, float | int]:
    return {
        "num_images": num_images,
        "num_model_response_attempts": num_model_response_attempts,
        "num_prompt_violation_attempts": num_prompt_violation_attempts,
        "num_prompt_violation_images": num_prompt_violation_images,
        "num_frame_mismatch_images": num_frame_mismatch_images,
        "num_failures": num_failures,
        "violation_attempt_ratio": _safe_ratio(
            num_prompt_violation_attempts,
            num_model_response_attempts,
        ),
        "violation_image_ratio": _safe_ratio(
            num_prompt_violation_images,
            num_images,
        ),
        "frame_mismatch_image_ratio": _safe_ratio(
            num_frame_mismatch_images,
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
    coord_tolerance: float,
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
        _validate_bbox_range(idx, [x1, y1, x2, y2], transform, coord_mode, coord_tolerance)


def _validate_bbox_range(
    idx: int,
    bbox: list[float],
    transform: Any,
    coord_mode: str,
    coord_tolerance: float,
) -> None:
    x1, y1, x2, y2 = bbox
    max_x, max_y = frame_bounds(transform, coord_mode)
    mode_desc = {
        "sent": f"sent image pixels [0,{transform.sent_width}] x [0,{transform.sent_height}]",
        "original": f"original image pixels [0,{transform.original_width}] x [0,{transform.original_height}]",
        "normalized": "normalized coordinates [0,1]",
        "normalized_1000": "normalized coordinates [0,1000]",
    }.get(coord_mode, coord_mode)

    # Coordinates within tolerance of the frame are accepted; reprojection
    # clamps the residual overflow (objects partly outside the image are
    # legitimately clipped). Only coordinates that fit no frame land here, so a
    # gross mismatch -- a wrong scale we could not auto-detect -- still retries.
    margin_x = max_x * coord_tolerance
    margin_y = max_y * coord_tolerance
    values = [x1, y1, x2, y2]
    if (
        min(values) < -max(margin_x, margin_y)
        or x1 > max_x + margin_x
        or x2 > max_x + margin_x
        or y1 > max_y + margin_y
        or y2 > max_y + margin_y
    ):
        raise ValueError(
            f"detections[{idx}].bbox_2d={bbox} does not fit any coordinate frame "
            f"(closest match {mode_desc}, tolerance {coord_tolerance:.0%}). "
            "The model ignored the requested coordinate frame."
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
