from __future__ import annotations

import json
from argparse import Namespace

import pytest
from PIL import Image

import src.predict as predict_module
from src.image_ops import (
    ImageTransform,
    infer_coord_mode,
    sent_xyxy_to_original_xyxy,
)
from src.predict import _load_append_log_resume_state, _validate_bbox_range, run_prediction
from src.tools.visdrone_to_coco import VISDRONE_CATEGORIES, convert_visdrone_to_coco


def _downscale_transform() -> ImageTransform:
    # 2000x1500 original sent at longest-side 1280 -> 1280x960, scale 0.64.
    return ImageTransform(
        original_width=2000,
        original_height=1500,
        sent_width=1280,
        sent_height=960,
        scale_x=0.64,
        scale_y=0.64,
        pad_x=0.0,
        pad_y=0.0,
        letterbox=False,
    )


def test_append_log_resume_restores_detections_and_completed_zero_detection_images(tmp_path):
    raw_path = tmp_path / "raw_responses.jsonl"
    pred_path = tmp_path / "detections.coco.json"

    raw_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": 1,
                        "coco_detections": [
                            {
                                "image_id": 1,
                                "category_id": 4,
                                "bbox": [1, 2, 3, 4],
                                "score": 0.9,
                            }
                        ],
                    }
                ),
                json.dumps({"image_id": 2, "coco_detections": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 5, "bbox": [0, 0, 1, 1], "score": 0.1},
                {"image_id": 3, "category_id": 6, "bbox": [5, 6, 7, 8], "score": 0.8},
            ]
        ),
        encoding="utf-8",
    )

    state = _load_append_log_resume_state(raw_path, pred_path)

    assert state.completed_image_ids == {1, 2, 3}
    assert state.detections == [
        {"image_id": 1, "category_id": 4, "bbox": [1, 2, 3, 4], "score": 0.9},
        {"image_id": 3, "category_id": 6, "bbox": [5, 6, 7, 8], "score": 0.8},
    ]


def test_visdrone_ignored_region_expands_to_crowd_annotations_for_all_categories(tmp_path):
    image_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    out_path = tmp_path / "val.json"
    image_dir.mkdir()
    ann_dir.mkdir()

    Image.new("RGB", (100, 80), color="white").save(image_dir / "000001.jpg")
    (ann_dir / "000001.txt").write_text(
        "\n".join(
            [
                "10,20,30,40,1,0,0,0",
                "1,2,3,4,1,4,0,0",
            ]
        ),
        encoding="utf-8",
    )

    convert_visdrone_to_coco(image_dir, ann_dir, out_path, include_ignored=True)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    ignored = [ann for ann in data["annotations"] if ann["visdrone_category_id"] == 0]
    normal = [ann for ann in data["annotations"] if ann["visdrone_category_id"] == 4]

    assert len(ignored) == len(VISDRONE_CATEGORIES)
    assert {ann["category_id"] for ann in ignored} == {
        cat["id"] for cat in VISDRONE_CATEGORIES
    }
    assert all(ann["iscrowd"] == 1 and ann["ignore"] == 1 for ann in ignored)
    assert normal == [
        {
            "id": len(VISDRONE_CATEGORIES) + 1,
            "image_id": 1,
            "category_id": 4,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "area": 12.0,
            "iscrowd": 0,
            "ignore": 0,
            "visdrone_category_id": 4,
            "truncation": 0,
            "occlusion": 0,
        }
    ]


def test_infer_coord_mode_trusts_requested_sent_frame():
    transform = _downscale_transform()
    mode = infer_coord_mode([[100, 100, 500, 500]], transform, preferred="sent")
    assert mode == "sent"


def test_infer_coord_mode_detects_fractional_normalized():
    transform = _downscale_transform()
    mode = infer_coord_mode([[0.1, 0.1, 0.5, 0.6]], transform, preferred="sent")
    assert mode == "normalized"


def test_infer_coord_mode_recovers_original_frame_when_model_ignores_resize():
    transform = _downscale_transform()
    # Coordinates far beyond the sent frame but inside the original image.
    mode = infer_coord_mode([[100, 100, 1850, 1450]], transform, preferred="sent")
    assert mode == "original"


def test_infer_coord_mode_detects_0_1000_when_normalized_requested():
    transform = _downscale_transform()
    mode = infer_coord_mode([[100, 100, 800, 900]], transform, preferred="normalized")
    assert mode == "normalized_1000"


def test_infer_coord_mode_empty_falls_back_to_requested():
    transform = _downscale_transform()
    assert infer_coord_mode([], transform, preferred="sent") == "sent"


def test_normalized_1000_reprojects_to_original_pixels():
    transform = _downscale_transform()
    box = sent_xyxy_to_original_xyxy([0, 0, 500, 500], transform, "normalized_1000")
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 == pytest.approx(0.0)
    assert y1 == pytest.approx(0.0)
    assert x2 == pytest.approx(1000.0, abs=1.0)
    assert y2 == pytest.approx(750.0, abs=1.0)


def test_validate_bbox_range_accepts_small_overflow_within_tolerance():
    transform = _downscale_transform()
    # 1300 slightly exceeds sent_width 1280 but is within the 15% margin.
    _validate_bbox_range(0, [0.0, 0.0, 1300.0, 900.0], transform, "sent", 0.15)


def test_validate_bbox_range_rejects_gross_out_of_frame_box():
    transform = _downscale_transform()
    with pytest.raises(ValueError):
        _validate_bbox_range(0, [0.0, 0.0, 2000.0, 900.0], transform, "sent", 0.15)


def test_predict_resume_skips_completed_images_and_preserves_existing_detections(tmp_path):
    visdrone_root = tmp_path / "VisDrone"
    ann_root = visdrone_root / "annotations"
    image_root = visdrone_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images"
    output_dir = tmp_path / "outputs"
    ann_root.mkdir(parents=True)
    image_root.mkdir(parents=True)
    output_dir.mkdir()

    Image.new("RGB", (32, 24), color="white").save(image_root / "000001.jpg")
    Image.new("RGB", (32, 24), color="white").save(image_root / "000002.jpg")
    (ann_root / "val.json").write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {"id": 1, "file_name": "000001.jpg", "width": 32, "height": 24},
                    {"id": 2, "file_name": "000002.jpg", "width": 32, "height": 24},
                ],
                "annotations": [],
                "categories": VISDRONE_CATEGORIES,
            }
        ),
        encoding="utf-8",
    )
    existing_detection = {
        "image_id": 1,
        "category_id": 4,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "score": 0.9,
    }
    (output_dir / "raw_responses.jsonl").write_text(
        json.dumps({"image_id": 1, "coco_detections": [existing_detection]}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detections.coco.json").write_text(
        json.dumps([existing_detection]),
        encoding="utf-8",
    )

    run_prediction(
        Namespace(
            visdrone_root=str(visdrone_root),
            output_dir=str(output_dir),
            model=None,
            api_key=None,
            base_url=None,
            max_images=2,
            start_index=0,
            image_max_side=1280,
            no_resize=True,
            letterbox=False,
            coord_mode="sent",
            coord_tolerance=0.15,
            temperature=0.0,
            max_tokens=None,
            json_mode="none",
            retries=0,
            retry_sleep=0.0,
            format_retries=0,
            sleep=0.0,
            min_score=0.0,
            extra_instruction=None,
            save_vis=False,
            vis_dir=None,
            vis_score_thr=0.0,
            vis_random_count=None,
            vis_random_seed=42,
            append_logs=False,
            resume=True,
            mock_empty=True,
            quiet=True,
        )
    )

    detections = json.loads((output_dir / "detections.coco.json").read_text(encoding="utf-8"))
    raw_lines = (output_dir / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    assert detections == [existing_detection]
    assert [json.loads(line)["image_id"] for line in raw_lines] == [1, 2]
    assert run_config["resume"] is True
    assert run_config["resume_enabled"] is True
    assert run_config["num_resume_skipped_images"] == 1

def test_predict_writes_detection_snapshots_during_run(tmp_path, monkeypatch):
    visdrone_root = tmp_path / "VisDrone"
    ann_root = visdrone_root / "annotations"
    image_root = visdrone_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images"
    output_dir = tmp_path / "outputs"
    ann_root.mkdir(parents=True)
    image_root.mkdir(parents=True)

    Image.new("RGB", (32, 24), color="white").save(image_root / "000001.jpg")
    Image.new("RGB", (32, 24), color="white").save(image_root / "000002.jpg")
    (ann_root / "val.json").write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {"id": 1, "file_name": "000001.jpg", "width": 32, "height": 24},
                    {"id": 2, "file_name": "000002.jpg", "width": 32, "height": 24},
                ],
                "annotations": [],
                "categories": VISDRONE_CATEGORIES,
            }
        ),
        encoding="utf-8",
    )

    original_write_json = predict_module.write_json
    prediction_snapshot_writes = 0

    def spy_write_json(path, payload):
        nonlocal prediction_snapshot_writes
        if getattr(path, "name", "") == "detections.coco.json":
            prediction_snapshot_writes += 1
        original_write_json(path, payload)

    monkeypatch.setattr(predict_module, "write_json", spy_write_json)

    run_prediction(
        Namespace(
            visdrone_root=str(visdrone_root),
            output_dir=str(output_dir),
            model=None,
            api_key=None,
            base_url=None,
            max_images=2,
            start_index=0,
            image_max_side=1280,
            no_resize=True,
            letterbox=False,
            coord_mode="sent",
            coord_tolerance=0.15,
            temperature=0.0,
            max_tokens=None,
            json_mode="none",
            retries=0,
            retry_sleep=0.0,
            format_retries=0,
            sleep=0.0,
            min_score=0.0,
            extra_instruction=None,
            save_vis=False,
            vis_dir=None,
            vis_score_thr=0.0,
            vis_random_count=None,
            vis_random_seed=42,
            append_logs=False,
            resume=False,
            mock_empty=True,
            quiet=True,
        )
    )

    assert prediction_snapshot_writes >= 4
    assert json.loads((output_dir / "detections.coco.json").read_text(encoding="utf-8")) == []
