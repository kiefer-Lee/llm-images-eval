from __future__ import annotations

import json

import pytest
from PIL import Image

from src.image_ops import (
    ImageTransform,
    infer_coord_mode,
    sent_xyxy_to_original_xyxy,
)
from src.predict import _load_append_log_resume_state, _validate_bbox_range
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
