from __future__ import annotations

import json

from PIL import Image

from src.predict import _load_append_log_resume_state
from src.tools.visdrone_to_coco import VISDRONE_CATEGORIES, convert_visdrone_to_coco


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
