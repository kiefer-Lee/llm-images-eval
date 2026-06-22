from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco_io import write_json


METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def main() -> None:
    args = parse_args()
    metrics = evaluate_coco(
        ann_file=args.ann_file,
        pred_file=args.pred_file,
        iou_type=args.iou_type,
        max_dets=args.max_dets,
        output=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO-style detection evaluation.")
    parser.add_argument("--ann-file", required=True, help="Ground-truth COCO annotation JSON.")
    parser.add_argument("--pred-file", required=True, help="COCO detection result JSON list.")
    parser.add_argument("--iou-type", default="bbox", choices=["bbox", "segm"])
    parser.add_argument("--max-dets", default="1,10,100", help="Comma-separated maxDets.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def evaluate_coco(
    ann_file: str | Path,
    pred_file: str | Path,
    iou_type: str = "bbox",
    max_dets: str = "1,10,100",
    output: str | Path | None = None,
) -> dict[str, Any]:
    coco_gt = COCO(str(ann_file))
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_gt.createIndex()

    with Path(pred_file).open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        raise ValueError("Prediction file must be a COCO detection JSON list.")

    if len(predictions) == 0:
        metrics = {name: 0.0 for name in METRIC_NAMES}
        metrics["num_predictions"] = 0
        metrics["summary"] = "No predictions were provided; all COCO metrics are set to 0."
        if output:
            write_json(output, metrics)
        return metrics

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.maxDets = [int(v) for v in max_dets.split(",")]
    evaluator.evaluate()
    evaluator.accumulate()

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        evaluator.summarize()

    stats = [float(v) for v in evaluator.stats]
    metrics = {name: stats[idx] for idx, name in enumerate(METRIC_NAMES[: len(stats)])}
    metrics["num_predictions"] = len(predictions)
    metrics["summary"] = capture.getvalue()
    if output:
        write_json(output, metrics)
    return metrics


if __name__ == "__main__":
    main()
