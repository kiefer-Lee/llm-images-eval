from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisDroneValPaths:
    root: Path
    ann_file: Path
    image_root: Path


def default_visdrone_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "Datasets" / "VisDrone",
        project_root.parent / "Datasets" / "VisDrone",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_visdrone_val_paths(visdrone_root: str | Path | None = None) -> VisDroneValPaths:
    root = Path(visdrone_root) if visdrone_root is not None else default_visdrone_root()
    root = root.resolve()
    return VisDroneValPaths(
        root=root,
        ann_file=root / "annotations" / "val.json",
        image_root=root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images",
    )


def validate_visdrone_val_paths(paths: VisDroneValPaths) -> None:
    if paths.ann_file.name != "val.json" or paths.ann_file.parent.name != "annotations":
        raise ValueError(f"Expected VisDrone val annotation file, got: {paths.ann_file}")
    if "VisDrone2019-DET-val" not in paths.image_root.parts or paths.image_root.name != "images":
        raise ValueError(f"Expected VisDrone val image directory, got: {paths.image_root}")
    if not paths.ann_file.exists():
        raise FileNotFoundError(f"VisDrone val annotation not found: {paths.ann_file}")
    if not paths.image_root.exists():
        raise FileNotFoundError(f"VisDrone val image directory not found: {paths.image_root}")
