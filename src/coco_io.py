from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str
    supercategory: str | None = None


class CocoDataset:
    def __init__(self, ann_file: str | Path, image_root: str | Path | None = None) -> None:
        self.ann_file = Path(ann_file)
        self.image_root = Path(image_root) if image_root is not None else self.ann_file.parent
        with self.ann_file.open("r", encoding="utf-8") as f:
            self.data: dict[str, Any] = json.load(f)

        self.images = [
            CocoImage(
                id=int(img["id"]),
                file_name=str(img["file_name"]),
                width=int(img["width"]),
                height=int(img["height"]),
            )
            for img in self.data.get("images", [])
        ]
        self.categories = [
            CocoCategory(
                id=int(cat["id"]),
                name=str(cat["name"]),
                supercategory=cat.get("supercategory"),
            )
            for cat in self.data.get("categories", [])
        ]
        self.category_by_id = {cat.id: cat for cat in self.categories}
        self.category_id_by_name = {
            normalize_label(cat.name): cat.id for cat in self.categories
        }

    def resolve_image_path(self, image: CocoImage) -> Path:
        path = self.image_root / image.file_name
        if path.exists():
            return path

        # Some COCO files include nested paths while the caller points at a
        # leaf image directory. Fall back to the basename before failing.
        fallback = self.image_root / Path(image.file_name).name
        if fallback.exists():
            return fallback
        return path

    def category_prompt(self) -> str:
        rows = [f"{cat.id}: {cat.name}" for cat in self.categories]
        return "\n".join(rows)

    def resolve_category_id(self, label: str | None, category_id: int | None) -> int | None:
        if category_id is not None and category_id in self.category_by_id:
            return category_id
        if not label:
            return None
        return self.category_id_by_name.get(normalize_label(label))


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
