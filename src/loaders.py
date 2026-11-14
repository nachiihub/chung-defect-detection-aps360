# src/loaders.py
# each row: source, source_image_id, class_name, image_path, bbox, role
# bbox is [x, y, w, h] for GC10 and None for whole-image sources

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
                                        
def load_gc10(root =REPO / "data/raw/gc10"):
    rows = []
    for split in ["train", "test"]:
        folder = Path(root) / split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {i["id"]: i for i in coco["images"]}
        for ann in coco["annotations"]:
            img = imgs[ann["image_id"]]
            rows.append({
                "source": "gc10", "category": None,
                "source_image_id": img["file_name"],
                "class_name": cats[ann["category_id"]],
                "image_path": str(folder / img["file_name"]),
                "bbox": ann["bbox"],        # [x, y, w, h] in pixels
                "role": "defect",
            })
    return rows

def load_mvtec(root =REPO / "data/raw/mvtec"):
    categories = ["metal_nut", "screw", "grid"]
    rows = []
    for cat in categories:
        base = Path(root) / cat
        for p in sorted((base / "train" / "good").glob("*.png")):
            rows.append({
                "source": "mvtec", "category": cat,
                "source_image_id": f"{cat}/train/{p.name}",
                "class_name": "good", "image_path": str(p),
                "bbox": None, "role": "mvtec_train_good",
            })

        for folder in sorted((base / "test").iterdir()):
            if not folder.is_dir():
                continue
            is_good = folder.name == "good"
            for p in sorted(folder.glob("*.png")):
                rows.append({
                    "source": "mvtec", "category": cat,
                    "source_image_id": f"{cat}/test/{folder.name}/{p.name}",
                    "class_name": "good" if is_good else folder.name,
                    "image_path": str(p), "bbox": None,
                    "role": "mvtec_test_good" if is_good else "mvtec_test_defect",
                })
    return rows

def load_neu(root = REPO / "data/raw/NEU-CLS"):
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        class_name = p.stem.rsplit("_", 1)[0]
        rows.append({
            "source": "neu", "category": None,
            "source_image_id": p.name,
            "class_name": class_name, "image_path": str(p),
            "bbox": None, "role": "defect",
        })
    return rows