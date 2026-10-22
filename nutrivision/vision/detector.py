"""Produce detection and counting with YOLO (ultralytics).

The detector is label-agnostic: whatever class names the weights use are mapped
to ingredient ids through the knowledge base aliases, so the same code works for
a stock COCO model and for custom fine-tuned weights.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from nutrivision import config
from nutrivision.knowledge.kb import KnowledgeBase
from nutrivision.vision.segmentation import RipenessResult, assess_ripeness, count_by_colour

STAGE_COLOURS = {"unripe": (60, 180, 60), "ripe": (0, 200, 255), "fresh": (0, 200, 0),
                 "overripe": (30, 60, 160), "spoiling": (30, 60, 160)}


@dataclass
class Detection:
    ingredient_id: str
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    ripeness: RipenessResult | None = None


@dataclass
class InventoryRow:
    ingredient_id: str
    name: str
    count: int
    confidence: float
    stages: dict[str, int]
    shelf_factor: float
    colour_count: int | None = None

    @property
    def main_stage(self) -> str:
        return max(self.stages, key=self.stages.get) if self.stages else "fresh"


@dataclass
class DetectionResult:
    detections: list[Detection]
    inventory: list[InventoryRow]
    annotated: np.ndarray
    ignored_labels: dict[str, int] = field(default_factory=dict)


# Text prompts for open-vocabulary models, where the plain ingredient name is ambiguous
# or uses Indian English that CLIP-style text encoders saw less often.
PROMPTS = {
    "capsicum": "bell pepper", "brinjal": "eggplant", "beans": "green beans",
    "coriander": "coriander leaves", "green_chilli": "green chili pepper",
    "spinach": "spinach leaves", "beetroot": "beetroot", "grapes": "bunch of grapes",
}


def is_open_vocab(weights: str) -> bool:
    name = str(weights).lower()
    return "world" in name or "yoloe" in name


class ProduceDetector:
    """Wraps either a closed-set YOLO model (COCO or your fine-tuned weights) or an
    open-vocabulary YOLO-World model. YOLO-World is given the produce names from the
    knowledge base as text prompts, so it can find grapes, pears, tomatoes and so on
    without any training. Fine-tuned weights are still more accurate once you have them."""

    def __init__(self, kb: KnowledgeBase, weights: str = config.YOLO_WEIGHTS,
                 conf: float = config.DETECTION_CONFIDENCE, iou: float = config.DETECTION_IOU):
        self.kb = kb
        self.conf, self.iou = conf, iou
        self.open_vocab = is_open_vocab(weights)
        if self.open_vocab:
            from ultralytics import YOLOWorld   # imported lazily so the rest of the app runs without it
            produce = [ing.id for ing in kb.produce()]
            prompts = [PROMPTS.get(i, kb.ingredients[i].name.lower()) for i in produce]
            self.model = YOLOWorld(weights)
            self.model.set_classes(prompts)
            self.class_map = dict(enumerate(produce))
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.class_map = {cid: kb.resolve(name) for cid, name in self.model.names.items()}

    @property
    def supported(self) -> list[str]:
        return sorted({v for v in self.class_map.values() if v and self.kb.ingredients[v].is_produce})

    def detect(self, img_bgr: np.ndarray, *, with_ripeness: bool = True,
               colour_cross_check: bool = True) -> DetectionResult:
        result = self.model.predict(img_bgr, conf=self.conf, iou=self.iou,
                                    agnostic_nms=True, verbose=False)[0]
        detections, ignored = [], defaultdict(int)
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, cid, cf in zip(xyxy, classes, confs):
                ing = self.class_map.get(int(cid))
                label = self.model.names.get(int(cid), str(cid)) if isinstance(self.model.names, dict) \
                    else self.model.names[int(cid)]
                if not ing or not self.kb.ingredients[ing].is_produce:
                    ignored[label] += 1
                    continue
                x1, y1, x2, y2 = (int(v) for v in box)
                det = Detection(ing, label, float(cf), (x1, y1, x2, y2))
                if with_ripeness:
                    crop = img_bgr[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size:
                        det.ripeness = assess_ripeness(crop, ing)
                detections.append(det)

        inventory = build_inventory(self.kb, detections)
        if colour_cross_check:
            for row in inventory:
                row.colour_count, _ = count_by_colour(img_bgr, row.ingredient_id)
        return DetectionResult(detections, inventory, annotate(img_bgr, detections, self.kb), dict(ignored))


def build_inventory(kb: KnowledgeBase, detections: list[Detection]) -> list[InventoryRow]:
    groups: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        groups[d.ingredient_id].append(d)
    rows = []
    for ing, dets in sorted(groups.items()):
        stages: dict[str, int] = defaultdict(int)
        factors = []
        for d in dets:
            if d.ripeness:
                stages[d.ripeness.stage] += 1
                factors.append(d.ripeness.shelf_factor)
        rows.append(InventoryRow(
            ingredient_id=ing, name=kb.ingredients[ing].name, count=len(dets),
            confidence=round(float(np.mean([d.confidence for d in dets])), 2),
            stages=dict(stages), shelf_factor=round(float(np.mean(factors)), 2) if factors else 1.0))
    return rows


def annotate(img_bgr: np.ndarray, detections: list[Detection], kb: KnowledgeBase) -> np.ndarray:
    out = img_bgr.copy()
    scale = max(out.shape[:2]) / 1000
    for d in detections:
        stage = d.ripeness.stage if d.ripeness else "fresh"
        colour = STAGE_COLOURS.get(stage, (255, 255, 255))
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, max(2, int(2 * scale)))
        text = f"{kb.ingredients[d.ingredient_id].name} {d.confidence:.2f} {stage}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.2, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale + 0.2, (0, 0, 0), 1, cv2.LINE_AA)
    return out
