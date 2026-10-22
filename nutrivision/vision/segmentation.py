"""Colour segmentation for ripeness/spoilage estimation and count cross-checking.

All colour work is done in HSV, where hue is mostly independent of brightness.
OpenCV hue runs 0-179.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# name -> list of (lower, upper) HSV bounds
COLOURS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red":    [((0, 80, 50), (9, 255, 255)), ((170, 80, 50), (179, 255, 255))],
    "orange": [((10, 110, 110), (21, 255, 255))],
    "yellow": [((22, 80, 110), (34, 255, 255))],
    "green":  [((35, 45, 40), (88, 255, 255))],
    "purple": [((120, 40, 25), (169, 255, 255))],
    "white":  [((0, 0, 170), (179, 45, 255))],
    "brown":  [((5, 50, 20), (24, 255, 105))],
    "dark":   [((0, 0, 0), (179, 255, 40))],
}

# ripe colours, unripe colours, colours used for counting, whether brown is a natural colour
PROFILES: dict[str, dict] = {
    "tomato":   {"ripe": ["red"], "unripe": ["green"], "count": ["red"]},
    "banana":   {"ripe": ["yellow"], "unripe": ["green"], "count": ["yellow", "green"]},
    "mango":    {"ripe": ["yellow", "orange"], "unripe": ["green"], "count": ["yellow", "orange", "green"]},
    "orange":   {"ripe": ["orange"], "unripe": ["green"], "count": ["orange"]},
    "apple":    {"ripe": ["red", "yellow", "green"], "unripe": [], "count": ["red"]},
    "lemon":    {"ripe": ["yellow"], "unripe": ["green"], "count": ["yellow", "green"]},
    "carrot":   {"ripe": ["orange"], "unripe": [], "count": ["orange"]},
    "capsicum": {"ripe": ["green", "red", "yellow"], "unripe": [], "count": ["green", "red", "yellow"]},
    "cucumber": {"ripe": ["green"], "unripe": [], "count": ["green"]},
    "broccoli": {"ripe": ["green"], "unripe": [], "count": ["green"], "yellowing_spoils": True},
    "spinach":  {"ripe": ["green"], "unripe": [], "count": ["green"], "yellowing_spoils": True},
    "coriander": {"ripe": ["green"], "unripe": [], "count": ["green"], "yellowing_spoils": True},
    "beans":    {"ripe": ["green"], "unripe": [], "count": ["green"], "yellowing_spoils": True},
    "green_chilli": {"ripe": ["green"], "unripe": [], "count": ["green"]},
    "brinjal":  {"ripe": ["purple"], "unripe": [], "count": ["purple"]},
    "beetroot": {"ripe": ["purple", "red"], "unripe": [], "count": ["purple", "red"]},
    "cauliflower": {"ripe": ["white"], "unripe": [], "count": ["white"]},
    "cabbage":  {"ripe": ["green", "white"], "unripe": [], "count": ["green"]},
    "grapes":   {"ripe": ["green", "purple"], "unripe": [], "count": ["green", "purple"]},
    "pear":     {"ripe": ["green", "yellow"], "unripe": [], "count": ["green", "yellow"]},
    "pomegranate": {"ripe": ["red"], "unripe": [], "count": ["red"]},
    "pineapple": {"ripe": ["yellow", "orange"], "unripe": ["green"], "count": ["yellow", "orange"],
                  "brown_natural": True},
    "potato":   {"ripe": [], "unripe": [], "count": ["brown"], "brown_natural": True},
    "onion":    {"ripe": [], "unripe": [], "count": ["purple", "red"], "brown_natural": True},
    "garlic":   {"ripe": [], "unripe": [], "count": ["white"], "brown_natural": True},
    "ginger":   {"ripe": [], "unripe": [], "count": ["brown"], "brown_natural": True},
}

SHELF_FACTOR = {"unripe": 1.3, "ripe": 1.0, "fresh": 1.0, "overripe": 0.3, "spoiling": 0.3}


@dataclass
class RipenessResult:
    stage: str              # unripe / ripe / fresh / overripe / spoiling
    freshness: float        # 0 (spoiled) .. 1 (fresh)
    fractions: dict         # colour -> share of the item's pixels
    shelf_factor: float


def colour_mask(hsv: np.ndarray, names: list[str]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for name in names:
        for lo, hi in COLOURS[name]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def foreground_mask(crop: np.ndarray, iterations: int = 3) -> np.ndarray:
    """GrabCut seeded with a rectangle slightly inside the detection box, so the
    table or basket behind the item is excluded from the colour statistics."""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    dx, dy = max(1, w // 20), max(1, h // 20)
    rect = (dx, dy, w - 2 * dx, h - 2 * dy)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros((h, w), np.uint8)
    if fg.mean() < 255 * 0.10:          # GrabCut failed; fall back to "anything saturated"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        fg = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
    return fg


def assess_ripeness(crop_bgr: np.ndarray, ingredient_id: str) -> RipenessResult:
    profile = PROFILES.get(ingredient_id, {"ripe": [], "unripe": [], "count": []})
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    fg = foreground_mask(crop_bgr) > 0
    total = max(int(fg.sum()), 1)

    def share(names: list[str]) -> float:
        return float((colour_mask(hsv, names)[fg] > 0).sum()) / total if names else 0.0

    fractions = {name: round(share([name]), 3) for name in COLOURS}
    ripe, unripe = share(profile["ripe"]), share(profile["unripe"])
    if profile.get("brown_natural"):
        spoil = max(0.0, fractions["dark"] - 0.15)        # only heavy dark rot counts
    else:
        spoil = fractions["brown"] + fractions["dark"]
        if profile.get("yellowing_spoils"):
            spoil += fractions["yellow"]

    if spoil > 0.30:
        stage = "spoiling" if profile.get("yellowing_spoils") else "overripe"
    elif profile["unripe"] and unripe > max(0.35, ripe):
        stage = "unripe"
    elif profile["unripe"]:
        stage = "ripe"
    else:
        stage = "fresh"
    freshness = float(np.clip(1.0 - 2.0 * spoil, 0.0, 1.0))
    return RipenessResult(stage, round(freshness, 2), fractions, SHELF_FACTOR[stage])


def count_by_colour(img_bgr: np.ndarray, ingredient_id: str, min_area_frac: float = 0.002,
                    split_threshold: float = 0.55) -> tuple[int, np.ndarray]:
    """Classical count estimate: colour mask -> morphology -> distance transform
    -> watershed, so touching items are split. Used to cross-check YOLO counts.
    Returns (count, BGR visualisation)."""
    names = PROFILES.get(ingredient_id, {}).get("count")
    if not names:
        return 0, img_bgr.copy()
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = colour_mask(hsv, names)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return 0, img_bgr.copy()
    _, sure_fg = cv2.threshold(dist, split_threshold * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr.copy(), markers)

    min_area = min_area_frac * img_bgr.shape[0] * img_bgr.shape[1]
    vis = img_bgr.copy()
    count = 0
    for label in range(2, n + 1):
        region = (markers == label).astype(np.uint8)
        if region.sum() < min_area:
            continue
        count += 1
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)
    vis[markers == -1] = (0, 0, 255)
    return count, vis
