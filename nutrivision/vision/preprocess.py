"""Image preprocessing applied before detection and colour analysis."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR image. Falls back to Pillow for formats
    OpenCV can't read (WebP/AVIF saved as .jpg, GIF, etc.) and fixes phone rotation."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise ValueError("Could not read the image. Try saving it as a PNG and uploading again.")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def resize_max_side(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def normalise_lighting(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel of LAB: evens out shadows from kitchen lighting
    without shifting hue, which matters for the colour-based ripeness checks."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Gray-world assumption: scale channels so their means match. Removes the
    yellow cast of warm bulbs that would otherwise make produce look riper."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray, *, white_balance: bool = True, clahe: bool = True,
               denoise: bool = False) -> np.ndarray:
    out = resize_max_side(img)
    if white_balance:
        out = gray_world_white_balance(out)
    if clahe:
        out = normalise_lighting(out)
    if denoise:
        out = cv2.bilateralFilter(out, d=7, sigmaColor=50, sigmaSpace=50)
    return out
