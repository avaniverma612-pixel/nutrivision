"""Project-wide settings. Values can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INGREDIENTS_PATH = DATA_DIR / "ingredients.json"
RECIPES_PATH = DATA_DIR / "recipes.json"

# YOLO weights. The default is YOLO-World, an open-vocabulary detector that finds any
# produce in the knowledge base from text prompts, with no training needed.
# A plain COCO model (yolo11n.pt) only knows apple, orange, banana, broccoli and carrot.
# Once you have fine-tuned weights (training/train_yolo.py), point this at best.pt.
YOLO_WEIGHTS = os.getenv("NUTRIVISION_YOLO_WEIGHTS", "yolov8s-worldv2.pt")
# Open-vocabulary models give lower confidence scores, so the default threshold is lower.
DETECTION_CONFIDENCE = float(os.getenv("NUTRIVISION_CONF", "0.15"))
DETECTION_IOU = float(os.getenv("NUTRIVISION_IOU", "0.5"))

# Any LiteLLM model string works with CrewAI, e.g. "gpt-4o-mini",
# "gemini/gemini-2.0-flash" or "groq/llama-3.3-70b-versatile".
LLM_MODEL = os.getenv("NUTRIVISION_LLM_MODEL", "gpt-4o-mini")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEALS = ["breakfast", "lunch", "dinner"]
