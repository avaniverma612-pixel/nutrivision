# NutriVision

NutriVision looks at a photo of your fruit and vegetables, counts what is there and judges how ripe it is, then builds a 7-day meal plan that uses the produce before it spoils while respecting your diet. It explains the nutrition behind the plan and links each dish to a recipe video.

## Running it

Python 3.10 or newer is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first detection downloads the YOLO weights automatically. API keys are optional and can be typed into the sidebar or exported as environment variables (see `.env.example`). Without an LLM key the plan, explanations and shopping list all still work, and only the agent advice tab is unavailable. Without a YouTube key each dish gets a YouTube search link instead of embedded videos. CrewAI uses LiteLLM model names, so `gpt-4o-mini`, `gemini/gemini-2.0-flash` or `groq/llama-3.3-70b-versatile` all work with the matching key.

Run the tests with `pytest -q`, and the planner evaluation with `python evaluate.py planner --trials 50`.

## How it works

The pipeline runs in four stages, and each one maps to a part of the syllabus.

Image processing (`nutrivision/vision/preprocess.py`) resizes the photo, applies gray-world white balancing to remove the colour cast of kitchen bulbs, and uses CLAHE on the L channel of LAB space to even out shadows without shifting hue. Hue matters because the ripeness checks depend on it.

Object detection (`nutrivision/vision/detector.py`) runs YOLO and maps whatever class names the weights use onto ingredient ids through aliases in the knowledge base. That means the same code works with the stock COCO model, which only knows apple, orange, banana, broccoli and carrot, and with your own fine-tuned weights. Counts come from the number of boxes per class after non-maximum suppression.

Colour segmentation (`nutrivision/vision/segmentation.py`) does two jobs. For each detection it separates the item from its background with GrabCut, then measures the share of ripe, unripe, brown and dark pixels in HSV space to label the item unripe, ripe, overripe or spoiling. That label scales its expected shelf life, which the planner uses. Separately, it estimates counts the classical way (colour mask, morphological opening and closing, distance transform, watershed) so touching items are split. The app shows this beside the YOLO count as a cross-check.

Knowledge representation (`nutrivision/knowledge/kb.py`, `data/*.json`) stores ingredients and recipes as frames. Recipe nutrition and diet tags are derived from the ingredient frames rather than stored, so correcting one ingredient updates every recipe that uses it.

Rule-based reasoning (`nutrivision/reasoning/rules.py`) applies IF-THEN rules for diet (vegetarian, eggetarian, vegan, Jain, non-vegetarian), allergies, a diabetic-friendly glycaemic rule and disliked ingredients. Every rule that fires is recorded, so the app can say exactly why a dish was left out.

Constraint-based planning (`nutrivision/planning/planner.py`) formulates the week as an integer programme. Hard constraints guarantee one allowed dish per meal, no dish twice in a day, a weekly repeat limit, never using more of an ingredient than you have, and never using produce after its expected spoil date. The objective rewards using pantry produce (perishable items earlier), keeps daily calories near the target, avoids repeats, and minimises what must be bought. Anything missing goes on a shopping list. The model is solved with OR-Tools CP-SAT, falling back to SciPy's HiGHS solver. An independent validator then re-checks every hard constraint without trusting the solver.

AI agents (`nutrivision/agents/crew.py`) are coordinated with CrewAI. A pantry analyst, a nutritionist and a home chef run in sequence, each building on the previous one's output. The nutritionist looks up facts through a knowledge-base tool instead of relying on the LLM's memory. The agents explain and present the plan but are not allowed to change dishes or quantities, because LLMs are unreliable at respecting quantity limits and the solver isn't.

## Training a better detector

Out of the box only five COCO classes are produce, so for a real demo you should fine-tune. Collect or download images (Roboflow Universe has several fruit and vegetable detection datasets, and Fruits-360 is useful for classification pretraining), label them in YOLO format with class names matching the ids in `training/data.yaml`, then run:

```bash
python training/train_yolo.py --data training/data.yaml --epochs 80
export NUTRIVISION_YOLO_WEIGHTS=runs/detect/nutrivision/weights/best.pt
```

Hue augmentation is deliberately kept small in the training script, because hue carries the ripeness signal.

To measure counting accuracy, put test photos in a folder and list the true counts in a CSV with the columns `image,ingredient,count`, then run `python evaluate.py counting --images eval_images --labels eval_counts.csv`. It reports the mean absolute error for both YOLO and the colour-segmentation count.

## Extending it

Recipes and ingredients live in `data/recipes.json` and `data/ingredients.json`. Add a recipe by listing grams per serving of each tracked ingredient, and it is picked up automatically. Nutrition values are approximate figures adapted from USDA FoodData Central. For the final report, replace them with exact values from FoodData Central or the Indian Food Composition Tables (IFCT 2017).

## Project layout

```
app.py                     Streamlit interface
evaluate.py                counting and planner evaluation
data/                      ingredient and recipe knowledge base
nutrivision/vision/        preprocessing, YOLO detection, colour segmentation
nutrivision/knowledge/     knowledge base and nutrition inference
nutrivision/reasoning/     diet rules and plain-language explanations
nutrivision/planning/      constraint model and solver backends
nutrivision/agents/        CrewAI agents
nutrivision/services/      YouTube Data API client
training/                  YOLO fine-tuning script and dataset config
tests/                     unit tests
```
