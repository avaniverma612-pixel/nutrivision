"""Evaluation for the project report.

Counting accuracy (needs ultralytics and a labelled folder):
    python evaluate.py counting --images eval_images --labels eval_counts.csv
    eval_counts.csv columns: image,ingredient,count   (one row per ingredient per image)

Planner quality (runs anywhere):
    python evaluate.py planner --trials 50
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from nutrivision.knowledge.kb import get_kb


def eval_counting(images: Path, labels: Path, weights: str | None):
    import csv

    import cv2

    from nutrivision.vision.detector import ProduceDetector
    from nutrivision.vision.preprocess import preprocess

    kb = get_kb()
    detector = ProduceDetector(kb, **({"weights": weights} if weights else {}))
    truth: dict[str, dict[str, int]] = defaultdict(dict)
    with open(labels, newline="") as fh:
        for row in csv.DictReader(fh):
            truth[row["image"]][kb.resolve(row["ingredient"]) or row["ingredient"]] = int(row["count"])

    errs_yolo, errs_colour = defaultdict(list), defaultdict(list)
    for name, counts in truth.items():
        img = cv2.imread(str(images / name))
        if img is None:
            print(f"skipping unreadable {name}")
            continue
        res = detector.detect(preprocess(img))
        found = {r.ingredient_id: r for r in res.inventory}
        for ing, true_n in counts.items():
            row = found.get(ing)
            errs_yolo[ing].append(abs((row.count if row else 0) - true_n))
            if row and row.colour_count is not None:
                errs_colour[ing].append(abs(row.colour_count - true_n))

    print(f"{'ingredient':<14}{'n':>4}{'YOLO MAE':>10}{'exact':>8}{'colour MAE':>12}")
    all_y = []
    for ing in sorted(errs_yolo):
        e = errs_yolo[ing]; all_y += e
        c = errs_colour.get(ing)
        print(f"{ing:<14}{len(e):>4}{statistics.mean(e):>10.2f}{sum(x == 0 for x in e) / len(e):>8.0%}"
              f"{(statistics.mean(c) if c else float('nan')):>12.2f}")
    if all_y:
        print(f"\nOverall YOLO count MAE {statistics.mean(all_y):.2f}, "
              f"exact-count rate {sum(x == 0 for x in all_y) / len(all_y):.0%}")


def eval_planner(trials: int, seed: int):
    from nutrivision.planning.planner import PantryItem, PlanningError, plan_week, validate_plan
    from nutrivision.reasoning.rules import ALLERGY_RULES, DIET_RULES, UserProfile

    kb = get_kb()
    rng = random.Random(seed)
    produce = kb.produce()
    util, times, violations, failures, repeats = [], [], 0, 0, []
    for _ in range(trials):
        pantry = {}
        for ing in rng.sample(produce, rng.randint(3, 10)):
            n = rng.randint(1, 6)
            pantry[ing.id] = PantryItem(ing.id, n * ing.unit_weight_g,
                                        ing.shelf_life_days * rng.choice([0.3, 1.0, 1.0, 1.3]))
        profile = UserProfile(diet=rng.choice(list(DIET_RULES)),
                              allergies=rng.sample(list(ALLERGY_RULES), rng.choice([0, 0, 1])),
                              diabetic_friendly=rng.random() < 0.2, household_size=rng.randint(1, 5))
        try:
            plan = plan_week(kb, profile, pantry)
        except PlanningError:
            failures += 1
            continue
        problems = validate_plan(kb, plan)
        violations += bool(problems)
        util.append(plan.produce_utilisation)
        times.append(plan.seconds)
        repeats.append(len({r for d in plan.schedule for r in d.values()}))
    ok = trials - failures
    print(f"trials {trials}, solved {ok}, no feasible plan {failures}")
    if ok:
        print(f"plans with any constraint violation: {violations}/{ok}")
        print(f"mean produce utilisation {statistics.mean(util):.0%} (min {min(util):.0%})")
        print(f"mean distinct dishes per week {statistics.mean(repeats):.1f}")
        print(f"solve time mean {statistics.mean(times):.2f}s, max {max(times):.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counting")
    c.add_argument("--images", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True)
    c.add_argument("--weights")
    p = sub.add_parser("planner")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "counting":
        eval_counting(a.images, a.labels, a.weights)
    else:
        eval_planner(a.trials, a.seed)
