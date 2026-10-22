"""Run with:  pytest -q"""
import cv2
import numpy as np
import pytest

from nutrivision.knowledge.kb import get_kb
from nutrivision.planning.planner import PantryItem, PlanningError, plan_week, validate_plan
from nutrivision.reasoning.rules import UserProfile, filter_recipes
from nutrivision.vision.segmentation import assess_ripeness, count_by_colour

kb = get_kb()


def pantry_of(**counts):
    return {i: PantryItem(i, n * kb.ingredients[i].unit_weight_g, kb.ingredients[i].shelf_life_days)
            for i, n in counts.items()}


def test_alias_resolution():
    assert kb.resolve("Bell Pepper") == "capsicum"
    assert kb.resolve("eggplant") == "brinjal"
    assert kb.resolve("tomatoes") == "tomato"
    assert kb.resolve("pizza") is None


def test_recipe_nutrition_is_derived_from_ingredients():
    rec = kb.recipes["banana_oats"]
    expected = sum(kb.ingredients[i].per_100g["kcal"] * g / 100 for i, g in rec.ingredients.items())
    assert kb.recipe_nutrition("banana_oats").kcal == pytest.approx(expected)
    assert kb.recipe_nutrition("banana_oats", 2).kcal == pytest.approx(2 * expected)


def test_jain_rule_excludes_root_vegetables():
    allowed, trace = filter_recipes(kb, UserProfile(diet="jain"))
    for rid in allowed:
        assert "root" not in kb.recipe_tags(rid)
    assert any(f.recipe_id == "aloo_paratha" for f in trace)


def test_vegan_with_nut_allergy():
    allowed, _ = filter_recipes(kb, UserProfile(diet="vegan", allergies=["nuts"]))
    for rid in allowed:
        assert not kb.recipe_tags(rid) & {"meat", "egg", "dairy", "nuts"}


@pytest.mark.parametrize("diet", ["vegetarian", "vegan", "jain", "eggetarian", "non-vegetarian"])
def test_plan_satisfies_all_hard_constraints(diet):
    pantry = pantry_of(tomato=6, onion=4, banana=4, spinach=1, carrot=4, capsicum=2, cabbage=1)
    plan = plan_week(kb, UserProfile(diet=diet, household_size=3), pantry, time_limit=20)
    assert validate_plan(kb, plan) == []
    assert len(plan.schedule) == 7 and all(len(d) == 3 for d in plan.schedule)


def test_spoiling_items_are_not_used_after_expiry():
    pantry = {"spinach": PantryItem("spinach", 500, 1.0), "tomato": PantryItem("tomato", 800, 7)}
    plan = plan_week(kb, UserProfile(diet="vegetarian"), pantry, time_limit=20)
    assert all(g == 0 for g in plan.pantry_daily["spinach"][1:])
    assert validate_plan(kb, plan) == []


def test_pantry_only_mode_never_buys_or_fails_cleanly():
    pantry = pantry_of(tomato=2)
    try:
        plan = plan_week(kb, UserProfile(), pantry, allow_buying=False, time_limit=10)
    except PlanningError:
        return
    assert plan.shopping_list == {}


def test_watershed_splits_touching_items():
    img = np.full((400, 600, 3), 200, np.uint8)
    for c in [(100, 100), (160, 100), (400, 250), (500, 100)]:
        cv2.circle(img, c, 32, (30, 30, 210), -1)
    n, _ = count_by_colour(img, "tomato")
    assert n == 4


def test_ripeness_stages():
    def blob(colour):
        c = np.full((120, 120, 3), 200, np.uint8)
        cv2.ellipse(c, (60, 60), (45, 30), 0, 0, 360, colour, -1)
        return c
    assert assess_ripeness(blob((40, 180, 60)), "banana").stage == "unripe"
    assert assess_ripeness(blob((30, 220, 230)), "banana").stage == "ripe"
    assert assess_ripeness(blob((30, 200, 220)), "spinach").stage == "spoiling"
