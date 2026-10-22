"""Knowledge representation for NutriVision.

Ingredients and recipes are stored as frames (typed records with slots). The
knowledge base links them: a recipe frame references ingredient frames, and
nutrition for a recipe is *derived* from the ingredient frames rather than stored,
so correcting one ingredient's data updates every recipe that uses it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from nutrivision import config

NUTRIENT_KEYS = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")


def normalise(name: str) -> str:
    """'Bell Pepper' / 'bell-pepper' / 'bell_pepper' -> 'bell_pepper'."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    kind: str                 # "produce" or "staple"
    unit: str                 # what one detected item is ("piece", "bunch", "head") or "g"
    unit_weight_g: float
    shelf_life_days: int
    per_100g: dict
    tags: frozenset
    aliases: tuple
    key_nutrients: tuple
    benefit: str

    @property
    def is_produce(self) -> bool:
        return self.kind == "produce"


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    meal_types: tuple
    ingredients: dict          # ingredient id -> grams per serving
    description: str
    video_query: str


@dataclass
class Nutrition:
    kcal: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0

    def __add__(self, other: "Nutrition") -> "Nutrition":
        return Nutrition(*(getattr(self, k) + getattr(other, k) for k in NUTRIENT_KEYS))

    def scaled(self, factor: float) -> "Nutrition":
        return Nutrition(*(getattr(self, k) * factor for k in NUTRIENT_KEYS))

    def as_dict(self, digits: int = 1) -> dict:
        return {k: round(getattr(self, k), digits) for k in NUTRIENT_KEYS}


class KnowledgeBase:
    def __init__(self, ingredients_path: Path = config.INGREDIENTS_PATH,
                 recipes_path: Path = config.RECIPES_PATH):
        raw_ing = json.loads(Path(ingredients_path).read_text(encoding="utf-8"))["ingredients"]
        raw_rec = json.loads(Path(recipes_path).read_text(encoding="utf-8"))["recipes"]

        self.ingredients: dict[str, Ingredient] = {}
        self._alias_index: dict[str, str] = {}
        for item in raw_ing:
            ing = Ingredient(
                id=item["id"], name=item["name"], kind=item["kind"], unit=item["unit"],
                unit_weight_g=float(item["unit_weight_g"]),
                shelf_life_days=int(item["shelf_life_days"]),
                per_100g=item["per_100g"], tags=frozenset(item.get("tags", [])),
                aliases=tuple(item.get("aliases", [])),
                key_nutrients=tuple(item.get("key_nutrients", [])),
                benefit=item.get("benefit", ""),
            )
            self.ingredients[ing.id] = ing
            for key in (ing.id, ing.name, *ing.aliases):
                self._alias_index[normalise(key)] = ing.id

        self.recipes: dict[str, Recipe] = {}
        for item in raw_rec:
            missing = [i for i in item["ingredients"] if i not in self.ingredients]
            if missing:
                raise ValueError(f"Recipe '{item['id']}' uses unknown ingredients: {missing}")
            rec = Recipe(
                id=item["id"], name=item["name"], meal_types=tuple(item["meal_types"]),
                ingredients={k: float(v) for k, v in item["ingredients"].items()},
                description=item.get("description", ""),
                video_query=item.get("video_query", item["name"] + " recipe"),
            )
            self.recipes[rec.id] = rec

    # --- lookup -----------------------------------------------------------
    def resolve(self, name: str) -> str | None:
        """Map a detector label or user-typed name to an ingredient id."""
        key = normalise(name)
        if key in self._alias_index:
            return self._alias_index[key]
        if key.endswith("s") and key[:-1] in self._alias_index:   # naive plural
            return self._alias_index[key[:-1]]
        return None

    def produce(self) -> list[Ingredient]:
        return [i for i in self.ingredients.values() if i.is_produce]

    def staples(self) -> list[Ingredient]:
        return [i for i in self.ingredients.values() if not i.is_produce]

    # --- inference ----------------------------------------------------------
    def ingredient_nutrition(self, ingredient_id: str, grams: float) -> Nutrition:
        per = self.ingredients[ingredient_id].per_100g
        return Nutrition(*(per.get(k, 0.0) * grams / 100.0 for k in NUTRIENT_KEYS))

    def recipe_nutrition(self, recipe_id: str, servings: float = 1.0) -> Nutrition:
        total = Nutrition()
        for ing_id, grams in self.recipes[recipe_id].ingredients.items():
            total = total + self.ingredient_nutrition(ing_id, grams)
        return total.scaled(servings)

    def recipe_tags(self, recipe_id: str) -> set[str]:
        """A recipe inherits every tag of the ingredients it contains."""
        tags: set[str] = set()
        for ing_id in self.recipes[recipe_id].ingredients:
            tags |= self.ingredients[ing_id].tags
        return tags

    def facts(self, ingredient_id: str) -> str:
        """Human-readable fact sheet, used to ground the LLM agents."""
        ing = self.ingredients[ingredient_id]
        p = ing.per_100g
        return (f"{ing.name}: per 100 g about {p['kcal']} kcal, {p['protein_g']} g protein, "
                f"{p['carbs_g']} g carbs, {p['fat_g']} g fat, {p['fiber_g']} g fibre. "
                f"Key nutrients: {', '.join(ing.key_nutrients)}. {ing.benefit}")


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
