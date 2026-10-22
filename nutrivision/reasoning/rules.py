"""Rule-based reasoning over the knowledge base.

Each rule is an IF-THEN production: IF the user profile activates the rule AND
a recipe matches its condition, THEN the recipe is excluded, with a reason.
Every firing is recorded so the app can show *why* a dish was left out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from nutrivision.knowledge.kb import KnowledgeBase

# Diet -> ingredient tags that diet forbids.
DIET_RULES: dict[str, set[str]] = {
    "non-vegetarian": set(),
    "eggetarian": {"meat"},
    "vegetarian": {"meat", "egg"},
    "vegan": {"meat", "egg", "dairy"},
    "jain": {"meat", "egg", "root"},
}

# Allergy / intolerance -> forbidden tags.
ALLERGY_RULES: dict[str, set[str]] = {
    "gluten": {"gluten"},
    "nuts": {"nuts"},
    "dairy": {"dairy"},
    "egg": {"egg"},
    "soy": {"soy"},
}

HIGH_GI_LIMIT = 0.35   # diabetic-friendly: high-GI ingredients may be at most 35% of a dish by weight


@dataclass
class UserProfile:
    diet: str = "vegetarian"
    allergies: list[str] = field(default_factory=list)
    diabetic_friendly: bool = False
    dislikes: list[str] = field(default_factory=list)       # ingredient ids
    household_size: int = 2
    kcal_target: int = 1400                                  # per person per day, from the three main meals


@dataclass
class Rule:
    name: str
    active: Callable[[UserProfile], bool]
    violates: Callable[[str, UserProfile, KnowledgeBase], str | None]   # returns reason or None


@dataclass
class RuleFiring:
    recipe_id: str
    recipe_name: str
    rule: str
    reason: str


def _forbidden_tag_check(tag_map: dict[str, set[str]], keys: Callable[[UserProfile], list[str]], label: str):
    def check(recipe_id: str, profile: UserProfile, kb: KnowledgeBase) -> str | None:
        recipe = kb.recipes[recipe_id]
        for key in keys(profile):
            forbidden = tag_map.get(key, set())
            for ing_id in recipe.ingredients:
                hit = kb.ingredients[ing_id].tags & forbidden
                if hit:
                    return (f"{label} '{key}' excludes {', '.join(sorted(hit))} foods, "
                            f"and this dish contains {kb.ingredients[ing_id].name.lower()}")
        return None
    return check


def _high_gi_check(recipe_id: str, profile: UserProfile, kb: KnowledgeBase) -> str | None:
    recipe = kb.recipes[recipe_id]
    total = sum(recipe.ingredients.values())
    high = sum(g for i, g in recipe.ingredients.items() if "high_gi" in kb.ingredients[i].tags)
    share = high / total if total else 0.0
    if share > HIGH_GI_LIMIT:
        return (f"high-glycaemic ingredients make up {share:.0%} of the dish "
                f"(limit {HIGH_GI_LIMIT:.0%} for diabetic-friendly plans)")
    return None


def _dislike_check(recipe_id: str, profile: UserProfile, kb: KnowledgeBase) -> str | None:
    for ing_id in kb.recipes[recipe_id].ingredients:
        if ing_id in profile.dislikes:
            return f"you asked to avoid {kb.ingredients[ing_id].name.lower()}"
    return None


RULES: list[Rule] = [
    Rule("diet", lambda p: True, _forbidden_tag_check(DIET_RULES, lambda p: [p.diet], "Diet")),
    Rule("allergy", lambda p: bool(p.allergies), _forbidden_tag_check(ALLERGY_RULES, lambda p: p.allergies, "Allergy")),
    Rule("diabetic_friendly", lambda p: p.diabetic_friendly, _high_gi_check),
    Rule("dislikes", lambda p: bool(p.dislikes), _dislike_check),
]


def filter_recipes(kb: KnowledgeBase, profile: UserProfile) -> tuple[list[str], list[RuleFiring]]:
    """Apply every active rule to every recipe. Returns (allowed ids, firings)."""
    if profile.diet not in DIET_RULES:
        raise ValueError(f"Unknown diet '{profile.diet}'. Choose from {sorted(DIET_RULES)}")
    allowed, trace = [], []
    active_rules = [r for r in RULES if r.active(profile)]
    for recipe_id, recipe in kb.recipes.items():
        reason = None
        for rule in active_rules:
            reason = rule.violates(recipe_id, profile, kb)
            if reason:
                trace.append(RuleFiring(recipe_id, recipe.name, rule.name, reason))
                break
        if reason is None:
            allowed.append(recipe_id)
    return allowed, trace


def allowed_ingredients(kb: KnowledgeBase, profile: UserProfile) -> set[str]:
    """Ingredients the user may eat at all (useful for flagging pantry items the plan can't use)."""
    forbidden = set(DIET_RULES.get(profile.diet, set()))
    for a in profile.allergies:
        forbidden |= ALLERGY_RULES.get(a, set())
    return {i for i, ing in kb.ingredients.items()
            if not (ing.tags & forbidden) and i not in profile.dislikes}
