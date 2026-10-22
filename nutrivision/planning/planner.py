"""Constraint-based weekly meal planning.

Decision variables
    x[r,d,m]  = 1 if recipe r is cooked for meal m on day d
    p[i,d]    = grams of ingredient i taken from the pantry on day d
    b[i,d]    = grams of ingredient i that must be bought for day d

Hard constraints
    * every meal slot gets exactly one allowed recipe of the right meal type
    * a recipe is not served twice on the same day, and at most `max_repeats` times a week
    * p[i,d] + b[i,d] = grams of i needed by the recipes chosen for day d
    * total pantry use of i never exceeds what the user has
    * pantry produce cannot be used on or after the day it is expected to spoil
    * if buying is switched off, b = 0

Soft goals (objective)
    + use pantry produce, the more perishable the earlier
    - buy as little as possible
    - keep each day's calories within +/-10% of the per-person target
    - avoid repeating dishes
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from nutrivision import config
from nutrivision.knowledge.kb import KnowledgeBase, Nutrition
from nutrivision.planning.model import LinearModel
from nutrivision.reasoning.rules import RuleFiring, UserProfile, filter_recipes

W_BUY = 30          # penalty per gram bought
W_KCAL = 3          # penalty per kcal outside the target band (per person per day)
W_REPEAT = 400      # penalty per extra use of the same dish
KCAL_BAND = 0.10


class PlanningError(Exception):
    pass


@dataclass
class PantryItem:
    ingredient_id: str
    quantity_g: float
    shelf_life_days: float          # effective days left (after ripeness adjustment)


@dataclass
class PlanResult:
    schedule: list[dict[str, str]]                     # day index -> {meal: recipe_id}
    daily_nutrition: list[Nutrition]                   # per person
    pantry_daily: dict[str, list[int]]                 # grams from pantry per day
    bought_daily: dict[str, list[int]]                 # grams bought per day
    pantry: dict[str, PantryItem]
    servings: int
    solver: str
    status: str
    seconds: float
    max_repeats: int
    rule_trace: list[RuleFiring] = field(default_factory=list)
    allowed_recipes: list[str] = field(default_factory=list)

    @property
    def shopping_list(self) -> dict[str, int]:
        return {i: sum(v) for i, v in self.bought_daily.items() if sum(v) > 0}

    def pantry_used(self, ingredient_id: str) -> int:
        return sum(self.pantry_daily.get(ingredient_id, []))

    @property
    def produce_utilisation(self) -> float:
        avail = sum(p.quantity_g for p in self.pantry.values())
        used = sum(min(self.pantry_used(i), p.quantity_g) for i, p in self.pantry.items())
        return used / avail if avail else 0.0


def _demand(kb: KnowledgeBase, schedule: list[dict[str, str]], servings: int) -> dict[str, list[int]]:
    """Grams of each ingredient needed per day for a given schedule."""
    need: dict[str, list[int]] = {}
    for d, meals in enumerate(schedule):
        for rid in meals.values():
            for ing, grams in kb.recipes[rid].ingredients.items():
                need.setdefault(ing, [0] * len(schedule))[d] += round(grams * servings)
    return need


def plan_week(kb: KnowledgeBase, profile: UserProfile, pantry: dict[str, PantryItem], *,
              staples_unlimited: bool = True, allow_buying: bool = True, max_repeats: int = 2,
              days: int = 7, meals: list[str] | None = None, time_limit: float = 20.0,
              solver: str = "auto") -> PlanResult:
    meals = meals or list(config.MEALS)
    servings = max(1, int(profile.household_size))
    allowed, trace = filter_recipes(kb, profile)

    by_meal = {m: [r for r in allowed if m in kb.recipes[r].meal_types] for m in meals}
    empty = [m for m, rs in by_meal.items() if not rs]
    if empty:
        raise PlanningError(f"No recipe satisfies your diet rules for: {', '.join(empty)}. "
                            "Relax a restriction or add recipes to data/recipes.json.")

    # Make sure the repeat limit leaves enough distinct dishes to fill the week:
    # for every group of meal types, the dishes that can fill them must cover all their slots.
    for size in range(1, len(meals) + 1):
        for group in itertools.combinations(meals, size):
            pool = {r for m in group for r in by_meal[m]}
            max_repeats = max(max_repeats, math.ceil(days * len(group) / len(pool)))
    max_repeats = min(max_repeats, days)
    shared = {r for m in meals for r in by_meal[m]}

    mdl = LinearModel()
    x: dict[tuple[str, int, str], int] = {}
    for d in range(days):
        for m in meals:
            for r in by_meal[m]:
                x[r, d, m] = mdl.var(f"x[{r},{d},{m}]")
            mdl.add({x[r, d, m]: 1 for r in by_meal[m]}, 1, 1)          # one dish per slot

    for r in shared:
        slots = [(d, m) for d in range(days) for m in meals if (r, d, m) in x]
        for d in range(days):
            mdl.add({x[r, d, m]: 1 for (dd, m) in slots if dd == d}, hi=1)   # not twice a day
        mdl.add({x[r, d, m]: 1 for d, m in slots}, hi=max_repeats)
        extra = mdl.var(f"repeat[{r}]", 0, max_repeats)
        mdl.add({extra: 1, **{x[r, d, m]: -1 for d, m in slots}}, lo=-1)   # extra >= uses - 1
        mdl.add_objective(extra, -W_REPEAT)

    # Ingredient flow.
    used_ings = {i for r in shared for i in kb.recipes[r].ingredients}
    tracked = {i for i in used_ings if kb.ingredients[i].is_produce or not staples_unlimited} | set(pantry)
    pantry_vars: dict[tuple[str, int], int] = {}
    buy_vars: dict[tuple[str, int], int] = {}
    for i in sorted(tracked):
        item = pantry.get(i)
        avail = int(item.quantity_g) if item else 0
        shelf = item.shelf_life_days if item else 0
        per_day_max = sum(round(g * servings) for r in shared
                          for ing, g in kb.recipes[r].ingredients.items() if ing == i) or 1
        is_produce = kb.ingredients[i].is_produce
        for d in range(days):
            usable = avail if (d < shelf or not is_produce) else 0
            p = mdl.var(f"p[{i},{d}]", 0, min(usable, per_day_max * len(meals)))
            b = mdl.var(f"b[{i},{d}]", 0, per_day_max * len(meals) if allow_buying else 0)
            pantry_vars[i, d], buy_vars[i, d] = p, b
            need = {x[r, d, m]: -round(kb.recipes[r].ingredients[i] * servings)
                    for (r, dd, m) in x if dd == d and i in kb.recipes[r].ingredients}
            mdl.add({p: 1, b: 1, **need}, 0, 0)                          # p + b = demand
            if is_produce and item:
                urgency = 10 + round(10 * max(0.0, shelf - d) / max(shelf, 1))
                mdl.add_objective(p, urgency)
            else:
                mdl.add_objective(p, 1)
            mdl.add_objective(b, -W_BUY)
        if avail:
            mdl.add({pantry_vars[i, d]: 1 for d in range(days)}, hi=avail)

    # Calories per person per day, soft band around target.
    kcal = {r: round(kb.recipe_nutrition(r).kcal) for r in shared}
    lo_t, hi_t = profile.kcal_target * (1 - KCAL_BAND), profile.kcal_target * (1 + KCAL_BAND)
    for d in range(days):
        day_kcal = {x[r, dd, m]: kcal[r] for (r, dd, m) in x if dd == d}
        under = mdl.var(f"under[{d}]", 0, int(hi_t))
        over = mdl.var(f"over[{d}]", 0, 10_000)
        mdl.add({under: 1, **day_kcal}, lo=lo_t)                          # under >= lo - kcal
        mdl.add({over: -1, **day_kcal}, hi=hi_t)                          # over  >= kcal - hi
        mdl.add_objective(under, -W_KCAL)
        mdl.add_objective(over, -W_KCAL)

    sol = mdl.solve(time_limit=time_limit, prefer=solver)
    if not sol.ok:
        hint = "" if allow_buying else " Try allowing the planner to add missing items to a shopping list."
        raise PlanningError(f"The solver could not find a valid plan ({sol.status}).{hint}")

    v = sol.values
    schedule = [{m: next(r for r in by_meal[m] if v[mdl.names[x[r, d, m]]] == 1) for m in meals}
                for d in range(days)]
    pantry_daily = {i: [v[mdl.names[pantry_vars[i, d]]] for d in range(days)] for i in tracked}
    bought_daily = {i: [v[mdl.names[buy_vars[i, d]]] for d in range(days)] for i in tracked}
    daily = []
    for meals_today in schedule:
        total = Nutrition()
        for rid in meals_today.values():
            total = total + kb.recipe_nutrition(rid)
        daily.append(total)

    return PlanResult(schedule, daily, pantry_daily, bought_daily, dict(pantry), servings,
                      sol.solver, sol.status, sol.seconds, max_repeats, trace, allowed)


def validate_plan(kb: KnowledgeBase, plan: PlanResult, *, allow_buying: bool = True) -> list[str]:
    """Re-check every hard constraint independently of the solver. Empty list = valid."""
    problems = []
    allowed = set(plan.allowed_recipes)
    counts: dict[str, int] = {}
    for d, meals in enumerate(plan.schedule):
        if len(set(meals.values())) != len(meals):
            problems.append(f"Day {d + 1}: same dish served twice")
        for m, rid in meals.items():
            if rid not in allowed:
                problems.append(f"Day {d + 1} {m}: {rid} breaks a diet rule")
            if m not in kb.recipes[rid].meal_types:
                problems.append(f"Day {d + 1} {m}: {rid} is not a {m} dish")
            counts[rid] = counts.get(rid, 0) + 1
    for rid, n in counts.items():
        if n > plan.max_repeats:
            problems.append(f"{rid} used {n} times (limit {plan.max_repeats})")

    demand = _demand(kb, plan.schedule, plan.servings)
    for i, need in demand.items():
        if i not in plan.pantry_daily:
            continue                                   # untracked staple
        for d, grams in enumerate(need):
            got = plan.pantry_daily[i][d] + plan.bought_daily[i][d]
            if got != grams:
                problems.append(f"Day {d + 1}: {i} supplied {got} g but needs {grams} g")
    for i, item in plan.pantry.items():
        used = plan.pantry_used(i)
        if used > item.quantity_g + 1e-6:
            problems.append(f"{i}: used {used} g from pantry but only {item.quantity_g:.0f} g available")
        if kb.ingredients[i].is_produce:
            for d, g in enumerate(plan.pantry_daily.get(i, [])):
                if g and d >= item.shelf_life_days:
                    problems.append(f"{i}: used on day {d + 1} after it is expected to spoil")
    if not allow_buying and plan.shopping_list:
        problems.append("items were bought although buying is disabled")
    return problems
