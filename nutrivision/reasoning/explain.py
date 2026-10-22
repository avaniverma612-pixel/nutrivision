"""Template-based explanations built straight from the knowledge base.

These work with no LLM at all, and the same facts are handed to the CrewAI agents
as grounding so they explain the plan rather than invent nutrition numbers.
"""
from __future__ import annotations

import math

from nutrivision import config
from nutrivision.knowledge.kb import KnowledgeBase, Nutrition
from nutrivision.planning.planner import PlanResult


def format_quantity(kb: KnowledgeBase, ingredient_id: str, grams: float, *, to_buy: bool = False) -> str:
    """Grams -> '3 pieces (~300 g)'. With to_buy, round up to whole items you can actually buy."""
    ing = kb.ingredients[ingredient_id]
    if ing.is_produce and ing.unit_weight_g > 1:
        units = grams / ing.unit_weight_g
        if to_buy:
            units = max(1, math.ceil(units - 0.05))
        shown = f"{units:.1f}".rstrip("0").rstrip(".")
        plural = "es" if ing.unit == "bunch" and units != 1 else ("s" if units != 1 else "")
        return f"{shown} {ing.unit}{plural} (~{grams:.0f} g)"
    return f"{grams:.0f} g"


def ingredient_benefits(kb: KnowledgeBase, ingredient_ids: list[str]) -> list[dict]:
    return [{"ingredient": kb.ingredients[i].name,
             "key_nutrients": ", ".join(kb.ingredients[i].key_nutrients),
             "why_it_helps": kb.ingredients[i].benefit} for i in ingredient_ids]


def weekly_average(plan: PlanResult) -> Nutrition:
    total = Nutrition()
    for n in plan.daily_nutrition:
        total = total + n
    return total.scaled(1 / max(len(plan.daily_nutrition), 1))


def pantry_notes(kb: KnowledgeBase, plan: PlanResult) -> list[str]:
    notes = []
    for i, item in plan.pantry.items():
        left = item.quantity_g - plan.pantry_used(i)
        if left <= 1:
            continue
        name = kb.ingredients[i].name.lower()
        if item.shelf_life_days <= 7:
            notes.append(f"About {format_quantity(kb, i, left)} of {name} is left over and will likely "
                         f"spoil within the week. Consider freezing it or using it in a snack.")
        else:
            notes.append(f"{format_quantity(kb, i, left).capitalize()} of {name} remains and keeps for "
                         f"about {item.shelf_life_days:.0f} days, so it can carry into next week.")
    return notes


def plan_as_text(kb: KnowledgeBase, plan: PlanResult) -> str:
    lines = []
    for d, meals in enumerate(plan.schedule):
        n = plan.daily_nutrition[d]
        dishes = "; ".join(f"{m}: {kb.recipes[r].name}" for m, r in meals.items())
        lines.append(f"{config.DAYS[d % 7]} - {dishes} (per person ~{n.kcal:.0f} kcal, "
                     f"{n.protein_g:.0f} g protein, {n.fiber_g:.0f} g fibre)")
    return "\n".join(lines)


def summary_paragraph(kb: KnowledgeBase, plan: PlanResult, kcal_target: int) -> str:
    avg = weekly_average(plan)
    dishes = {r for day in plan.schedule for r in day.values()}
    text = (f"The plan uses {len(dishes)} different dishes over the week and draws on "
            f"{plan.produce_utilisation:.0%} of the produce you have. Each person gets on average "
            f"about {avg.kcal:.0f} kcal from the three main meals (target {kcal_target}), with "
            f"{avg.protein_g:.0f} g protein and {avg.fiber_g:.0f} g fibre a day.")
    if avg.kcal < kcal_target * 0.9:
        text += (" Meals come in under the calorie target, so add snacks such as fruit, nuts "
                 "or buttermilk, or increase portion sizes.")
    if plan.shopping_list:
        text += f" You'll need to buy {len(plan.shopping_list)} items to complete it."
    return text
