"""CrewAI agents that explain and present the plan.

Division of labour: the constraint solver decides *what* is cooked and *how much*
of each ingredient is used, because LLMs are unreliable at respecting quantities.
The agents reason over the solver's output and the knowledge base to explain it,
flag risks and turn it into practical guidance. Their prompts forbid changing the
dishes or inventing numbers, and the nutritionist must look facts up through a
knowledge-base tool.
"""
from __future__ import annotations

from nutrivision.knowledge.kb import KnowledgeBase
from nutrivision.planning.planner import PlanResult
from nutrivision.reasoning import explain
from nutrivision.reasoning.rules import UserProfile


def crewai_available() -> bool:
    try:
        import crewai  # noqa: F401
        return True
    except ImportError:
        return False


def _make_tool(kb: KnowledgeBase):
    try:
        from crewai.tools import tool
    except ImportError:                     # older CrewAI releases
        from crewai_tools import tool

    @tool("Ingredient knowledge base lookup")
    def ingredient_facts(name: str) -> str:
        """Look up verified nutrition facts for one ingredient by name (e.g. 'spinach').
        Always use this instead of relying on memory for nutrient values."""
        ing = kb.resolve(name)
        if not ing:
            return f"'{name}' is not in the knowledge base."
        return kb.facts(ing)

    return ingredient_facts


def run_crew(kb: KnowledgeBase, plan: PlanResult, profile: UserProfile, *,
             model: str, api_key: str | None = None, temperature: float = 0.4) -> dict[str, str]:
    from crewai import LLM, Agent, Crew, Process, Task

    llm = LLM(model=model, api_key=api_key or None, temperature=temperature)
    lookup = _make_tool(kb)

    pantry_lines = []
    for i, item in plan.pantry.items():
        used = plan.pantry_used(i)
        pantry_lines.append(f"- {kb.ingredients[i].name}: have {explain.format_quantity(kb, i, item.quantity_g)}, "
                            f"plan uses {explain.format_quantity(kb, i, used)}, "
                            f"expected to keep ~{item.shelf_life_days:.0f} more days")
    shopping = ", ".join(f"{kb.ingredients[i].name} {explain.format_quantity(kb, i, g, to_buy=True)}"
                         for i, g in plan.shopping_list.items()) or "nothing"
    excluded = "\n".join(f"- {f.recipe_name}: {f.reason}" for f in plan.rule_trace[:15]) or "none"
    avg = explain.weekly_average(plan)
    profile_text = (f"Diet: {profile.diet}. Allergies: {', '.join(profile.allergies) or 'none'}. "
                    f"Diabetic-friendly: {'yes' if profile.diabetic_friendly else 'no'}. "
                    f"Household: {profile.household_size}. Target ~{profile.kcal_target} kcal per person "
                    f"per day from three meals.")
    guard = ("Do not change, add or remove any dish, and do not invent quantities or nutrient numbers; "
             "use only the figures given here or returned by the lookup tool. Write in plain, warm prose "
             "with short paragraphs and no tables.")

    analyst = Agent(
        role="Pantry Analyst",
        goal="Help the household waste as little food as possible",
        backstory="You run inventory for a busy home kitchen and know how long produce really lasts.",
        llm=llm, allow_delegation=False, verbose=False)
    nutritionist = Agent(
        role="Nutritionist",
        goal="Explain clearly and accurately what the week's food does for the household's health",
        backstory="A registered dietitian who explains nutrition without jargon or exaggerated claims.",
        llm=llm, tools=[lookup], allow_delegation=False, verbose=False)
    chef = Agent(
        role="Home Chef",
        goal="Make the weekly plan easy to cook on weeknights",
        backstory="An experienced home cook who plans prep so weekday cooking takes under 40 minutes.",
        llm=llm, allow_delegation=False, verbose=False)

    pantry_task = Task(
        description=(f"Household profile: {profile_text}\n\nPantry and planned usage:\n"
                     + "\n".join(pantry_lines) + f"\n\nShopping list: {shopping}\n\n"
                     "Write a short briefing: which items to use first and why, anything at risk of "
                     "spoiling, and what to do with leftovers. " + guard),
        expected_output="Two or three short paragraphs.", agent=analyst)
    nutrition_task = Task(
        description=(f"Household profile: {profile_text}\n\nWeekly plan:\n{explain.plan_as_text(kb, plan)}\n\n"
                     f"Weekly average per person: {avg.kcal:.0f} kcal, {avg.protein_g:.0f} g protein, "
                     f"{avg.carbs_g:.0f} g carbs, {avg.fat_g:.0f} g fat, {avg.fiber_g:.0f} g fibre.\n"
                     f"Dishes excluded by diet rules:\n{excluded}\n\n"
                     "Use the lookup tool for the main pantry ingredients, then explain their key nutritional "
                     "benefits, how the week compares with the targets, and two or three simple additions "
                     "(snacks or sides) that would close any gaps. " + guard),
        expected_output="A clear explanation in four to six short paragraphs.", agent=nutritionist,
        context=[pantry_task])
    chef_task = Task(
        description=(f"Weekly plan for {plan.servings} people:\n{explain.plan_as_text(kb, plan)}\n\n"
                     "Give practical cooking guidance: a Sunday prep list, which components can be batch "
                     "cooked and reused (for example dal or chopped onion-tomato masala), and one tip per "
                     "day. " + guard),
        expected_output="A prep list followed by one short tip for each day.", agent=chef,
        context=[pantry_task, nutrition_task])

    crew = Crew(agents=[analyst, nutritionist, chef], tasks=[pantry_task, nutrition_task, chef_task],
                process=Process.sequential, verbose=False)
    crew.kickoff()

    def text(task) -> str:
        out = getattr(task, "output", None)
        return getattr(out, "raw", None) or str(out or "")

    return {"pantry": text(pantry_task), "nutrition": text(nutrition_task), "chef": text(chef_task)}
