"""NutriVision Streamlit app.  Run with:  streamlit run app.py"""
from __future__ import annotations

# Use the Mac's own certificate store, so downloads work behind antivirus, VPNs and campus Wi-Fi.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import os

import cv2
import pandas as pd
import streamlit as st

from nutrivision import config
from nutrivision.agents.crew import crewai_available, run_crew
from nutrivision.knowledge.kb import get_kb
from nutrivision.planning.planner import PantryItem, PlanningError, plan_week, validate_plan
from nutrivision.reasoning import explain
from nutrivision.reasoning.rules import ALLERGY_RULES, DIET_RULES, UserProfile, allowed_ingredients
from nutrivision.services.youtube import search_link, search_recipe_videos
from nutrivision.vision.preprocess import load_image, preprocess
from nutrivision.vision.segmentation import SHELF_FACTOR

st.set_page_config(page_title="NutriVision", page_icon="🥕", layout="wide")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Source Sans 3', system-ui, sans-serif; }
  h1, h2, h3 { font-family: 'Bricolage Grotesque', system-ui, sans-serif; letter-spacing: -0.01em; }
  h1 { color: #1F4D3A; }
  .nv-lede { font-size: 1.15rem; color: #4A5A52; max-width: 62ch; margin-top: -0.6rem; }
  .nv-meal { border-left: 4px solid #E3A72F; padding: 0.2rem 0 0.2rem 0.9rem; margin: 0.6rem 0 1rem; }
  .nv-meal small { color: #6B7A72; }
</style>
""", unsafe_allow_html=True)

kb = get_kb()
PRODUCE = sorted(kb.produce(), key=lambda i: i.name)
NAME_TO_ID = {i.name: i.id for i in PRODUCE}
STAGES = list(SHELF_FACTOR)


@st.cache_resource(show_spinner="Loading the detection model...")
def load_detector(weights: str, conf: float, iou: float):
    from nutrivision.vision.detector import ProduceDetector
    return ProduceDetector(kb, weights=weights, conf=conf, iou=iou)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def videos_for(query: str, key: str):
    try:
        return search_recipe_videos(query, key)
    except Exception:
        return ()


def empty_inventory() -> pd.DataFrame:
    return pd.DataFrame({"Ingredient": pd.Series(dtype="str"), "Count": pd.Series(dtype="float"),
                         "Condition": pd.Series(dtype="str"), "Detected": pd.Series(dtype="float"),
                         "Colour estimate": pd.Series(dtype="float")})


if "inventory" not in st.session_state:
    st.session_state.inventory = empty_inventory()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Household")
    diet = st.selectbox("Diet", list(DIET_RULES), index=list(DIET_RULES).index("vegetarian"))
    allergies = st.multiselect("Allergies or intolerances", list(ALLERGY_RULES))
    diabetic = st.checkbox("Diabetic-friendly (limit high-GI dishes)")
    dislikes = st.multiselect("Ingredients to avoid", sorted(i.name for i in kb.ingredients.values()))
    household = st.number_input("People eating", 1, 10, 2)
    kcal_target = st.slider("Calories per person from the three meals", 1000, 3000, 1400, 50)

    st.header("Planning")
    staples_unlimited = st.toggle("Assume staples are in stock", value=True,
                                  help="Rice, flour, dal, oil, dairy and so on. Turn off to enter amounts.")
    allow_buying = st.toggle("Add missing items to a shopping list", value=True)
    max_repeats = st.slider("Times a dish may repeat per week", 1, 4, 2)

    st.header("Detection")
    weights = st.text_input("YOLO weights", config.YOLO_WEIGHTS)
    conf = st.slider("Confidence threshold", 0.1, 0.9, config.DETECTION_CONFIDENCE, 0.05)
    enhance = st.checkbox("Correct lighting before detection", value=False)

    st.header("AI agents and videos")
    llm_model = st.text_input("LLM model (LiteLLM name)", config.LLM_MODEL)
    llm_key = st.text_input("LLM API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    yt_key = st.text_input("YouTube API key", type="password", value=config.YOUTUBE_API_KEY)

profile = UserProfile(diet=diet, allergies=allergies, diabetic_friendly=diabetic,
                      dislikes=[kb.resolve(n) for n in dislikes], household_size=int(household),
                      kcal_target=int(kcal_target))

# ---------------------------------------------------------------- header
st.title("NutriVision")
st.markdown('<p class="nv-lede">Photograph your fruit and vegetables, check what was found, '
            'and get a week of meals that uses them before they spoil.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- step 1: detect
st.subheader("1. Photograph your produce")
upload = st.file_uploader("Upload a photo of your produce", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    img = load_image(upload.getvalue())
    work = preprocess(img, white_balance=enhance, clahe=enhance)
    col_a, col_b = st.columns(2)
    col_a.image(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), caption="Your photo (after lighting correction)"
                if enhance else "Your photo")
    if st.button("Detect produce", type="primary"):
        try:
            detector = load_detector(weights, conf, config.DETECTION_IOU)
        except ImportError:
            st.error("The ultralytics package isn't installed, so detection is unavailable. "
                     "Run pip install -r requirements.txt, or add items by hand below.")
        except Exception as exc:
            st.error(f"Couldn't load the model from '{weights}': {exc}")
        else:
            with st.spinner("Finding fruit and vegetables..."):
                res = detector.detect(work)
            st.session_state.annotated = cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB)
            st.session_state.ignored = res.ignored_labels
            st.session_state.supported = detector.supported
            st.session_state.inventory = pd.DataFrame([{
                "Ingredient": r.name, "Count": float(r.count), "Condition": r.main_stage,
                "Detected": float(r.count), "Colour estimate": float(r.colour_count or 0),
            } for r in res.inventory]) if res.inventory else empty_inventory()
            st.session_state.pop("plan", None)
    if "annotated" in st.session_state:
        col_b.image(st.session_state.annotated, caption="Detections, labelled with estimated condition")
        if st.session_state.get("supported") and len(st.session_state.supported) < 8:
            st.caption("This model only recognises " + ", ".join(st.session_state.supported) +
                       ". Add anything else below, or train custom weights with training/train_yolo.py.")

# ---------------------------------------------------------------- step 2: confirm pantry
st.subheader("2. Check your pantry")
st.caption("Fix any counts the camera got wrong and add items it missed. Condition changes how soon "
           "an item is used: unripe items keep longer, overripe ones go first.")
edited = st.data_editor(
    st.session_state.inventory, num_rows="dynamic", hide_index=True, width="stretch",
    key="inventory_editor",
    column_config={
        "Ingredient": st.column_config.SelectboxColumn(options=list(NAME_TO_ID), required=True),
        "Count": st.column_config.NumberColumn(min_value=0.0, step=0.5, required=True,
                                               help="Pieces, or bunches/heads for greens and cauliflower"),
        "Condition": st.column_config.SelectboxColumn(options=STAGES, default="fresh"),
        "Detected": st.column_config.NumberColumn("Detected by YOLO", disabled=True),
        "Colour estimate": st.column_config.NumberColumn("Colour-segmentation count", disabled=True),
    })

staple_amounts: dict[str, float] = {}
if not staples_unlimited:
    st.caption("Staple amounts in grams")
    cols = st.columns(4)
    for n, ing in enumerate(sorted(kb.staples(), key=lambda i: i.name)):
        staple_amounts[ing.id] = cols[n % 4].number_input(ing.name, 0, 20000, 1000, 100, key=f"st_{ing.id}")

pantry: dict[str, PantryItem] = {}
for _, row in edited.dropna(subset=["Ingredient"]).iterrows():
    ing_id = NAME_TO_ID.get(row["Ingredient"])
    count = 0.0 if pd.isna(row.get("Count")) else float(row["Count"])
    if not ing_id or count <= 0:
        continue
    ing = kb.ingredients[ing_id]
    condition = row.get("Condition")
    factor = SHELF_FACTOR.get(condition if isinstance(condition, str) else "fresh", 1.0)
    prev = pantry.get(ing_id)
    grams = count * ing.unit_weight_g + (prev.quantity_g if prev else 0)
    shelf = ing.shelf_life_days * factor if not prev else min(prev.shelf_life_days, ing.shelf_life_days * factor)
    pantry[ing_id] = PantryItem(ing_id, grams, shelf)
for ing_id, grams in staple_amounts.items():
    pantry[ing_id] = PantryItem(ing_id, grams, kb.ingredients[ing_id].shelf_life_days)

unusable = [kb.ingredients[i].name for i in pantry if i not in allowed_ingredients(kb, profile)]
if unusable:
    st.info(f"Your diet settings exclude {', '.join(unusable)}, so the plan won't use them.")

# ---------------------------------------------------------------- step 3: plan
st.subheader("3. Plan the week")
if st.button("Create meal plan", type="primary"):
    try:
        with st.spinner("Solving the meal plan..."):
            plan = plan_week(kb, profile, pantry, staples_unlimited=staples_unlimited,
                             allow_buying=allow_buying, max_repeats=max_repeats)
        st.session_state.plan = plan
        st.session_state.plan_problems = validate_plan(kb, plan, allow_buying=allow_buying)
        st.session_state.pop("crew", None)
    except PlanningError as exc:
        st.error(str(exc))

plan = st.session_state.get("plan")
if plan:
    st.write(explain.summary_paragraph(kb, plan, profile.kcal_target))
    problems = st.session_state.get("plan_problems", [])
    if problems:
        st.warning("Validation found problems: " + "; ".join(problems))
    else:
        st.caption(f"Solved with {plan.solver} in {plan.seconds:.1f} s ({plan.status}). "
                   "An independent check confirmed every hard constraint is met.")

    tabs = st.tabs(config.DAYS[:len(plan.schedule)])
    for d, tab in enumerate(tabs):
        with tab:
            for meal, rid in plan.schedule[d].items():
                rec = kb.recipes[rid]
                n = kb.recipe_nutrition(rid)
                st.markdown(f'<div class="nv-meal"><strong>{meal.capitalize()}: {rec.name}</strong><br>'
                            f'{rec.description}<br><small>Per person about {n.kcal:.0f} kcal, '
                            f'{n.protein_g:.0f} g protein, {n.fiber_g:.0f} g fibre</small></div>',
                            unsafe_allow_html=True)
                with st.expander("Ingredients and videos"):
                    st.write(", ".join(f"{kb.ingredients[i].name} {explain.format_quantity(kb, i, g * plan.servings)}"
                                       for i, g in rec.ingredients.items()))
                    vids = videos_for(rec.video_query, yt_key) if yt_key else ()
                    if vids:
                        st.video(vids[0].url)
                        for v in vids[1:]:
                            st.markdown(f"[{v.title}]({v.url}) from {v.channel}")
                    else:
                        st.markdown(f"[Search YouTube for {rec.name.lower()}]({search_link(rec.video_query)})")
            dn = plan.daily_nutrition[d]
            st.caption(f"Day total per person: {dn.kcal:.0f} kcal, {dn.protein_g:.0f} g protein, "
                       f"{dn.carbs_g:.0f} g carbs, {dn.fat_g:.0f} g fat, {dn.fiber_g:.0f} g fibre")

    chart = pd.DataFrame({"Calories": [n.kcal for n in plan.daily_nutrition],
                          "Target": [profile.kcal_target] * len(plan.daily_nutrition)},
                         index=[d[:3] for d in config.DAYS[:len(plan.schedule)]])
    st.markdown("**Calories per person each day**")
    st.line_chart(chart)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shopping list**")
        if plan.shopping_list:
            st.dataframe(pd.DataFrame([{"Item": kb.ingredients[i].name,
                                        "Amount": explain.format_quantity(kb, i, g, to_buy=True)}
                                       for i, g in plan.shopping_list.items()]),
                         hide_index=True, width="stretch")
        else:
            st.write("Nothing to buy. The plan runs entirely on what you have.")
    with col2:
        st.markdown("**Leftovers**")
        notes = explain.pantry_notes(kb, plan)
        st.write("\n\n".join(notes) if notes else "Everything in your pantry gets used.")

    with st.expander("Why some dishes were left out"):
        if plan.rule_trace:
            for f in plan.rule_trace:
                st.write(f"{f.recipe_name}: {f.reason}.")
        else:
            st.write("No dishes were excluded by your settings.")

    with st.expander("What your produce does for you"):
        for b in explain.ingredient_benefits(kb, [i for i in pantry if kb.ingredients[i].is_produce]):
            st.write(f"{b['ingredient']} provides {b['key_nutrients']}. {b['why_it_helps']}")
        st.caption("Nutrition figures are approximate values per 100 g adapted from USDA FoodData Central.")

    st.subheader("4. Ask the AI agents")
    if not crewai_available():
        st.caption("Install crewai to get a pantry briefing, a nutritionist's review and a cooking guide.")
    elif st.button("Get advice from the agents"):
        try:
            with st.spinner("The pantry analyst, nutritionist and chef are reviewing your plan..."):
                st.session_state.crew = run_crew(kb, plan, profile, model=llm_model, api_key=llm_key)
        except Exception as exc:
            st.error(f"The agents couldn't run: {exc}. Check the model name and API key in the sidebar.")
    if "crew" in st.session_state:
        out = st.session_state.crew
        t1, t2, t3 = st.tabs(["Pantry analyst", "Nutritionist", "Chef"])
        t1.markdown(out["pantry"])
        t2.markdown(out["nutrition"])
        t3.markdown(out["chef"])
