"""
AthleteEdge AI — Smart Meal Planner (v2)
==========================================
XGBoost predicts macro targets -> Groq LLM composes meals from OUR food
database within those targets -> output is validated.
Falls back to offline greedy planner if Groq is unavailable.
"""

import os
import json
import traceback
from itertools import combinations
from dotenv import load_dotenv
from groq import Groq
from offline import load_models, predict_targets
from fooddb import get_food_db

load_dotenv()

_groq_client = None

BUDGET_LIMITS = {"low": 80, "medium": 150, "high": 250}

STANDARD_SLOTS = [
    {"slot": "Breakfast", "tag": "breakfast", "cal_frac": 0.24, "protein_frac": 0.22, "min_items": 2, "max_items": 4},
    {"slot": "Pre-Workout", "tag": "pre_workout", "cal_frac": 0.11, "protein_frac": 0.08, "min_items": 1, "max_items": 2},
    {"slot": "Lunch", "tag": "lunch", "cal_frac": 0.31, "protein_frac": 0.31, "min_items": 2, "max_items": 4},
    {"slot": "Post-Workout", "tag": "post_workout", "cal_frac": 0.14, "protein_frac": 0.19, "min_items": 1, "max_items": 3},
    {"slot": "Dinner", "tag": "dinner", "cal_frac": 0.20, "protein_frac": 0.20, "min_items": 2, "max_items": 4},
]

RECOVERY_SLOTS = [
    {"slot": "Breakfast", "tag": "breakfast", "cal_frac": 0.24, "protein_frac": 0.22, "min_items": 2, "max_items": 4},
    {"slot": "Recovery Hydration", "tag": "recovery", "cal_frac": 0.10, "protein_frac": 0.08, "min_items": 1, "max_items": 3},
    {"slot": "Lunch", "tag": "lunch", "cal_frac": 0.32, "protein_frac": 0.32, "min_items": 2, "max_items": 4},
    {"slot": "Evening Recovery", "tag": "recovery", "cal_frac": 0.12, "protein_frac": 0.14, "min_items": 1, "max_items": 3},
    {"slot": "Dinner", "tag": "dinner", "cal_frac": 0.22, "protein_frac": 0.24, "min_items": 2, "max_items": 4},
]

RECOVERY_FOODS = {
    "haldi doodh (turmeric milk)",
    "coconut water (1 glass)",
    "chaas (buttermilk)",
    "lassi (salted)",
    "pomegranate (arils)",
    "papaya (ripe)",
    "dal palak (lentil spinach)",
    "palak (spinach, cooked)",
    "mackerel / bangda fish",
}

SPORT_FOOD_HINTS = {
    "Cricket": {"banana", "sattu", "coconut water", "chicken", "curd", "egg"},
    "Football": {"banana", "rice", "coconut water", "chaas", "sweet potato", "sattu"},
    "Kabaddi": {"paneer", "soybean", "chicken", "egg", "dal", "milk"},
    "Athletics": {"banana", "dates", "oats", "sweet potato", "coconut water", "curd"},
    "Wrestling": {"paneer", "egg", "chicken", "soybean", "milk", "dal"},
    "Badminton": {"banana", "poha", "curd", "coconut water", "rice", "sprouts"},
}


def _get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=key)
    return _groq_client


# =========================================================================
#  BUILD FOOD LIST FOR LLM (from our actual database)
# =========================================================================

def _build_food_list(diet_type: str) -> str:
    """Build a categorized food list from OUR database for the LLM prompt."""
    foods = get_food_db()
    by_cat = {}
    for f in foods:
        if diet_type == "veg" and not f["is_veg"]:
            continue
        cat = f["category"]
        cost = f.get("cost_inr", 0)
        entry = f"{f['name']} ({f['cal']}kcal, P:{f['protein_g']}g, C:{f['carbs_g']}g, F:{f['fat_g']}g, Rs.{cost})"
        by_cat.setdefault(cat, []).append(entry)

    lines = []
    for cat in sorted(by_cat):
        lines.append(f"[{cat.upper()}] " + " | ".join(by_cat[cat]))
    return "\n".join(lines)


# =========================================================================
#  CONSTRAINT OPTIMIZER (foods are chosen before Groq writes explanations)
# =========================================================================

def _food_name(food: dict) -> str:
    return food["name"].strip().lower()


def _food_matches(food: dict, terms: set[str]) -> bool:
    name = _food_name(food)
    return any(term in name for term in terms)


def _sum_combo(combo: tuple[dict, ...]) -> dict:
    return {
        "cal": sum(f["cal"] for f in combo),
        "protein": sum(f["protein_g"] for f in combo),
        "carbs": sum(f["carbs_g"] for f in combo),
        "fat": sum(f["fat_g"] for f in combo),
        "cost": sum(f.get("cost_inr", 0) for f in combo),
        "categories": {f["category"] for f in combo},
        "names": {f["name"] for f in combo},
    }


def _explain_meal(meal: dict, profile: dict, targets: dict) -> str:
    slot = meal["slot"]
    names = ", ".join(meal["items"])
    sport = profile["sport"]

    if profile.get("is_recovery_day"):
        return (
            f"{slot} keeps training stress low while supporting repair: {names} "
            f"adds protein, fluids, and anti-inflammatory recovery support."
        )
    if slot == "Pre-Workout":
        return f"Timed for {sport}: quick carbs and light digestion before the session."
    if slot == "Post-Workout":
        return f"Rebuilds after {sport} with protein plus carbs to refill glycogen."
    return (
        f"Balances the {targets['calories']} kcal and {targets['protein_g']}g protein "
        f"targets with foods that suit {sport} training."
    )


def _build_tip(plan: dict, profile: dict, targets: dict) -> str:
    validation = plan.get("validation", {})
    if profile.get("is_recovery_day") or profile.get("fatigue_score", 0) >= 7:
        return (
            "High fatigue detected: prioritize sleep, electrolytes, coconut water or chaas, "
            "and anti-inflammatory foods like turmeric milk, dal palak, papaya, or pomegranate."
        )
    if profile.get("is_match_day"):
        return "Match-day focus: keep fluids steady, use quick carbs before play, and keep post-session protein within 60 minutes."
    return (
        f"Optimized to land close to target: {validation.get('actual_cal', plan['daily_total']['cal'])} kcal "
        f"and {validation.get('actual_protein', plan['daily_total']['protein'])}g protein."
    )


def _combo_score(combo: tuple[dict, ...], slot: dict, targets: dict, profile: dict,
                 budget_cap: int) -> float:
    totals = _sum_combo(combo)
    slot_cal = targets["calories"] * slot["cal_frac"]
    slot_pro = targets["protein_g"] * slot["protein_frac"]
    cal_error = abs(totals["cal"] - slot_cal) / max(slot_cal, 1)
    pro_error = abs(totals["protein"] - slot_pro) / max(slot_pro, 1)
    cost_pressure = totals["cost"] / max(budget_cap, 1)
    diversity_bonus = len(totals["categories"]) * 0.08

    recovery_bonus = 0.0
    if profile.get("is_recovery_day") or profile.get("fatigue_score", 0) >= 7:
        recovery_bonus = sum(0.18 for food in combo if _food_name(food) in RECOVERY_FOODS)

    sport_terms = SPORT_FOOD_HINTS.get(profile.get("sport"), set())
    sport_bonus = sum(0.07 for food in combo if _food_matches(food, sport_terms))

    return (
        (cal_error * 5.0)
        + (pro_error * 1.8)
        + (cost_pressure * 0.8)
        - diversity_bonus
        - recovery_bonus
        - sport_bonus
    )


def _candidate_combos(foods: list[dict], slot: dict, targets: dict, profile: dict,
                      budget_cap: int) -> list[dict]:
    eligible = [
        food for food in foods
        if slot["tag"] in food["meal_slots"]
        and (profile["diet_type"] != "veg" or food["is_veg"])
    ]
    slot_cal = targets["calories"] * slot["cal_frac"]
    slot_pro = targets["protein_g"] * slot["protein_frac"]
    sport_terms = SPORT_FOOD_HINTS.get(profile.get("sport"), set())

    def relevance(food: dict) -> float:
        cal_fit = abs(food["cal"] - slot_cal / max(slot["min_items"] + 1, 1)) / max(slot_cal, 1)
        pro_fit = abs(food["protein_g"] - slot_pro / max(slot["min_items"] + 1, 1)) / max(slot_pro, 1)
        recovery_bonus = 0.0
        if profile.get("is_recovery_day") or profile.get("fatigue_score", 0) >= 7:
            recovery_bonus = 0.7 if _food_name(food) in RECOVERY_FOODS else 0.0
        sport_bonus = 0.25 if _food_matches(food, sport_terms) else 0.0
        protein_density = food["protein_g"] / max(food["cal"], 1)
        return protein_density + recovery_bonus + sport_bonus - cal_fit - pro_fit - (food.get("cost_inr", 0) / max(budget_cap, 1))

    eligible = sorted(eligible, key=relevance, reverse=True)[:34]

    combos = []
    for size in range(slot["min_items"], slot["max_items"] + 1):
        for combo in combinations(eligible, size):
            totals = _sum_combo(combo)
            if totals["cost"] > budget_cap:
                continue
            if totals["cal"] > targets["calories"] * 0.55:
                continue
            combos.append({
                "foods": combo,
                "totals": totals,
                "score": _combo_score(combo, slot, targets, profile, budget_cap),
            })

    combos.sort(key=lambda item: item["score"])
    return combos[:260]


def _plan_score(plan: list[dict], targets: dict, budget_cap: int, profile: dict) -> float:
    total_cal = sum(meal["totals"]["cal"] for meal in plan)
    total_pro = sum(meal["totals"]["protein"] for meal in plan)
    total_cost = sum(meal["totals"]["cost"] for meal in plan)
    all_categories = set().union(*(meal["totals"]["categories"] for meal in plan))
    all_names = [name for meal in plan for name in meal["totals"]["names"]]

    cal_error = abs(total_cal - targets["calories"]) / max(targets["calories"], 1)
    pro_error = abs(total_pro - targets["protein_g"]) / max(targets["protein_g"], 1)
    budget_error = max(0, total_cost - budget_cap) / max(budget_cap, 1)
    repeat_penalty = len(all_names) - len(set(all_names))
    diversity_bonus = len(all_categories) * 0.035

    recovery_names = {_food_name(food) for meal in plan for food in meal["foods"]}
    recovery_bonus = 0.0
    if profile.get("is_recovery_day") or profile.get("fatigue_score", 0) >= 7:
        recovery_bonus = 0.18 * len(recovery_names & RECOVERY_FOODS)

    return (
        cal_error * 24.0
        + pro_error * 7.0
        + budget_error * 12.0
        + repeat_penalty * 5.0
        - diversity_bonus
        - recovery_bonus
    )


def _optimize_plan(profile: dict, targets: dict, budget: str) -> dict:
    foods = get_food_db()
    budget_cap = BUDGET_LIMITS.get(budget, BUDGET_LIMITS["medium"])
    slots = RECOVERY_SLOTS if profile.get("is_recovery_day") else STANDARD_SLOTS
    slot_candidates = [
        _candidate_combos(foods, slot, targets, profile, budget_cap)
        for slot in slots
    ]

    beam = [{"plan": [], "names": set(), "cost": 0, "score": 0.0}]
    for slot, candidates in zip(slots, slot_candidates):
        next_beam = []
        for state in beam:
            for candidate in candidates:
                if state["names"] & candidate["totals"]["names"]:
                    continue
                new_cost = state["cost"] + candidate["totals"]["cost"]
                if new_cost > budget_cap * 1.12:
                    continue
                new_plan = state["plan"] + [{**candidate, "slot": slot["slot"]}]
                next_beam.append({
                    "plan": new_plan,
                    "names": state["names"] | candidate["totals"]["names"],
                    "cost": new_cost,
                    "score": _plan_score(new_plan, targets, budget_cap, profile),
                })

        if not next_beam:
            raise RuntimeError(f"No valid meal combinations for {slot['slot']}")
        next_beam.sort(key=lambda state: state["score"])
        beam = next_beam[:160]

    best = min(beam, key=lambda state: _plan_score(state["plan"], targets, budget_cap, profile))
    meals = []
    for selected in best["plan"]:
        totals = selected["totals"]
        meal = {
            "slot": selected["slot"],
            "items": [food["name"] for food in selected["foods"]],
            "why": "",
            "approx_cal": round(totals["cal"]),
            "approx_protein": round(totals["protein"], 1),
            "approx_cost": round(totals["cost"]),
        }
        meal["why"] = _explain_meal(meal, profile, targets)
        meals.append(meal)

    plan = {
        "meals": meals,
        "daily_total": {
            "cal": sum(meal["approx_cal"] for meal in meals),
            "protein": round(sum(meal["approx_protein"] for meal in meals), 1),
            "cost": sum(meal["approx_cost"] for meal in meals),
            "tip": "",
        },
        "optimizer": {
            "budget_cap": budget_cap,
            "variety_score": round((len({item for meal in meals for item in meal["items"]}) / max(sum(len(meal["items"]) for meal in meals), 1)) * 100),
            "recovery_intelligence": bool(profile.get("is_recovery_day") or profile.get("fatigue_score", 0) >= 7),
            "sport_engine": profile.get("sport"),
        },
    }
    plan["daily_total"]["tip"] = _build_tip(plan, profile, targets)
    return plan



#  VALIDATE LLM OUTPUT AGAINST XGBOOST TARGETS
# =========================================================================

def _validate_plan(plan: dict, targets: dict) -> dict:
    """Check that LLM meal totals are within acceptable range of XGBoost targets."""
    meals = plan.get("meals", [])
    total_cal = sum(m.get("approx_cal", 0) for m in meals)
    total_pro = sum(m.get("approx_protein", 0) for m in meals)
    total_cost = sum(m.get("approx_cost", 0) for m in meals)

    target_cal = targets["calories"]
    target_pro = targets["protein_g"]

    cal_diff = abs(total_cal - target_cal) / target_cal if target_cal else 0
    pro_diff = abs(total_pro - target_pro) / target_pro if target_pro else 0

    plan["daily_total"] = plan.get("daily_total", {})
    plan["daily_total"]["cal"] = total_cal
    plan["daily_total"]["protein"] = total_pro
    plan["daily_total"]["cost"] = total_cost

    plan["validation"] = {
        "target_cal": target_cal,
        "actual_cal": total_cal,
        "cal_accuracy": f"{max(0, (1 - cal_diff) * 100):.0f}%",
        "target_protein": target_pro,
        "actual_protein": total_pro,
        "protein_accuracy": f"{max(0, (1 - pro_diff) * 100):.0f}%",
        "within_range": cal_diff <= 0.10 and pro_diff <= 0.15,
    }
    return plan


# =========================================================================
#  GROQ EXPLANATION LAYER
# =========================================================================

def _enhance_plan_with_groq(plan: dict, profile: dict, targets: dict,
                            budget_range: str) -> dict:
    """Ask Groq for reasoning only; foods and macro numbers stay locked."""
    locked_plan = {
        "meals": [
            {
                "slot": meal["slot"],
                "items": meal["items"],
                "approx_cal": meal["approx_cal"],
                "approx_protein": meal["approx_protein"],
                "approx_cost": meal.get("approx_cost", 0),
            }
            for meal in plan["meals"]
        ],
        "daily_total": plan["daily_total"],
    }

    prompt = f"""You are AthleteEdge AI's Indian sports nutrition reasoning layer.

The optimizer already selected this meal plan. Do not change any slot, food item, calories, protein, or cost.

ATHLETE:
- Sport: {profile['sport']} | Role: {profile.get('role', '')} | Day: {"Recovery Day" if profile.get('is_recovery_day') else "Match Day" if profile.get('is_match_day') else "Training Day"}
- Intensity: {profile['training_intensity']}/10 | Duration: {profile['session_duration_min']} min | Fatigue: {profile['fatigue_score']}/10
- Goal: {profile['goal']} | Diet: {profile['diet_type']} | Budget: {budget_range}/day

TARGETS:
- Calories: {targets['calories']} kcal
- Protein: {targets['protein_g']} g
- Carbs: {targets['carbs_g']} g | Fat: {targets['fat_g']} g

LOCKED OPTIMIZED PLAN:
{json.dumps(locked_plan, indent=2)}

Write concise, actual reasoning for every meal. Mention sport-specific fueling, calorie/protein fit, budget awareness, food variety, and recovery logic when fatigue is high.
If recovery day, do not mention pre-workout or post-workout. If fatigue is 7 or above, mention turmeric milk, electrolytes, coconut water, and anti-inflammatory foods only when relevant to the plan or as a tip.

Return exactly this JSON shape:
{{
  "meal_reasons": [
    {{"slot": "same slot name", "why": "one practical sentence"}},
    {{"slot": "same slot name", "why": "one practical sentence"}}
  ],
  "daily_tip": "one practical sentence"
}}"""

    client = _get_groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    reasoning = json.loads(response.choices[0].message.content)
    if isinstance(reasoning, list):
        reasoning = {"meal_reasons": reasoning}
    elif not isinstance(reasoning, dict):
        reasoning = {}

    reasons = {
        item.get("slot"): item.get("why", "")
        for item in reasoning.get("meal_reasons", [])
        if isinstance(item, dict) and item.get("slot") and item.get("why")
    }
    for meal in plan["meals"]:
        if reasons.get(meal["slot"]):
            meal["why"] = reasons[meal["slot"]]
    if reasoning.get("daily_tip"):
        plan["daily_total"]["tip"] = reasoning["daily_tip"]
    return plan


# =========================================================================
#  MAIN ENTRY POINT
# =========================================================================

def get_smart_meal_plan(profile: dict, model_dir: str = "models",
                        budget: str = "medium") -> dict:
    """
    Main entry point.
    1. XGBoost predicts macro targets
    2. Constraint optimizer selects foods for calories, protein, budget, slots
    3. Groq writes reasoning around the locked optimized plan
    4. Output is validated against targets
    """
    # Step 1: Predict targets via XGBoost
    models = load_models(model_dir)
    targets = predict_targets(profile, models)

    day_type = ("Match Day" if profile.get("is_match_day")
                else "Recovery Day" if profile.get("is_recovery_day")
                else "Training Day")

    budget_map = {"low": "Rs.50-80", "medium": "Rs.80-150", "high": "Rs.150-250"}
    budget_range = budget_map.get(budget, "Rs.80-150")

    # Step 2: Optimizer owns foods, macro totals, budget, slot structure, and variety.
    plan_data = _optimize_plan(profile, targets, budget)
    mode = "optimizer"

    # Step 3: Try Groq only for explanations. The optimized foods/numbers are locked.
    try:
        plan_data = _enhance_plan_with_groq(plan_data, profile, targets, budget_range)
        mode = "groq+optimizer"
        print("[SMART PLANNER] Groq reasoning received (optimizer-locked mode)")
    except Exception as exc:
        traceback.print_exc()
        print(f"[SMART PLANNER] Groq unavailable ({exc}), using optimizer reasoning")

    # Step 4: Validate against XGBoost targets
    plan_data = _validate_plan(plan_data, targets)
    plan_data["mode"] = mode

    return {
        "success": True,
        "profile": {
            "name": profile.get("name", "Athlete"),
            "sport": profile["sport"],
            "role": profile.get("role", ""),
            "gender": profile["gender"],
            "age": profile["age"],
            "weight_kg": profile["weight_kg"],
            "height_cm": profile["height_cm"],
            "goal": profile["goal"],
            "diet_type": profile["diet_type"],
            "is_match_day": profile.get("is_match_day", 0),
            "is_recovery_day": profile.get("is_recovery_day", 0),
            "day_type": day_type,
        },
        "targets": targets,
        "plan": plan_data,
    }


if __name__ == "__main__":
    test = {
        "name": "Rahul", "sport": "Cricket", "role": "Fast bowler",
        "gender": "male", "age": 22, "weight_kg": 72, "height_cm": 175,
        "training_intensity": 8, "session_duration_min": 120,
        "training_days_per_week": 6, "is_match_day": 1, "is_recovery_day": 0,
        "fatigue_score": 6, "goal": "maintain", "diet_type": "non-veg",
    }
    result = get_smart_meal_plan(test, budget="low")
    v = result["plan"]["validation"]
    print(f"Mode: {result['plan']['mode']}")
    print(f"Cal accuracy: {v['cal_accuracy']} ({v['actual_cal']}/{v['target_cal']})")
    print(f"Pro accuracy: {v['protein_accuracy']} ({v['actual_protein']}/{v['target_protein']})")
    print(f"Within range: {v['within_range']}")
    print()
    for m in result["plan"]["meals"]:
        cost = m.get('approx_cost', '?')
        print(f"{m['slot']} ({m['approx_cal']} kcal, {m['approx_protein']}g protein, ~Rs.{cost})")
        for item in m["items"]:
            print(f"  > {item}")
        print(f"  Why: {m['why']}")
        print()
