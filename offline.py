"""
AthleteEdge AI — Greedy Meal Planner
=====================================
Given an athlete profile, predicts daily macro targets via the trained
XGBoost models, then uses a greedy knapsack algorithm to fill 5 meals
(Breakfast, Lunch, Pre-workout snack, Dinner, Post-match/recovery)
from an Indian food database — outputting exact food items with grams.

Run:
    python meal_planner.py

Requires:
    models/  folder with trained .pkl files  ← run train_model.py first

Usage as a module:
    from meal_planner import get_meal_plan
    plan = get_meal_plan(profile_dict)
    print(plan["meal_plan_text"])
"""

import os, joblib
import pandas as pd
import numpy as np
from copy import deepcopy

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  INDIAN FOOD DATABASE
#     Each entry: name, calories/100g, protein/100g, carbs/100g, fat/100g,
#                 tags (meal slots it suits), diet (veg/non-veg/both),
#                 iron_rich, calcium_rich, vitd_rich, unit_g (natural serving g)
# ═══════════════════════════════════════════════════════════════════════════════

FOOD_DB = [
    # ── Grains & Staples ──────────────────────────────────────────────────────
    {"name": "Whole wheat roti",        "cal": 264, "pro": 8.0,  "carb": 52.0, "fat": 3.7,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 40},
    {"name": "Bajra roti",              "cal": 360, "pro": 11.6, "carb": 67.5, "fat": 5.0,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 40},
    {"name": "Jowar roti",              "cal": 349, "pro": 10.4, "carb": 72.6, "fat": 1.9,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 40},
    {"name": "Brown rice (cooked)",     "cal": 111, "pro": 2.6,  "carb": 23.0, "fat": 0.9,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "White rice (cooked)",     "cal": 130, "pro": 2.7,  "carb": 28.2, "fat": 0.3,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Poha (cooked)",           "cal": 180, "pro": 3.5,  "carb": 36.5, "fat": 2.7,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Upma (rava)",             "cal": 172, "pro": 4.8,  "carb": 31.2, "fat": 3.9,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Idli (2 pieces)",         "cal": 138, "pro": 4.4,  "carb": 28.0, "fat": 0.4,  "tags": ["breakfast"],                            "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 100},
    {"name": "Dosa (plain)",            "cal": 133, "pro": 3.0,  "carb": 24.0, "fat": 2.9,  "tags": ["breakfast"],                            "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 100},
    {"name": "Paratha (plain)",         "cal": 297, "pro": 6.9,  "carb": 44.5, "fat": 9.8,  "tags": ["breakfast","lunch"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 60},
    {"name": "Chapati (multigrain)",    "cal": 240, "pro": 7.5,  "carb": 47.0, "fat": 2.5,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 40},
    {"name": "Oats (cooked)",           "cal": 71,  "pro": 2.5,  "carb": 12.0, "fat": 1.5,  "tags": ["breakfast"],                            "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 200},

    # ── Pulses & Legumes ──────────────────────────────────────────────────────
    {"name": "Moong dal (cooked)",      "cal": 104, "pro": 7.0,  "carb": 16.3, "fat": 0.4,  "tags": ["lunch","dinner","snack"],               "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Toor dal (cooked)",       "cal": 116, "pro": 6.8,  "carb": 18.3, "fat": 0.7,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Chana dal (cooked)",      "cal": 164, "pro": 8.7,  "carb": 26.2, "fat": 2.6,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Masoor dal (cooked)",     "cal": 116, "pro": 9.0,  "carb": 20.1, "fat": 0.4,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Rajma (cooked)",          "cal": 127, "pro": 8.7,  "carb": 22.8, "fat": 0.5,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 150},
    {"name": "Chole / chickpeas",       "cal": 164, "pro": 8.9,  "carb": 27.4, "fat": 2.6,  "tags": ["lunch","dinner","snack"],               "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 150},
    {"name": "Soybean curry",           "cal": 173, "pro": 16.6, "carb": 9.9,  "fat": 9.0,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 150},
    {"name": "Sprouted moong",          "cal": 30,  "pro": 3.0,  "carb": 4.1,  "fat": 0.2,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 100},

    # ── Dairy & Eggs ──────────────────────────────────────────────────────────
    {"name": "Whole milk (glass)",      "cal": 61,  "pro": 3.2,  "carb": 4.8,  "fat": 3.3,  "tags": ["breakfast","snack","post"],             "diet": "veg",     "iron": False, "ca": True,  "vitd": True,  "unit_g": 250},
    {"name": "Low-fat milk",            "cal": 42,  "pro": 3.4,  "carb": 5.0,  "fat": 1.0,  "tags": ["breakfast","snack","post"],             "diet": "veg",     "iron": False, "ca": True,  "vitd": True,  "unit_g": 250},
    {"name": "Paneer (cottage cheese)", "cal": 265, "pro": 18.3, "carb": 1.2,  "fat": 20.8, "tags": ["breakfast","lunch","dinner","snack"],   "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 100},
    {"name": "Curd / dahi",             "cal": 98,  "pro": 11.0, "carb": 3.4,  "fat": 4.3,  "tags": ["breakfast","lunch","dinner","snack"],   "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 150},
    {"name": "Buttermilk (chaas)",      "cal": 40,  "pro": 3.3,  "carb": 4.9,  "fat": 0.9,  "tags": ["snack","post"],                         "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 200},
    {"name": "Boiled egg",              "cal": 155, "pro": 13.0, "carb": 1.1,  "fat": 11.0, "tags": ["breakfast","snack","post"],             "diet": "non-veg", "iron": True,  "ca": False, "vitd": True,  "unit_g": 50},
    {"name": "Egg white (2 nos)",       "cal": 34,  "pro": 7.2,  "carb": 0.5,  "fat": 0.1,  "tags": ["breakfast","snack","post"],             "diet": "non-veg", "iron": False, "ca": False, "vitd": False, "unit_g": 66},
    {"name": "Whey protein (scoop)",    "cal": 120, "pro": 25.0, "carb": 3.0,  "fat": 1.5,  "tags": ["snack","post"],                         "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 30},

    # ── Meat, Fish & Poultry ──────────────────────────────────────────────────
    {"name": "Chicken breast (grilled)","cal": 165, "pro": 31.0, "carb": 0.0,  "fat": 3.6,  "tags": ["lunch","dinner","post"],                "diet": "non-veg", "iron": False, "ca": False, "vitd": False, "unit_g": 120},
    {"name": "Chicken curry (desi)",    "cal": 243, "pro": 23.5, "carb": 4.0,  "fat": 14.8, "tags": ["lunch","dinner"],                       "diet": "non-veg", "iron": True,  "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Egg bhurji (2 eggs)",     "cal": 189, "pro": 14.5, "carb": 3.2,  "fat": 13.2, "tags": ["breakfast","snack"],                    "diet": "non-veg", "iron": True,  "ca": False, "vitd": True,  "unit_g": 120},
    {"name": "Fish curry (rohu)",       "cal": 194, "pro": 20.5, "carb": 5.1,  "fat": 9.5,  "tags": ["lunch","dinner"],                       "diet": "non-veg", "iron": True,  "ca": True,  "vitd": True,  "unit_g": 150},
    {"name": "Tuna (canned in water)",  "cal": 116, "pro": 25.5, "carb": 0.0,  "fat": 1.0,  "tags": ["lunch","snack","post"],                 "diet": "non-veg", "iron": False, "ca": False, "vitd": True,  "unit_g": 100},
    {"name": "Mutton curry",            "cal": 310, "pro": 25.5, "carb": 3.0,  "fat": 21.5, "tags": ["lunch","dinner"],                       "diet": "non-veg", "iron": True,  "ca": False, "vitd": False, "unit_g": 150},

    # ── Vegetables ────────────────────────────────────────────────────────────
    {"name": "Palak (spinach) sabzi",   "cal": 49,  "pro": 3.5,  "carb": 5.6,  "fat": 1.5,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 100},
    {"name": "Methi (fenugreek) sabzi", "cal": 49,  "pro": 4.4,  "carb": 6.0,  "fat": 1.0,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 100},
    {"name": "Mixed vegetable sabzi",   "cal": 60,  "pro": 2.5,  "carb": 9.5,  "fat": 1.8,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 120},
    {"name": "Aloo sabzi",              "cal": 93,  "pro": 2.0,  "carb": 18.8, "fat": 1.5,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 120},
    {"name": "Bhindi (okra) sabzi",     "cal": 38,  "pro": 1.9,  "carb": 6.4,  "fat": 0.7,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 100},
    {"name": "Broccoli (steamed)",      "cal": 34,  "pro": 2.8,  "carb": 6.6,  "fat": 0.4,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 100},

    # ── Fruits ────────────────────────────────────────────────────────────────
    {"name": "Banana",                  "cal": 89,  "pro": 1.1,  "carb": 23.0, "fat": 0.3,  "tags": ["breakfast","snack","post"],             "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 120},
    {"name": "Apple",                   "cal": 52,  "pro": 0.3,  "carb": 14.0, "fat": 0.2,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Mango (seasonal)",        "cal": 60,  "pro": 0.8,  "carb": 15.0, "fat": 0.4,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Papaya",                  "cal": 43,  "pro": 0.5,  "carb": 11.0, "fat": 0.3,  "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 150},
    {"name": "Guava",                   "cal": 68,  "pro": 2.6,  "carb": 14.3, "fat": 1.0,  "tags": ["snack"],                                "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 100},

    # ── Nuts & Seeds ──────────────────────────────────────────────────────────
    {"name": "Peanuts (roasted)",       "cal": 567, "pro": 25.8, "carb": 16.1, "fat": 49.2, "tags": ["snack","post"],                         "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 30},
    {"name": "Almonds",                 "cal": 579, "pro": 21.2, "carb": 21.7, "fat": 49.9, "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 20},
    {"name": "Walnuts",                 "cal": 654, "pro": 15.2, "carb": 13.7, "fat": 65.2, "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 20},
    {"name": "Flaxseeds",               "cal": 534, "pro": 18.3, "carb": 28.9, "fat": 42.2, "tags": ["breakfast","snack"],                    "diet": "veg",     "iron": True,  "ca": True,  "vitd": False, "unit_g": 10},
    {"name": "Pumpkin seeds",           "cal": 559, "pro": 30.2, "carb": 10.7, "fat": 49.1, "tags": ["snack"],                                "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 20},

    # ── Snacks & Fast-energy Foods ────────────────────────────────────────────
    {"name": "Chikki (peanut)",         "cal": 452, "pro": 12.0, "carb": 62.0, "fat": 18.0, "tags": ["snack"],                                "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 40},
    {"name": "Sattu drink (shakti)",    "cal": 108, "pro": 5.6,  "carb": 18.0, "fat": 1.2,  "tags": ["snack","post"],                         "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 30},
    {"name": "Murmura chivda",          "cal": 402, "pro": 7.7,  "carb": 85.8, "fat": 2.3,  "tags": ["snack"],                                "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 50},
    {"name": "Coconut water",           "cal": 19,  "pro": 0.7,  "carb": 3.7,  "fat": 0.2,  "tags": ["snack","post"],                         "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 250},
    {"name": "Lassi (salted)",          "cal": 75,  "pro": 3.5,  "carb": 6.0,  "fat": 3.8,  "tags": ["snack","post"],                         "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 200},
    {"name": "Banana shake (milk)",     "cal": 130, "pro": 4.5,  "carb": 25.0, "fat": 2.5,  "tags": ["breakfast","post"],                     "diet": "veg",     "iron": False, "ca": True,  "vitd": False, "unit_g": 300},

    # ── Fats & Condiments ─────────────────────────────────────────────────────
    {"name": "Ghee",                    "cal": 900, "pro": 0.0,  "carb": 0.0,  "fat": 99.5, "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": False, "ca": False, "vitd": True,  "unit_g": 5},
    {"name": "Groundnut oil",           "cal": 884, "pro": 0.0,  "carb": 0.0,  "fat": 100,  "tags": ["lunch","dinner"],                       "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 5},
    {"name": "Coconut chutney",         "cal": 250, "pro": 2.7,  "carb": 10.4, "fat": 22.4, "tags": ["breakfast"],                            "diet": "veg",     "iron": False, "ca": False, "vitd": False, "unit_g": 30},
    {"name": "Sambar (200ml)",          "cal": 60,  "pro": 3.2,  "carb": 9.8,  "fat": 1.3,  "tags": ["breakfast","lunch","dinner"],            "diet": "veg",     "iron": True,  "ca": False, "vitd": False, "unit_g": 200},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MEAL SLOT DEFINITIONS
#     name, slot_tag, calorie_fraction, protein_fraction
# ═══════════════════════════════════════════════════════════════════════════════

MEAL_SLOTS = [
    {"name": "Breakfast",             "tag": "breakfast", "cal_frac": 0.25, "max_items": 4},
    {"name": "Lunch",                 "tag": "lunch",     "cal_frac": 0.35, "max_items": 5},
    {"name": "Pre-workout Snack",     "tag": "snack",     "cal_frac": 0.10, "max_items": 2},
    {"name": "Dinner",                "tag": "dinner",    "cal_frac": 0.25, "max_items": 4},
    {"name": "Post-match / Recovery", "tag": "post",      "cal_frac": 0.05, "max_items": 2},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ENCODER MAPS  (must match train_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

SPORT_ENC  = {"Cricket": 0, "Football": 1, "Kabaddi": 2,
               "Athletics": 3, "Wrestling": 4, "Badminton": 5}
GENDER_ENC = {"male": 0, "female": 1}
GOAL_ENC   = {"maintain": 0, "bulk": 1, "cut": 2}
DIET_ENC   = {"veg": 0, "non-veg": 1}


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    models = {}
    for key in ["calories", "protein", "carbs", "fat",
                "needs_iron", "needs_calcium", "needs_vitd"]:
        path = os.path.join(model_dir, f"xgb_{key}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Please run train_model.py first."
            )
        models[key] = joblib.load(path)
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MACRO PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def predict_targets(profile: dict, models: dict) -> dict:
    """
    profile keys: sport, gender, age, weight_kg, height_cm,
                  training_intensity, session_duration_min,
                  training_days_per_week, is_match_day, is_recovery_day,
                  fatigue_score, goal, diet_type
    """
    row = pd.DataFrame([{
        "sport_enc":              SPORT_ENC[profile["sport"]],
        "gender_enc":             GENDER_ENC[profile["gender"]],
        "age":                    profile["age"],
        "weight_kg":              profile["weight_kg"],
        "height_cm":              profile["height_cm"],
        "training_intensity":     profile["training_intensity"],
        "session_duration_min":   profile["session_duration_min"],
        "training_days_per_week": profile["training_days_per_week"],
        "is_match_day":           profile["is_match_day"],
        "is_recovery_day":        profile["is_recovery_day"],
        "fatigue_score":          profile["fatigue_score"],
        "goal_enc":               GOAL_ENC[profile["goal"]],
        "diet_enc":               DIET_ENC[profile["diet_type"]],
    }])

    return {
        "calories":      max(1200, round(float(models["calories"].predict(row)[0]))),
        "protein_g":     max(40,   round(float(models["protein"].predict(row)[0]),  1)),
        "carbs_g":       max(100,  round(float(models["carbs"].predict(row)[0]),    1)),
        "fat_g":         max(20,   round(float(models["fat"].predict(row)[0]),      1)),
        "needs_iron":    int(models["needs_iron"].predict(row)[0]),
        "needs_calcium": int(models["needs_calcium"].predict(row)[0]),
        "needs_vitd":    int(models["needs_vitd"].predict(row)[0]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  GREEDY MEAL PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def _score_food(food: dict, remaining_cal: float, remaining_pro: float,
                needs_iron: bool, needs_calcium: bool, needs_vitd: bool) -> float:
    """
    Composite score balancing protein density, calorie fit, and micro flags.
    Higher is better.
    """
    if food["cal"] == 0:
        return 0.0

    # Protein-per-calorie efficiency
    pro_density = food["pro"] / food["cal"]

    # Penalise if food would blow calorie budget by > 20%
    serving_cal = food["cal"] * food["unit_g"] / 100
    overshoot   = max(0, serving_cal - remaining_cal * 1.20)
    overshoot_penalty = overshoot / max(remaining_cal, 1)

    # Micro bonus
    micro_bonus = (
        (0.15 if needs_iron    and food["iron"] else 0.0) +
        (0.15 if needs_calcium and food["ca"]   else 0.0) +
        (0.15 if needs_vitd    and food["vitd"] else 0.0)
    )

    return pro_density * 10 + micro_bonus - overshoot_penalty


def _fill_slot(slot: dict, slot_cal_budget: float, slot_pro_budget: float,
               diet_type: str, needs_iron: bool, needs_calcium: bool,
               needs_vitd: bool, used_today: set) -> list:
    """
    Greedy fill of one meal slot.
    Returns list of {"name", "grams", "cal", "pro", "carb", "fat"}.
    """
    tag       = slot["tag"]
    max_items = slot["max_items"]

    # Filter eligible foods
    eligible = [
        f for f in FOOD_DB
        if tag in f["tags"]
        and (f["diet"] == "veg" or diet_type == "non-veg")
        and f["name"] not in used_today
    ]

    selected    = []
    rem_cal     = slot_cal_budget
    rem_pro     = slot_pro_budget

    for _ in range(max_items):
        if rem_cal <= 30 or not eligible:
            break

        # Score each eligible food
        scored = sorted(
            eligible,
            key=lambda f: _score_food(f, rem_cal, rem_pro,
                                      needs_iron, needs_calcium, needs_vitd),
            reverse=True
        )

        for best in scored:
            # Decide serving size (try natural unit; scale down if needed)
            base_g    = best["unit_g"]
            base_cal  = best["cal"] * base_g / 100

            if base_cal > rem_cal * 1.35 and base_g > 30:
                # Scale down to fit budget
                scale_g = int(rem_cal / best["cal"] * 100 / 10) * 10
                if scale_g < 20:
                    continue
                serve_g = scale_g
            else:
                serve_g = base_g

            serve_cal  = round(best["cal"]  * serve_g / 100, 1)
            serve_pro  = round(best["pro"]  * serve_g / 100, 1)
            serve_carb = round(best["carb"] * serve_g / 100, 1)
            serve_fat  = round(best["fat"]  * serve_g / 100, 1)

            selected.append({
                "name": best["name"],
                "grams": serve_g,
                "cal": serve_cal,
                "pro": serve_pro,
                "carb": serve_carb,
                "fat": serve_fat,
                "iron": best["iron"],
                "ca":   best["ca"],
                "vitd": best["vitd"],
            })

            rem_cal -= serve_cal
            rem_pro -= serve_pro
            eligible.remove(best)
            used_today.add(best["name"])
            break

    return selected


def build_meal_plan(targets: dict, diet_type: str) -> dict:
    """
    Returns structured meal plan dict.
    """
    total_cal = targets["calories"]
    total_pro = targets["protein_g"]

    needs_iron    = bool(targets["needs_iron"])
    needs_calcium = bool(targets["needs_calcium"])
    needs_vitd    = bool(targets["needs_vitd"])

    used_today = set()
    meals      = []

    for slot in MEAL_SLOTS:
        slot_cal = total_cal * slot["cal_frac"]
        slot_pro = total_pro * slot["cal_frac"]

        items = _fill_slot(
            slot, slot_cal, slot_pro,
            diet_type, needs_iron, needs_calcium, needs_vitd,
            used_today
        )
        meals.append({"slot": slot["name"], "items": items})

    return {
        "targets":  targets,
        "meals":    meals,
        "diet_type": diet_type,
        "flags": {
            "needs_iron":    needs_iron,
            "needs_calcium": needs_calcium,
            "needs_vitd":    needs_vitd,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TEXT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_plan(plan: dict, profile: dict) -> str:
    t  = plan["targets"]
    fl = plan["flags"]

    lines = []
    lines.append("=" * 62)
    lines.append("  AthleteEdge AI — Personalised Meal Plan")
    lines.append("=" * 62)
    lines.append(f"  Athlete  : {profile.get('name', 'Athlete')}")
    lines.append(f"  Sport    : {profile['sport']}  |  Role : {profile.get('role','—')}")
    lines.append(f"  Age/Wt   : {profile['age']} y  |  {profile['weight_kg']} kg  |  {profile['height_cm']} cm")
    lines.append(f"  Day Type : {'Match Day' if profile['is_match_day'] else ('Recovery Day' if profile['is_recovery_day'] else 'Training Day')}")
    lines.append(f"  Goal     : {profile['goal'].upper()}  |  Diet: {profile['diet_type'].upper()}")
    lines.append("-" * 62)
    lines.append("  DAILY TARGETS")
    lines.append(f"  Calories : {t['calories']} kcal")
    lines.append(f"  Protein  : {t['protein_g']} g  |  Carbs : {t['carbs_g']} g  |  Fat : {t['fat_g']} g")

    micro_alerts = []
    if fl["needs_iron"]:    micro_alerts.append("[Fe] Iron")
    if fl["needs_calcium"]: micro_alerts.append("[Ca] Calcium")
    if fl["needs_vitd"]:    micro_alerts.append("[D] Vit-D")
    if micro_alerts:
        lines.append(f"  Micro flags : {' | '.join(micro_alerts)}  (focus foods marked below)")
    lines.append("=" * 62)

    total_cal = total_pro = total_carb = total_fat = 0.0

    for meal in plan["meals"]:
        lines.append(f"\n{meal['slot']}")
        lines.append("  " + "-" * 55)

        if not meal["items"]:
            lines.append("  (no items — budget fully used in earlier meals)")
            continue

        meal_cal = meal_pro = meal_carb = meal_fat = 0.0
        for item in meal["items"]:
            micro_tag = ""
            if item.get("iron")  and fl["needs_iron"]:    micro_tag += "[Fe]"
            if item.get("ca")    and fl["needs_calcium"]: micro_tag += "[Ca]"
            if item.get("vitd")  and fl["needs_vitd"]:    micro_tag += "[D]"
            grams_str = f"{item['grams']}g"
            lines.append(
                f"  {micro_tag:<3} {item['name']:<28} {grams_str:>5}   "
                f"{item['cal']:.0f} kcal  |  P {item['pro']:.1f}g  C {item['carb']:.1f}g  F {item['fat']:.1f}g"
            )
            meal_cal  += item["cal"]
            meal_pro  += item["pro"]
            meal_carb += item["carb"]
            meal_fat  += item["fat"]

        lines.append("  " + "─" * 55)
        lines.append(
            f"  Meal total → {meal_cal:.0f} kcal  |  P {meal_pro:.1f}g  C {meal_carb:.1f}g  F {meal_fat:.1f}g"
        )
        total_cal  += meal_cal
        total_pro  += meal_pro
        total_carb += meal_carb
        total_fat  += meal_fat

    lines.append("\n" + "=" * 62)
    lines.append("  PLAN TOTALS")
    lines.append(f"  Calories : {total_cal:.0f}  (target {t['calories']})")
    lines.append(f"  Protein  : {total_pro:.1f}g  (target {t['protein_g']}g)")
    lines.append(f"  Carbs    : {total_carb:.1f}g  (target {t['carbs_g']}g)")
    lines.append(f"  Fat      : {total_fat:.1f}g  (target {t['fat_g']}g)")
    lines.append("=" * 62)

    lines.append("\nNote: This plan is AI-generated guidance only.")
    lines.append("Consult a qualified nutritionist for clinical advice.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_meal_plan(profile: dict, model_dir: str = "models") -> dict:
    """
    Main entry point.

    profile = {
        "name":                   "Rahul",          # optional
        "sport":                  "Cricket",
        "role":                   "Fast bowler",     # optional, for display
        "gender":                 "male",
        "age":                    22,
        "weight_kg":              72.0,
        "height_cm":              175,
        "training_intensity":     8,                 # 0-10
        "session_duration_min":   120,
        "training_days_per_week": 6,
        "is_match_day":           1,                 # 1 / 0
        "is_recovery_day":        0,
        "fatigue_score":          6,                 # 1-10
        "goal":                   "maintain",        # maintain / bulk / cut
        "diet_type":              "non-veg",         # veg / non-veg
    }

    Returns dict with keys:
        targets, meals, flags, diet_type, meal_plan_text
    """
    models  = load_models(model_dir)
    targets = predict_targets(profile, models)
    plan    = build_meal_plan(targets, profile["diet_type"])
    plan["meal_plan_text"] = format_plan(plan, profile)
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DEMO — run two athlete profiles
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_PROFILES = [
    {
        "name": "Rahul (Cricket, Match Day)",
        "sport": "Cricket", "role": "Fast bowler",
        "gender": "male", "age": 22, "weight_kg": 72, "height_cm": 175,
        "training_intensity": 8, "session_duration_min": 120,
        "training_days_per_week": 6, "is_match_day": 1, "is_recovery_day": 0,
        "fatigue_score": 6, "goal": "maintain", "diet_type": "non-veg",
    },
    {
        "name": "Priya (Athletics, Recovery Day)",
        "sport": "Athletics", "role": "Sprinter",
        "gender": "female", "age": 19, "weight_kg": 55, "height_cm": 163,
        "training_intensity": 4, "session_duration_min": 60,
        "training_days_per_week": 6, "is_match_day": 0, "is_recovery_day": 1,
        "fatigue_score": 8, "goal": "maintain", "diet_type": "veg",
    },
    {
        "name": "Arjun (Kabaddi, Bulk Training)",
        "sport": "Kabaddi", "role": "Raider",
        "gender": "male", "age": 25, "weight_kg": 80, "height_cm": 178,
        "training_intensity": 7, "session_duration_min": 100,
        "training_days_per_week": 5, "is_match_day": 0, "is_recovery_day": 0,
        "fatigue_score": 5, "goal": "bulk", "diet_type": "veg",
    },
]

if __name__ == "__main__":
    print("\nAthleteEdge AI — Meal Planner Demo\n")
    for profile in DEMO_PROFILES:
        plan = get_meal_plan(profile)
        print(plan["meal_plan_text"])
        print("\n")