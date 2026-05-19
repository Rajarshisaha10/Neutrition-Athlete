# AthleteEdge AI — Nutrition Engine

> Free, AI-powered sport-specific meal planning for grassroots Indian athletes.

---

## What this is

AthleteEdge AI is a nutrition engine that takes an athlete's profile — sport, age, weight, training load, match/recovery day — and outputs a personalised full-day Indian meal plan built from affordable local foods. No Western supplements, no urban assumptions.

This repo covers the **Nutrition Engine** only. The Recovery Engine lives separately.

---

## Repo structure

```
Neutrition-Athlete/
│
├── data/
│   ├── food_db.json               ← canonical food database
│   └── nutrition_dataset.csv      ← model training data
│
├── models/                        ← auto-created by train_model.py
│   ├── xgb_calories.pkl
│   ├── xgb_protein.pkl
│   ├── xgb_carbs.pkl
│   ├── xgb_fat.pkl
│   ├── xgb_needs_iron.pkl
│   ├── xgb_needs_calcium.pkl
│   ├── xgb_needs_vitd.pkl
│   └── feature_list.pkl
│
├── fooddb.py                      ← food DB schema loader & validator
├── meal_planner.py                ← main planner: predict macros → build meal plan
├── train_model.py                 ← trains XGBoost models on nutrition_dataset.csv
├── requirements.txt
└── tests/
    └── test_meal_planner.py
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
python train_model.py
# Trains 4 regressors + 3 classifiers, saves to models/
```

### 3. Generate a meal plan

```python
from meal_planner import get_meal_plan

profile = {
    "name":                   "Rahul",
    "sport":                  "Cricket",
    "role":                   "Fast bowler",
    "gender":                 "male",
    "age":                    22,
    "weight_kg":              72.0,
    "height_cm":              175,
    "training_intensity":     8,        # 0–10
    "session_duration_min":   120,
    "training_days_per_week": 6,
    "is_match_day":           1,        # 1 or 0
    "is_recovery_day":        0,
    "fatigue_score":          6,        # 1–10
    "goal":                   "maintain",   # maintain / bulk / cut
    "diet_type":              "non-veg",    # veg / non-veg
}

plan = get_meal_plan(profile)
print(plan["meal_plan_text"])
```

### 4. Run the built-in demo

```bash
python meal_planner.py
# Prints plans for 3 demo athletes: Rahul (Cricket), Priya (Athletics), Arjun (Kabaddi)
```

---

## How it works

```
Athlete profile
      │
      ▼
XGBoost regressors  →  daily targets: calories, protein, carbs, fat
XGBoost classifiers →  micronutrient flags: iron / calcium / vit-D
      │
      ▼
Greedy knapsack planner
  - 5 meal slots: Breakfast · Lunch · Pre-workout · Dinner · Post-match
  - Scores foods by protein density + micro bonus + calorie fit
  - Filters by diet preference (veg / non-veg)
  - Avoids repeating foods across slots
      │
      ▼
Formatted meal plan (text + structured dict)
```

---

## ML models

| Model | Type | Target |
|---|---|---|
| `xgb_calories` | XGBRegressor | Daily calorie need (kcal) |
| `xgb_protein` | XGBRegressor | Protein target (g) |
| `xgb_carbs` | XGBRegressor | Carbohydrate target (g) |
| `xgb_fat` | XGBRegressor | Fat target (g) |
| `xgb_needs_iron` | XGBClassifier | Iron deficiency risk (0/1) |
| `xgb_needs_calcium` | XGBClassifier | Calcium deficiency risk (0/1) |
| `xgb_needs_vitd` | XGBClassifier | Vitamin-D deficiency risk (0/1) |

### Input features

| Feature | Description |
|---|---|
| `sport_enc` | Cricket=0, Football=1, Kabaddi=2, Athletics=3, Wrestling=4, Badminton=5 |
| `gender_enc` | male=0, female=1 |
| `age` | Years |
| `weight_kg` | Body weight |
| `height_cm` | Height |
| `training_intensity` | 0–10 scale |
| `session_duration_min` | Session length in minutes |
| `training_days_per_week` | 1–7 |
| `is_match_day` | 1 if competition day |
| `is_recovery_day` | 1 if active recovery day |
| `fatigue_score` | 1–10 self-reported fatigue |
| `goal_enc` | maintain=0, bulk=1, cut=2 |
| `diet_enc` | veg=0, non-veg=1 |

---

## Food database

The food database lives in `data/food_db.json` and is loaded through `fooddb.py`. It includes Indian foods covering:

- Grains & staples (roti, bajra, jowar, rice, poha, idli, dosa)
- Pulses & legumes (moong, toor, chana, masoor, rajma, chole)
- Dairy & eggs
- Meat, fish & poultry
- Vegetables, fruits, nuts & seeds
- Regional snacks (chikki, sattu, murmura, coconut water)

Each food has: serving size, calories, protein, carbs, fat, fiber, iron, calcium, vitamin-D, meal slots, veg/non-veg flag, and approximate cost.

---

## Supported sports

Cricket · Football · Kabaddi · Athletics · Wrestling · Badminton

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Roadmap

- [ ] 7-day meal rotation (no repeated plans)
- [ ] Budget tiers: ₹50 / ₹100 / ₹200 per day
- [ ] Hydration & electrolyte output for match days
- [ ] Hindi / Bengali / Tamil vernacular output
- [ ] FastAPI inference endpoint for mobile app
- [ ] Recovery Engine integration

---

## Disclaimer

AthleteEdge AI provides general nutritional guidance only. It is not a substitute for advice from a qualified sports nutritionist or medical professional. Always consult a doctor for clinical dietary needs.

---

## License

MIT
