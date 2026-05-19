"""
AthleteEdge AI — Flask Web Server
===================================
Serves the landing page, nutrition planner, and injuries page.
Exposes /api/meal-plan endpoint powered by Groq LLM.

Run:
    python app.py
"""

import os
import traceback
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from smart_planner import get_smart_meal_plan

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Valid values (must match meal_planner.py encoders) ───
VALID_SPORTS  = ["Cricket", "Football", "Kabaddi", "Athletics", "Wrestling", "Badminton"]
VALID_GENDERS = ["male", "female"]
VALID_GOALS   = ["maintain", "bulk", "cut"]
VALID_DIETS   = ["veg", "non-veg"]


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Landing page."""
    return render_template("index.html")


@app.route("/nutrition")
def nutrition():
    """Nutrition / Meal Planner page."""
    return render_template("nutrition.html")


@app.route("/injuries")
def injuries():
    """Injuries page (under maintenance)."""
    return render_template("injuries.html")


# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sports", methods=["GET"])
def api_sports():
    """Return available sports."""
    return jsonify({"sports": VALID_SPORTS})


@app.route("/api/meal-plan", methods=["POST"])
def api_meal_plan():
    """
    Accept athlete profile JSON, run XGBoost prediction + greedy planner,
    return structured meal plan.
    """
    data = request.get_json(force=True)

    # ── Validate required fields ──
    required = [
        "sport", "gender", "age", "weight_kg", "height_cm",
        "training_intensity", "session_duration_min",
        "training_days_per_week", "is_match_day", "is_recovery_day",
        "fatigue_score", "goal", "diet_type",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # ── Validate enum values ──
    if data["sport"] not in VALID_SPORTS:
        return jsonify({"error": f"Invalid sport. Choose from: {VALID_SPORTS}"}), 400
    if data["gender"] not in VALID_GENDERS:
        return jsonify({"error": f"Invalid gender. Choose from: {VALID_GENDERS}"}), 400
    if data["goal"] not in VALID_GOALS:
        return jsonify({"error": f"Invalid goal. Choose from: {VALID_GOALS}"}), 400
    if data["diet_type"] not in VALID_DIETS:
        return jsonify({"error": f"Invalid diet_type. Choose from: {VALID_DIETS}"}), 400

    # ── Cast numeric fields ──
    try:
        profile = {
            "name":                   data.get("name", "Athlete"),
            "sport":                  data["sport"],
            "role":                   data.get("role", ""),
            "gender":                 data["gender"],
            "age":                    int(data["age"]),
            "weight_kg":              float(data["weight_kg"]),
            "height_cm":              float(data["height_cm"]),
            "training_intensity":     int(data["training_intensity"]),
            "session_duration_min":   int(data["session_duration_min"]),
            "training_days_per_week": int(data["training_days_per_week"]),
            "is_match_day":           int(data["is_match_day"]),
            "is_recovery_day":        int(data["is_recovery_day"]),
            "fatigue_score":          int(data["fatigue_score"]),
            "goal":                   data["goal"],
            "diet_type":              data["diet_type"],
        }
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid data types: {exc}"}), 400

    # ── Run the smart planner (XGBoost + Groq LLM / offline fallback) ──
    budget = data.get("budget", "medium")  # low / medium / high
    try:
        result = get_smart_meal_plan(profile, budget=budget)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  AthleteEdge AI server running at http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
