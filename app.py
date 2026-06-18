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

# Injury backend imports
from src.database.repository import DiagnosisRepository
from src.diagnosis_model.hybrid import DiagnosisEngine
from src.feedback.service import FeedbackService
from src.inference import InjuryRiskPredictor
from src.questionnaire.schemas import ConfirmationRequest, DiagnosisRequest, FeedbackRequest


load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Valid values (must match meal_planner.py encoders) ───
VALID_SPORTS  = ["Cricket", "Football", "Kabaddi", "Athletics", "Wrestling", "Badminton"]
VALID_GENDERS = ["male", "female"]
VALID_GOALS   = ["maintain", "bulk", "cut"]
VALID_DIETS   = ["veg", "non-veg"]

# ─── Lazy Loaded Injury Engines ───
_predictor = None
_diagnosis_engine = None
_diagnosis_repository = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = InjuryRiskPredictor()
    return _predictor

def get_diagnosis_engine():
    global _diagnosis_engine
    if _diagnosis_engine is None:
        _diagnosis_engine = DiagnosisEngine()
    return _diagnosis_engine

def get_diagnosis_repository():
    global _diagnosis_repository
    if _diagnosis_repository is None:
        _diagnosis_repository = DiagnosisRepository()
    return _diagnosis_repository


#  PAGE ROUTES

@app.route("/")
def home():
    """Landing page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight health check for Render."""
    return jsonify({"status": "ok"})


@app.route("/nutrition")
def nutrition():
    """Nutrition / Meal Planner page."""
    return render_template("nutrition.html")


@app.route("/injuries")
def injuries():
    """Injuries page (under maintenance)."""
    return render_template("injuries.html")


#  API ROUTES

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


# ─── INJURY API ROUTES ───

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        if "metrics" in data and isinstance(data["metrics"], dict):
            metrics = data.pop("metrics")
            data.update(metrics)
        return jsonify(get_predictor().predict_one(data))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400

@app.route("/api/batch_predict", methods=["POST"])
def api_batch_predict():
    data = request.get_json(force=True)
    try:
        return jsonify({"predictions": get_predictor().predict_batch(data)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    data = request.get_json(force=True)
    try:
        payload = DiagnosisRequest(**data)
        response = get_diagnosis_engine().diagnose(payload)
        try:
            get_diagnosis_repository().store_assessment(payload, response)
        except Exception:
            pass
        return jsonify(response.model_dump())
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400

@app.route("/api/confirm-diagnosis", methods=["POST"])
def api_confirm_diagnosis():
    data = request.get_json(force=True)
    try:
        payload = ConfirmationRequest(**data)
        result = FeedbackService(get_diagnosis_repository()).confirm_diagnosis(payload)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(force=True)
    try:
        payload = FeedbackRequest(**data)
        result = FeedbackService(get_diagnosis_repository()).store_feedback(payload)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/assessment/<assessment_id>", methods=["GET"])
def api_get_assessment(assessment_id):
    try:
        assessment = get_diagnosis_repository().get_assessment(assessment_id)
        if assessment is None:
            return jsonify({"error": "Assessment not found"}), 404
        return jsonify(assessment)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


#  RUN

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  AthleteEdge AI server running at http://localhost:{port}\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
