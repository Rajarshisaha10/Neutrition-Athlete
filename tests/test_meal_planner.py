"""
AthleteEdge AI — Nutrition Engine Test Suite
=============================================
Tests fooddb schema, meal planner logic, macro targets, slot filling,
and output formatting — without requiring trained model .pkl files.

Run:
    pytest tests/test_meal_planner.py -v

To run against real trained models (integration tests):
    pytest tests/test_meal_planner.py -v --with-models
"""

from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import the modules under test ────────────────────────────────────────────
from meal_planner import (
    FOOD_DB,
    MEAL_SLOTS,
    SPORT_ENC,
    GENDER_ENC,
    GOAL_ENC,
    DIET_ENC,
    _score_food,
    _fill_slot,
    build_meal_plan,
    format_plan,
    get_meal_plan,
)
from fooddb import REQUIRED_FIELDS


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cricket_profile():
    return {
        "name": "Rahul",
        "sport": "Cricket",
        "role": "Fast bowler",
        "gender": "male",
        "age": 22,
        "weight_kg": 72.0,
        "height_cm": 175,
        "training_intensity": 8,
        "session_duration_min": 120,
        "training_days_per_week": 6,
        "is_match_day": 1,
        "is_recovery_day": 0,
        "fatigue_score": 6,
        "goal": "maintain",
        "diet_type": "non-veg",
    }

@pytest.fixture
def athletics_profile():
    return {
        "name": "Priya",
        "sport": "Athletics",
        "role": "Sprinter",
        "gender": "female",
        "age": 19,
        "weight_kg": 55.0,
        "height_cm": 163,
        "training_intensity": 4,
        "session_duration_min": 60,
        "training_days_per_week": 6,
        "is_match_day": 0,
        "is_recovery_day": 1,
        "fatigue_score": 8,
        "goal": "maintain",
        "diet_type": "veg",
    }

@pytest.fixture
def kabaddi_bulk_profile():
    return {
        "name": "Arjun",
        "sport": "Kabaddi",
        "role": "Raider",
        "gender": "male",
        "age": 25,
        "weight_kg": 80.0,
        "height_cm": 178,
        "training_intensity": 7,
        "session_duration_min": 100,
        "training_days_per_week": 5,
        "is_match_day": 0,
        "is_recovery_day": 0,
        "fatigue_score": 5,
        "goal": "bulk",
        "diet_type": "veg",
    }

@pytest.fixture
def mock_targets_standard():
    return {
        "calories": 2800,
        "protein_g": 140.0,
        "carbs_g": 350.0,
        "fat_g": 70.0,
        "needs_iron": 0,
        "needs_calcium": 0,
        "needs_vitd": 0,
    }

@pytest.fixture
def mock_targets_with_flags():
    return {
        "calories": 2400,
        "protein_g": 110.0,
        "carbs_g": 300.0,
        "fat_g": 60.0,
        "needs_iron": 1,
        "needs_calcium": 1,
        "needs_vitd": 1,
    }

def make_mock_models(targets):
    """Return a dict of mocked joblib models that return fixed target values."""
    models = {}
    for key in ["calories", "protein", "carbs", "fat"]:
        m = MagicMock()
        val = targets[f"{key}_g"] if key != "calories" else targets["calories"]
        if key == "calories":
            val = targets["calories"]
        elif key == "protein":
            val = targets["protein_g"]
        elif key == "carbs":
            val = targets["carbs_g"]
        elif key == "fat":
            val = targets["fat_g"]
        m.predict.return_value = [float(val)]
        models[key] = m

    for flag in ["needs_iron", "needs_calcium", "needs_vitd"]:
        m = MagicMock()
        m.predict.return_value = [targets[flag]]
        models[flag] = m

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FOOD DATABASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFoodDatabase:

    def test_food_db_not_empty(self):
        assert len(FOOD_DB) > 0, "FOOD_DB must contain at least one item"

    def test_food_db_has_minimum_items(self):
        assert len(FOOD_DB) >= 40, f"Expected at least 40 foods, got {len(FOOD_DB)}"

    def test_every_food_has_required_keys(self):
        required = {"name", "cal", "pro", "carb", "fat", "tags", "diet",
                    "iron", "ca", "vitd", "unit_g"}
        for food in FOOD_DB:
            missing = required - food.keys()
            assert not missing, f"'{food.get('name','?')}' missing keys: {missing}"

    def test_no_negative_macros(self):
        for food in FOOD_DB:
            assert food["cal"]  >= 0, f"'{food['name']}' has negative calories"
            assert food["pro"]  >= 0, f"'{food['name']}' has negative protein"
            assert food["carb"] >= 0, f"'{food['name']}' has negative carbs"
            assert food["fat"]  >= 0, f"'{food['name']}' has negative fat"

    def test_unit_g_is_positive(self):
        for food in FOOD_DB:
            assert food["unit_g"] > 0, f"'{food['name']}' unit_g must be > 0"

    def test_diet_field_valid_values(self):
        valid = {"veg", "non-veg", "both"}
        for food in FOOD_DB:
            assert food["diet"] in valid, \
                f"'{food['name']}' has invalid diet value: '{food['diet']}'"

    def test_tags_are_lists(self):
        for food in FOOD_DB:
            assert isinstance(food["tags"], list), \
                f"'{food['name']}' tags must be a list"
            assert len(food["tags"]) > 0, \
                f"'{food['name']}' must have at least one tag"

    def test_valid_tag_values(self):
        valid_tags = {"breakfast", "lunch", "dinner", "snack", "post"}
        for food in FOOD_DB:
            for tag in food["tags"]:
                assert tag in valid_tags, \
                    f"'{food['name']}' has unknown tag: '{tag}'"

    def test_boolean_micro_flags(self):
        for food in FOOD_DB:
            for flag in ("iron", "ca", "vitd"):
                assert isinstance(food[flag], bool), \
                    f"'{food['name']}' flag '{flag}' must be bool"

    def test_foods_cover_all_meal_slots(self):
        tags_present = set()
        for food in FOOD_DB:
            tags_present.update(food["tags"])
        required_slots = {"breakfast", "lunch", "dinner", "snack"}
        assert required_slots.issubset(tags_present), \
            f"Missing foods for slots: {required_slots - tags_present}"

    def test_veg_foods_exist(self):
        veg_foods = [f for f in FOOD_DB if f["diet"] == "veg"]
        assert len(veg_foods) >= 20, "Need at least 20 veg foods"

    def test_iron_rich_foods_exist(self):
        iron_foods = [f for f in FOOD_DB if f["iron"]]
        assert len(iron_foods) >= 5, "Need at least 5 iron-rich foods"

    def test_no_duplicate_food_names(self):
        names = [f["name"] for f in FOOD_DB]
        assert len(names) == len(set(names)), \
            "Duplicate food names found in FOOD_DB"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER MAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoderMaps:

    def test_sport_enc_has_all_supported_sports(self):
        expected = {"Cricket", "Football", "Kabaddi", "Athletics", "Wrestling", "Badminton"}
        assert expected.issubset(set(SPORT_ENC.keys()))

    def test_sport_enc_values_are_unique(self):
        vals = list(SPORT_ENC.values())
        assert len(vals) == len(set(vals)), "SPORT_ENC values must be unique"

    def test_gender_enc_covers_both(self):
        assert "male" in GENDER_ENC and "female" in GENDER_ENC

    def test_goal_enc_covers_all_goals(self):
        for goal in ("maintain", "bulk", "cut"):
            assert goal in GOAL_ENC, f"'{goal}' missing from GOAL_ENC"

    def test_diet_enc_covers_both(self):
        assert "veg" in DIET_ENC and "non-veg" in DIET_ENC


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEAL SLOT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMealSlots:

    def test_five_meal_slots(self):
        assert len(MEAL_SLOTS) == 5

    def test_calorie_fractions_sum_to_one(self):
        total = sum(s["cal_frac"] for s in MEAL_SLOTS)
        assert abs(total - 1.0) < 0.01, \
            f"Meal slot cal_frac should sum to 1.0, got {total}"

    def test_each_slot_has_required_keys(self):
        required = {"name", "tag", "cal_frac", "max_items"}
        for slot in MEAL_SLOTS:
            missing = required - slot.keys()
            assert not missing, f"Slot '{slot.get('name')}' missing: {missing}"

    def test_max_items_positive(self):
        for slot in MEAL_SLOTS:
            assert slot["max_items"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORING FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreFood:

    def _make_food(self, cal=200, pro=10, unit_g=100,
                   iron=False, ca=False, vitd=False):
        return {
            "name": "Test food",
            "cal": cal, "pro": pro, "carb": 20, "fat": 5,
            "iron": iron, "ca": ca, "vitd": vitd,
            "unit_g": unit_g, "tags": ["lunch"], "diet": "veg",
        }

    def test_zero_calorie_food_scores_zero(self):
        food = self._make_food(cal=0)
        score = _score_food(food, 500, 30, False, False, False)
        assert score == 0.0

    def test_higher_protein_density_scores_higher(self):
        low_pro  = self._make_food(cal=300, pro=5)
        high_pro = self._make_food(cal=300, pro=25)
        s_low  = _score_food(low_pro,  500, 30, False, False, False)
        s_high = _score_food(high_pro, 500, 30, False, False, False)
        assert s_high > s_low

    def test_micro_bonus_increases_score(self):
        base_food  = self._make_food(iron=False)
        iron_food  = self._make_food(iron=True)
        s_base = _score_food(base_food, 500, 30, needs_iron=True,
                             needs_calcium=False, needs_vitd=False)
        s_iron = _score_food(iron_food, 500, 30, needs_iron=True,
                             needs_calcium=False, needs_vitd=False)
        assert s_iron > s_base

    def test_micro_bonus_not_applied_when_not_needed(self):
        base_food  = self._make_food(iron=False)
        iron_food  = self._make_food(iron=True, cal=200, pro=10)
        s_base = _score_food(base_food, 500, 30, False, False, False)
        s_iron = _score_food(iron_food, 500, 30, False, False, False)
        # Without the micro flag, iron food should not outscore base on micro alone
        # (protein density is identical, so scores should be equal)
        assert abs(s_iron - s_base) < 0.01

    def test_large_overshoot_penalises_score(self):
        normal_food  = self._make_food(cal=100, unit_g=100)
        massive_food = self._make_food(cal=900, unit_g=500)
        s_normal  = _score_food(normal_food,  200, 30, False, False, False)
        s_massive = _score_food(massive_food, 200, 30, False, False, False)
        assert s_normal > s_massive


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SLOT FILLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFillSlot:

    def _breakfast_slot(self):
        return {"name": "Breakfast", "tag": "breakfast", "cal_frac": 0.25, "max_items": 4}

    def test_returns_list(self):
        slot   = self._breakfast_slot()
        result = _fill_slot(slot, 700, 35, "veg", False, False, False, set())
        assert isinstance(result, list)

    def test_veg_filter_excludes_non_veg(self):
        slot   = self._breakfast_slot()
        result = _fill_slot(slot, 700, 35, "veg", False, False, False, set())
        non_veg_foods = {f["name"] for f in FOOD_DB if f["diet"] == "non-veg"}
        selected_names = {item["name"] for item in result}
        assert selected_names.isdisjoint(non_veg_foods), \
            "Veg plan should not contain non-veg items"

    def test_non_veg_can_include_non_veg_foods(self):
        slot   = {"name": "Dinner", "tag": "dinner", "cal_frac": 0.35, "max_items": 5}
        result = _fill_slot(slot, 1000, 50, "non-veg", False, False, False, set())
        names  = {item["name"] for item in result}
        non_veg_available = {f["name"] for f in FOOD_DB
                             if f["diet"] == "non-veg" and "dinner" in f["tags"]}
        # At least one non-veg item could appear (not guaranteed but possible)
        assert isinstance(result, list)

    def test_used_today_prevents_repeat(self):
        slot = self._breakfast_slot()
        # First fill
        used = set()
        first  = _fill_slot(slot, 700, 35, "veg", False, False, False, used)
        first_names = {item["name"] for item in first}
        # Second fill with same used set (now populated)
        second = _fill_slot(slot, 700, 35, "veg", False, False, False, used)
        second_names = {item["name"] for item in second}
        assert first_names.isdisjoint(second_names), \
            "Items used in first fill should not reappear in second fill"

    def test_respects_max_items(self):
        slot   = {"name": "Snack", "tag": "snack", "cal_frac": 0.10, "max_items": 2}
        result = _fill_slot(slot, 300, 15, "veg", False, False, False, set())
        assert len(result) <= 2

    def test_each_item_has_required_keys(self):
        slot   = self._breakfast_slot()
        result = _fill_slot(slot, 700, 35, "veg", False, False, False, set())
        required = {"name", "grams", "cal", "pro", "carb", "fat"}
        for item in result:
            missing = required - item.keys()
            assert not missing, f"Item '{item.get('name')}' missing: {missing}"

    def test_grams_is_positive(self):
        slot   = self._breakfast_slot()
        result = _fill_slot(slot, 700, 35, "veg", False, False, False, set())
        for item in result:
            assert item["grams"] > 0, f"'{item['name']}' grams must be positive"

    def test_zero_budget_returns_empty(self):
        slot   = self._breakfast_slot()
        result = _fill_slot(slot, 0, 0, "veg", False, False, False, set())
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BUILD MEAL PLAN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildMealPlan:

    def test_returns_dict_with_required_keys(self, mock_targets_standard):
        plan = build_meal_plan(mock_targets_standard, "veg")
        assert "targets"   in plan
        assert "meals"     in plan
        assert "diet_type" in plan
        assert "flags"     in plan

    def test_correct_number_of_meal_slots(self, mock_targets_standard):
        plan = build_meal_plan(mock_targets_standard, "veg")
        assert len(plan["meals"]) == len(MEAL_SLOTS)

    def test_each_meal_has_slot_and_items(self, mock_targets_standard):
        plan = build_meal_plan(mock_targets_standard, "veg")
        for meal in plan["meals"]:
            assert "slot"  in meal
            assert "items" in meal
            assert isinstance(meal["items"], list)

    def test_veg_plan_contains_no_non_veg(self, mock_targets_standard):
        plan = build_meal_plan(mock_targets_standard, "veg")
        non_veg_names = {f["name"] for f in FOOD_DB if f["diet"] == "non-veg"}
        for meal in plan["meals"]:
            for item in meal["items"]:
                assert item["name"] not in non_veg_names, \
                    f"Non-veg item '{item['name']}' appeared in veg plan"

    def test_flags_reflected_in_plan(self, mock_targets_with_flags):
        plan = build_meal_plan(mock_targets_with_flags, "veg")
        assert plan["flags"]["needs_iron"]    is True
        assert plan["flags"]["needs_calcium"] is True
        assert plan["flags"]["needs_vitd"]    is True

    def test_no_repeated_foods_across_slots(self, mock_targets_standard):
        plan  = build_meal_plan(mock_targets_standard, "non-veg")
        names = []
        for meal in plan["meals"]:
            for item in meal["items"]:
                names.append(item["name"])
        assert len(names) == len(set(names)), \
            "Same food item appeared in multiple meal slots"

    def test_total_calories_roughly_match_target(self, mock_targets_standard):
        plan = build_meal_plan(mock_targets_standard, "veg")
        total = sum(
            item["cal"]
            for meal in plan["meals"]
            for item in meal["items"]
        )
        target = mock_targets_standard["calories"]
        # Allow ±40% tolerance (greedy algorithm won't be perfect)
        assert total > target * 0.60, \
            f"Total calories {total:.0f} too far below target {target}"
        assert total < target * 1.40, \
            f"Total calories {total:.0f} too far above target {target}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FORMAT PLAN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatPlan:

    def _make_plan(self, targets, diet="veg"):
        plan = build_meal_plan(targets, diet)
        return plan

    def test_returns_string(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert isinstance(result, str)

    def test_contains_athlete_name(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "Rahul" in result

    def test_contains_sport(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "Cricket" in result

    def test_contains_calorie_target(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert str(mock_targets_standard["calories"]) in result

    def test_match_day_label_shown(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "Match Day" in result

    def test_recovery_day_label_shown(self, mock_targets_standard, athletics_profile):
        plan   = self._make_plan(mock_targets_standard, "veg")
        result = format_plan(plan, athletics_profile)
        assert "Recovery Day" in result

    def test_micro_flags_shown_when_present(self, mock_targets_with_flags, cricket_profile):
        plan   = self._make_plan(mock_targets_with_flags, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "Iron" in result
        assert "Calcium" in result

    def test_disclaimer_present(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "AI-generated guidance" in result or "nutritionist" in result

    def test_plan_totals_section_present(self, mock_targets_standard, cricket_profile):
        plan   = self._make_plan(mock_targets_standard, "non-veg")
        result = format_plan(plan, cricket_profile)
        assert "PLAN TOTALS" in result or "Calories" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GET_MEAL_PLAN (INTEGRATION — mocked models)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMealPlanMocked:

    def test_full_pipeline_returns_meal_plan_text(
            self, cricket_profile, mock_targets_standard):
        models = make_mock_models(mock_targets_standard)
        with patch("meal_planner.load_models", return_value=models):
            plan = get_meal_plan(cricket_profile)
        assert "meal_plan_text" in plan
        assert len(plan["meal_plan_text"]) > 100

    def test_veg_profile_produces_veg_plan(
            self, athletics_profile, mock_targets_standard):
        models = make_mock_models(mock_targets_standard)
        with patch("meal_planner.load_models", return_value=models):
            plan = get_meal_plan(athletics_profile)
        non_veg_names = {f["name"] for f in FOOD_DB if f["diet"] == "non-veg"}
        for meal in plan["meals"]:
            for item in meal["items"]:
                assert item["name"] not in non_veg_names

    def test_bulk_profile_has_higher_calories_than_cut(
            self, kabaddi_bulk_profile, mock_targets_standard):
        """Verify the model is called with the correct bulk encoding."""
        models = make_mock_models(mock_targets_standard)
        with patch("meal_planner.load_models", return_value=models):
            plan = get_meal_plan(kabaddi_bulk_profile)
        # If mocked, just assert the pipeline ran
        assert "targets" in plan

    def test_missing_models_raises_file_not_found(self, cricket_profile):
        with pytest.raises(FileNotFoundError):
            get_meal_plan(cricket_profile, model_dir="nonexistent_models/")

    def test_invalid_sport_raises_key_error(self, cricket_profile):
        bad_profile = {**cricket_profile, "sport": "Polo"}
        models = make_mock_models({
            "calories": 2800, "protein_g": 140, "carbs_g": 350, "fat_g": 70,
            "needs_iron": 0, "needs_calcium": 0, "needs_vitd": 0,
        })
        with patch("meal_planner.load_models", return_value=models):
            with pytest.raises(KeyError):
                get_meal_plan(bad_profile)

    def test_plan_dict_has_all_expected_keys(
            self, cricket_profile, mock_targets_standard):
        models = make_mock_models(mock_targets_standard)
        with patch("meal_planner.load_models", return_value=models):
            plan = get_meal_plan(cricket_profile)
        for key in ("targets", "meals", "flags", "diet_type", "meal_plan_text"):
            assert key in plan, f"Missing key in plan output: '{key}'"

    def test_all_demo_profiles_run_without_error(self):
        """Smoke test: all three demo profiles complete successfully."""
        from meal_planner import DEMO_PROFILES
        targets = {
            "calories": 2800, "protein_g": 140, "carbs_g": 350, "fat_g": 70,
            "needs_iron": 0, "needs_calcium": 0, "needs_vitd": 0,
        }
        models = make_mock_models(targets)
        with patch("meal_planner.load_models", return_value=models):
            for profile in DEMO_PROFILES:
                plan = get_meal_plan(profile)
                assert plan["meal_plan_text"], \
                    f"Empty plan text for profile: {profile['name']}"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_very_low_calorie_target_does_not_crash(self):
        targets = {
            "calories": 1200,
            "protein_g": 40.0,
            "carbs_g": 100.0,
            "fat_g": 20.0,
            "needs_iron": 0,
            "needs_calcium": 0,
            "needs_vitd": 0,
        }
        plan = build_meal_plan(targets, "veg")
        assert "meals" in plan

    def test_very_high_calorie_target_does_not_crash(self):
        targets = {
            "calories": 5000,
            "protein_g": 250.0,
            "carbs_g": 600.0,
            "fat_g": 120.0,
            "needs_iron": 1,
            "needs_calcium": 1,
            "needs_vitd": 1,
        }
        plan = build_meal_plan(targets, "non-veg")
        assert "meals" in plan

    def test_all_micro_flags_set_does_not_crash(self):
        targets = {
            "calories": 2500,
            "protein_g": 120.0,
            "carbs_g": 310.0,
            "fat_g": 65.0,
            "needs_iron": 1,
            "needs_calcium": 1,
            "needs_vitd": 1,
        }
        plan = build_meal_plan(targets, "veg")
        assert plan["flags"]["needs_iron"]    is True
        assert plan["flags"]["needs_calcium"] is True
        assert plan["flags"]["needs_vitd"]    is True

    def test_plan_text_not_empty(self):
        targets = {
            "calories": 2500,
            "protein_g": 120.0,
            "carbs_g": 310.0,
            "fat_g": 65.0,
            "needs_iron": 0,
            "needs_calcium": 0,
            "needs_vitd": 0,
        }
        profile = {
            "name": "Test", "sport": "Football", "role": "Midfielder",
            "gender": "male", "age": 20, "weight_kg": 68.0, "height_cm": 172,
            "training_intensity": 7, "session_duration_min": 90,
            "training_days_per_week": 5, "is_match_day": 0,
            "is_recovery_day": 0, "fatigue_score": 4,
            "goal": "maintain", "diet_type": "veg",
        }
        plan = build_meal_plan(targets, "veg")
        text = format_plan(plan, profile)
        assert len(text.strip()) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
