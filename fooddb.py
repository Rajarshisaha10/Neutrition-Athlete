"""
AthleteEdge AI food database loader.

The canonical food database lives in data/food_db.json so it can be reused by
training, planning, APIs, and notebooks without importing a large Python module.
"""

from __future__ import annotations

import json
from pathlib import Path


FOOD_DB_PATH = Path(__file__).resolve().parent / "data" / "food_db.json"

REQUIRED_FIELDS = {
    "name",
    "category",
    "serving_g",
    "cal",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "iron_mg",
    "calcium_mg",
    "vitd_iu",
    "is_veg",
    "meal_slots",
    "cost_inr",
}


def get_food_db(path: str | Path = FOOD_DB_PATH) -> list[dict]:
    """Load and validate the JSON food database."""
    db_path = Path(path)
    with db_path.open(encoding="utf-8") as fp:
        foods = json.load(fp)

    if not isinstance(foods, list):
        raise ValueError(f"{db_path} must contain a JSON list of foods")

    for index, food in enumerate(foods, start=1):
        if not isinstance(food, dict):
            raise ValueError(f"Food entry #{index} must be an object")

        missing = REQUIRED_FIELDS - food.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Food entry #{index} is missing: {missing_list}")

    return foods


if __name__ == "__main__":
    db = get_food_db()
    print(f"Food DB loaded: {len(db)} items from {FOOD_DB_PATH}")

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Install pandas to print the food database summary.")

    df = pd.DataFrame(db)
    print("\nCategory breakdown:")
    print(df["category"].value_counts())
    print("\nSample sorted by protein per serving:")
    print(
        df.sort_values("protein_g", ascending=False)[
            ["name", "serving_g", "cal", "protein_g", "carbs_g", "fat_g", "cost_inr"]
        ]
        .head(10)
        .to_string(index=False)
    )
