"""
Prophet model loading and inference logic for NutriLoop AI.
"""
import json
from pathlib import Path
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline

# Path to models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model() -> Optional[Pipeline]:
    """
    Load the global HistGradientBoostingRegressor model from disk if it exists.

    Returns:
        Sklearn Pipeline model instance or None if not found
    """
    model_path = MODELS_DIR / "global_model.pkl"

    if not model_path.exists():
        print(f"[NutriLoop] No model found at {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"[NutriLoop] Failed to load model {model_path}: {e}")
        return None


def load_model_registry() -> dict:
    """
    Load the model registry JSON tracking trained models and their MAE scores.

    Returns:
        Dictionary mapping restaurant_id__item_name to metadata
    """
    registry_path = MODELS_DIR / "model_registry.json"
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        return json.load(f)


def get_model_mae() -> float:
    """Get the MAE score for the global model."""
    registry = load_model_registry()
    return registry.get("global_model", {}).get("mae", 0.0)


def run_forecast(model: Pipeline, days: int, restaurant_id: str, item_name: str, 
                 latitude: float, longitude: float, cuisine_type: str, avg_daily_quantity: float) -> dict:
    """
    Generate a forecast for the specified number of days using the global model.

    Args:
        model: Trained global ML pipeline
        days: Number of days to forecast
        restaurant_id: The restaurant identifier
        item_name: The food item name
        latitude: Region latitude
        longitude: Region longitude
        cuisine_type: Restaurant cuisine type

    Returns:
        DataFrame with date and quantity columns
    """
    print(f"[NutriLoop] Running {days}-day Multivariate forecast for {restaurant_id}/{item_name}")
    
    # Needs to match features expected by train_global.py:
    # ["restaurant_id", "item_name", "cuisine_type", "day_of_week", "day_of_year", "month", "year", "is_holiday", "latitude", "longitude"]
    import pandas as pd
    from datetime import datetime, timedelta
    import holidays
    
    today = datetime.now()
    future_dates = [today + timedelta(days=i) for i in range(1, days + 1)]
    india_holidays = holidays.India(years=range(today.year, today.year+2))
    
    records = []
    for dt in future_dates:
        records.append({
            "restaurant_id": restaurant_id,
            "item_name": item_name,
            "cuisine_type": cuisine_type,
            "day_of_week": dt.weekday(),
            "day_of_year": dt.timetuple().tm_yday,
            "month": dt.month,
            "year": dt.year,
            "is_holiday": 1 if dt.date() in india_holidays else 0,
            "latitude": latitude,
            "longitude": longitude,
            "avg_daily_quantity": avg_daily_quantity
        })
        
        
    df_future = pd.DataFrame(records)
    predictions = model.predict(df_future)
    
    df_result = pd.DataFrame({
        "date": future_dates,
        "quantity": predictions
    })
    
    # Ensure non-negative bounds
    df_result["quantity"] = df_result["quantity"].clip(lower=0)
    
    return df_result