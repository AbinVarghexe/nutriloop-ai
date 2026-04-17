"""
Train a Global Multivariate Machine Learning model for NutriLoop AI.
Predicts quantity based on restaurant_id, item_name, region (lat/lon), and time elements.
Replaces the old univariate Prophet models.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import holidays
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from supabase import create_client
from dotenv import load_dotenv

# Ensure models directory exists
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Allow direct execution from the project root and load local env vars.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()


def get_india_holidays(years=range(2020, 2028)) -> set:
    """Return a set of holiday dates for India."""
    in_holidays = holidays.India(years=years)
    return set(in_holidays.keys())


def extract_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """Extract temporal features from a DataFrame containing 'sale_date'."""
    df = df.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['day_of_year'] = df['sale_date'].dt.dayofyear
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['is_holiday'] = df['sale_date'].dt.date.isin(holiday_dates).astype(int)
    return df


def train_global_model():
    """
    Main training loop for the Global Multivariate Model.
    """
    print("[NutriLoop] Starting Global Model training")

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("[NutriLoop] ERROR: SUPABASE_URL and SUPABASE_KEY must be set")
        return False, {}

    client = create_client(supabase_url, supabase_key)

    print("[NutriLoop] Fetching sales_logs from Supabase")
    try:
        response_sales = client.table("sales_logs").select("*").execute()
        response_restaurants = client.table("restaurants").select("*").execute()
    except Exception as e:
        print(f"[NutriLoop] ERROR: Could not load data from Supabase: {e}")
        return False, {}

    sales_df = pd.DataFrame(response_sales.data)
    rests_df = pd.DataFrame(response_restaurants.data)

    if sales_df.empty:
        print("[NutriLoop] No data in sales_logs.")
        return False, {}

    print(f"[NutriLoop] Merging {len(sales_df)} sales logs with {len(rests_df)} restaurants")
    
    # Merge sales with restaurant geographic data
    # Fallback missing data handles appropriately
    if not rests_df.empty:
        # Avoid column conflicts, keep only what we need from restaurants
        rests_df = rests_df[['restaurant_id', 'latitude', 'longitude', 'cuisine_type', 'avg_daily_quantity']]
        df = pd.merge(sales_df, rests_df, on="restaurant_id", how="left")
    else:
        df = sales_df
        df['latitude'] = 0.0
        df['longitude'] = 0.0
        df['cuisine_type'] = "Unknown"
        df['avg_daily_quantity'] = 0.0

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

    # Feature Engineering
    holiday_dates = get_india_holidays()
    df = extract_features(df, holiday_dates)
    
    # Fill any null latitude/longitudes with global averages just in case
    df['latitude'] = df['latitude'].fillna(0.0)
    df['longitude'] = df['longitude'].fillna(0.0)
    df['cuisine_type'] = df['cuisine_type'].fillna('Unknown')
    df['avg_daily_quantity'] = df['avg_daily_quantity'].fillna(0.0)
    
    # Sort chronologically for valid temporal holdout
    df.sort_values("sale_date", inplace=True)
    
    # Establish a 14-day holdout validation set
    holdout_days = 14
    cutoff_date = df["sale_date"].max() - timedelta(days=holdout_days)
    
    train_df = df[df["sale_date"] < cutoff_date].copy()
    holdout_df = df[df["sale_date"] >= cutoff_date].copy()
    
    if len(train_df) < 50:
        print("[NutriLoop] Insufficient training data.")
        return False, {}

    print(f"[NutriLoop] Training on : {len(train_df)} rows, Validation: {len(holdout_df)} rows")

    # Define the Model Pipeline
    categorical_features = ["restaurant_id", "item_name", "cuisine_type"]
    numeric_features = ["day_of_week", "day_of_year", "month", "year", "is_holiday", "latitude", "longitude", "avg_daily_quantity"]
    all_features = categorical_features + numeric_features

    X_train = train_df[all_features]
    y_train = train_df["quantity"]
    
    X_val = holdout_df[all_features]
    y_val = holdout_df["quantity"]

    # Preprocessor encodes strings to integers for HGBR native categorical support
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat", 
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), 
                categorical_features
            )
        ],
        remainder="passthrough", # Keep numeric as is
        verbose_feature_names_out=False
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=150,
            categorical_features=list(range(len(categorical_features))),
            loss='absolute_error', # MAE focus
            random_state=42
        ))
    ])

    print("[NutriLoop] Starting Fit...")
    model.fit(X_train, y_train)

    # Evaluation
    mae = 0.0
    if len(X_val) > 0:
        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))

    print(f"[NutriLoop] Model Trained! Global Validation MAE: {mae:.2f}")

    # Artifact generation
    model_path = MODELS_DIR / "global_model.pkl"
    joblib.dump(model, model_path)
    print(f"[NutriLoop] Saved Model: {model_path}")

    model_registry = {
        "global_model": {
            "trained_at": datetime.now().isoformat(),
            "mae": round(mae, 4),
            "rows_used": len(df),
            "features": all_features,
            "algorithms": "HistGradientBoostingRegressor"
        }
    }
    
    registry_path = MODELS_DIR / "model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(model_registry, f, indent=2)

    # Log to Supabase
    try:
        client.table("retrain_log").insert({
            "model_version": datetime.now().isoformat(),
            "rows_used": len(df),
            "mae_score": mae,
            "status": "success",
        }).execute()
    except Exception as e:
        print(f"[NutriLoop] Failed to log to Supabase retrain_log: {e}")

    return True, model_registry


if __name__ == "__main__":
    success, registry = train_global_model()
    if success:
        print(f"\n[NutriLoop Summary] Global Multivariate Model Trained | Valid MAE: {registry['global_model']['mae']}")
