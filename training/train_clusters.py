"""
Train KMeans clustering model for cold-start restaurants.
Groups restaurants by location, cuisine, and average daily quantity.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from supabase import create_client
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.restaurant_metadata import deterministic_restaurant_metadata, cuisine_to_label

load_dotenv()

# Ensure models directory exists
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_clusters():
    """
    Train KMeans clustering model on restaurant features.

    Feature vector per restaurant:
    - latitude, longitude (from location PostGIS point)
    - cuisine_type (label encoded)
    - avg_daily_quantity (computed from sales_logs)
    """
    print("[NutriLoop] Starting KMeans clustering training")

    client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )

    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print("[NutriLoop] ERROR: SUPABASE_URL and SUPABASE_KEY must be set")
        print("[NutriLoop] Copy .env.example to .env and fill in your Supabase credentials")
        return 0, {}

    # Load all unique restaurants from sales_logs
    # We derive avg_daily_quantity from the sales data
    print("[NutriLoop] Fetching restaurant data from Supabase")
    try:
        response = client.table("sales_logs").select("restaurant_id, quantity, sale_date").execute()
    except Exception as e:
        print(f"[NutriLoop] ERROR: Could not load sales_logs from Supabase: {e}")
        print("[NutriLoop] Make sure the sales_logs table exists and the Supabase schema has been created.")
        return 0, {}
    df = pd.DataFrame(response.data)

    if df.empty:
        print("[NutriLoop] No data in sales_logs. Run seed_supabase.py first.")
        return 0, {}

    # Compute avg_daily_quantity per restaurant
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

    # Get date range for normalization
    date_range_days = (df["sale_date"].max() - df["sale_date"].min()).days + 1
    restaurant_stats = df.groupby("restaurant_id").agg(
        total_quantity=("quantity", "sum"),
        avg_daily_quantity=("quantity", "mean"),
    ).reset_index()

    unique_restaurants = restaurant_stats["restaurant_id"].unique()
    n_restaurants = len(unique_restaurants)

    print(f"[NutriLoop] Clustering {n_restaurants} restaurants")

    if n_restaurants < 2:
        print("[NutriLoop] Not enough restaurants for clustering, using default model")
        n_clusters = 1
    else:
        n_clusters = min(10, n_restaurants)

    # Build feature matrix from real restaurant metadata when available.
    # If the optional restaurants table is missing, use deterministic fallback values.
    features = []
    labels = []

    restaurant_metadata_by_id: dict[str, object] = {}
    try:
        metadata_response = client.table("restaurants").select("*").execute()
        metadata_rows = getattr(metadata_response, "data", None) or []
        for row in metadata_rows:
            restaurant_id = str(row.get("restaurant_id"))
            if restaurant_id:
                restaurant_metadata_by_id[restaurant_id] = row
    except Exception:
        restaurant_metadata_by_id = {}

    for _, row in restaurant_stats.iterrows():
        restaurant_id = str(row["restaurant_id"])
        avg_qty = float(row["avg_daily_quantity"])

        metadata_row = restaurant_metadata_by_id.get(restaurant_id)
        if metadata_row is None:
            metadata = deterministic_restaurant_metadata(restaurant_id, avg_daily_quantity=avg_qty)
            latitude = metadata.latitude
            longitude = metadata.longitude
            cuisine_type = metadata.cuisine_type
        else:
            latitude = float(metadata_row.get("latitude") or deterministic_restaurant_metadata(restaurant_id, avg_qty).latitude)
            longitude = float(metadata_row.get("longitude") or deterministic_restaurant_metadata(restaurant_id, avg_qty).longitude)
            cuisine_type = metadata_row.get("cuisine_type") or deterministic_restaurant_metadata(restaurant_id, avg_qty).cuisine_type

        cuisine_label = cuisine_to_label(cuisine_type)

        features.append([latitude, longitude, cuisine_label, avg_qty])
        labels.append(restaurant_id)

    X = np.array(features)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    print(f"[NutriLoop] Fitting KMeans with n_clusters={n_clusters}")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)

    # Build cluster map
    cluster_map: dict[str, list[str]] = {str(i): [] for i in range(n_clusters)}
    for label, cluster_id in zip(labels, cluster_labels):
        cluster_map[str(cluster_id)].append(label)

    # Save model, scaler, and cluster map
    joblib.dump(model, MODELS_DIR / "cluster_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.pkl")

    with open(MODELS_DIR / "cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    print(f"[NutriLoop] Cluster model saved: {n_clusters} clusters")
    print(f"[NutriLoop] Cluster distribution: { {k: len(v) for k, v in cluster_map.items()} }")

    return n_clusters, cluster_map


if __name__ == "__main__":
    n_clusters, cluster_map = train_clusters()
    print(f"[NutriLoop] Clustering complete: {n_clusters} clusters created")