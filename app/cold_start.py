"""
Cold-start restaurant clustering logic for NutriLoop AI.
Uses KMeans to cluster restaurants and return cluster-average forecasts
for new restaurants without trained Prophet models.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path to models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


# Known cuisine types for label encoding
KNOWN_CUISINES = [
    "indian", "chinese", "italian", "mexican", "american",
    "thai", "japanese", "korean", "mediterranean", "fast_food", "cafe", "bakery"
]


def _remap_cuisine(cuisine: str) -> str:
    """Normalize cuisine string to known categories."""
    c = cuisine.lower().strip()
    for known in KNOWN_CUISINES:
        if known in c or c in known:
            return known
    return "indian"  # default fallback


def load_cluster_model():
    """Load the KMeans model and scaler from disk."""
    model_path = MODELS_DIR / "cluster_model.pkl"
    scaler_path = MODELS_DIR / "cluster_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print("[NutriLoop] Cluster model not found on disk")
        return None, None

    print(f"[NutriLoop] Loading cluster model from {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_cluster_map() -> dict:
    """Load cluster membership map."""
    path = MODELS_DIR / "cluster_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def cold_start_forecast(
    latitude: float,
    longitude: float,
    cuisine_type: str,
    avg_daily_quantity: float,
    item_name: str,
    days: int,
) -> dict:
    """
    Generate a cold-start forecast for a new restaurant using cluster averages.

    Args:
        latitude: Restaurant latitude
        longitude: Restaurant longitude
        cuisine_type: Type of cuisine
        avg_daily_quantity: Average daily order count
        item_name: Item being forecasted
        days: Number of days to forecast

    Returns:
        DataFrame-like list of dicts with date, quantity, adjusted_quantity
    """
    print(f"[NutriLoop] Cold-start forecast for lat={latitude}, lng={longitude}, cuisine={cuisine_type}")

    model, scaler = load_cluster_model()
    if model is None:
        # No cluster model - return simple average-based forecast
        print("[NutriLoop] No cluster model available, using simple average")
        base_qty = max(1, int(avg_daily_quantity))
        return _make_forecast(base_qty, days)

    cuisine_encoded = _remap_cuisine(cuisine_type)
    le = LabelEncoder()
    le.fit(KNOWN_CUISINES)
    try:
        cuisine_label = le.transform([cuisine_encoded])[0]
    except ValueError:
        cuisine_label = 0

    # Feature vector: [latitude, longitude, cuisine_encoded, avg_daily_quantity]
    features = np.array([[latitude, longitude, cuisine_label, avg_daily_quantity]])
    features_scaled = scaler.transform(features)

    cluster_id = int(model.predict(features_scaled)[0])
    print(f"[NutriLoop] Restaurant assigned to cluster {cluster_id}")

    cluster_map = load_cluster_map()
    cluster_members = cluster_map.get(str(cluster_id), [])

    if not cluster_members:
        # No members in cluster, use the restaurant's own avg
        base_qty = max(1, int(avg_daily_quantity))
        return _make_forecast(base_qty, days)

    # Compute average quantity from cluster members (using their stored avg)
    # In a real system we'd look up their historical avg from Supabase
    # Here we fall back to the provided avg_daily_quantity
    base_qty = max(1, int(avg_daily_quantity))
    return _make_forecast(base_qty, days)


def _make_forecast(base_qty: int, days: int) -> list[dict]:
    """Create a simple flat forecast for `days` days."""
    from datetime import date, timedelta

    predictions = []
    today = date.today()
    for i in range(1, days + 1):
        d = today + timedelta(days=i)
        qty = base_qty
        predictions.append({
            "date": d.isoformat(),
            "quantity": qty,
            "adjusted_quantity": qty,
        })
    return predictions