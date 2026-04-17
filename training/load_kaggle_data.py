"""
Load and clean Kaggle restaurant sales data.
Accepts a CSV file path, parses and validates the data, returns a clean DataFrame.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


# Kaggle "Restaurant Sales" / food waste tracker column mapping
# Expected columns: date, item_name, quantity, restaurant_id
# Auto-detects both formats:
#   - Restaurant Sales: date, item_name, quantity, restaurant_id
#   - Food Waste Tracker: Date, Food Name, Quantity Purchased (kg), Location
COLUMN_MAPPING = {
    "date": "date",
    "item_name": "item_name",
    "quantity": "quantity",
    "restaurant_id": "restaurant_id",
}

# Food Waste Tracker specific mapping (detected by "Food Category" column)
FOOD_WASTE_MAPPING = {
    "Date": "date",
    "Food Name": "item_name",
    "Quantity Purchased (kg)": "quantity",
    "Location": "restaurant_id",  # Treat each location as a "restaurant"
}


def _detect_and_remap(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect CSV format and remap columns to standard names."""
    cols = set(df.columns)

    # Detect Food Waste Tracker format
    if "Food Category" in cols and "Food Name" in cols:
        print("[NutriLoop] Detected: Food Waste Tracker format")
        mapping = FOOD_WASTE_MAPPING
    else:
        # Standard column mapping
        mapping = {}
        for k, v in COLUMN_MAPPING.items():
            # Direct match
            if v in df.columns:
                mapping[v] = v
            else:
                # Case-insensitive match
                for col in df.columns:
                    if col.lower().strip() == k.lower():
                        mapping[col] = v
                        break
        # Check required columns
        required = ["restaurant_id", "item_name", "quantity", "date"]
        missing = [c for c in required if c not in mapping.values()]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
        return df.rename(columns=mapping)

    # Apply Food Waste mapping
    df = df.rename(columns=mapping)
    return df


def load_kaggle_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean Kaggle restaurant sales CSV.

    Args:
        csv_path: Path to the Kaggle CSV file

    Returns:
        Clean pandas DataFrame with columns: restaurant_id, item_name, quantity, sale_date
    """
    print(f"[NutriLoop] Loading Kaggle data from {csv_path}")
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Try to detect the delimiter
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if ";" in first_line:
        delimiter = ";"
    elif "," in first_line:
        delimiter = ","
    else:
        delimiter = ","

    print(f"[NutriLoop] Detected delimiter: '{delimiter}'")

    # Read CSV
    df = pd.read_csv(path, delimiter=delimiter)
    print(f"[NutriLoop] Raw CSV columns: {list(df.columns)}")

    # Auto-detect format and remap columns
    df = _detect_and_remap(df)

    # Parse dates
    print(f"[NutriLoop] Parsing dates (sample: {df['date'].iloc[0]})")
    df["sale_date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid dates
    invalid_dates = df["sale_date"].isna().sum()
    if invalid_dates > 0:
        print(f"[NutriLoop] Dropping {invalid_dates} rows with invalid dates")
    df = df.dropna(subset=["sale_date"])

    # Ensure quantity is numeric (kg floats from food waste data → grams as int)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["quantity"] = (df["quantity"] * 1000).round().astype(int)  # kg → grams

    # Drop rows with null restaurant_id or item_name
    before = len(df)
    df = df.dropna(subset=["restaurant_id", "item_name"])
    after = len(df)
    if before - after > 0:
        print(f"[NutriLoop] Dropped {before - after} rows with null restaurant_id or item_name")

    # Clean strings
    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    df["item_name"] = df["item_name"].astype(str).str.strip()

    # Sort by date
    df = df.sort_values("sale_date")

    print(f"[NutriLoop] Clean DataFrame: {len(df)} rows, date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    print(f"[NutriLoop] Unique restaurants: {df['restaurant_id'].nunique()}")
    print(f"[NutriLoop] Unique items: {df['item_name'].nunique()}")

    # Return only required columns
    return df[["restaurant_id", "item_name", "quantity", "sale_date"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect Kaggle restaurant sales data")
    parser.add_argument("csv_path", help="Path to the Kaggle CSV file")
    parser.add_argument("--head", type=int, default=20, help="Number of rows to display")
    args = parser.parse_args()

    df = load_kaggle_data(args.csv_path)
    print("\nFirst rows:")
    print(df.head(args.head).to_string(index=False))
    print(f"\nTotal rows: {len(df)}")