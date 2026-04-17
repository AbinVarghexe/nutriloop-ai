"""
Seed Supabase with Kaggle restaurant sales data.
Creates the database schema and uploads initial data.
Run this once after setting up Supabase tables.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from supabase import create_client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.restaurant_metadata import deterministic_restaurant_metadata


def create_tables_sql() -> str:
    """
    SQL to create required Supabase tables.
    Run this in the Supabase SQL editor or via migration.
    """
    return """
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Table: sales_logs
-- Stores individual restaurant sales transactions
CREATE TABLE IF NOT EXISTS sales_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    restaurant_id text NOT NULL,
    item_name text NOT NULL,
    quantity integer NOT NULL,
    sale_date date NOT NULL,
    location geography(Point, 4326),
    created_at timestamptz DEFAULT now()
);

-- Index for faster queries by restaurant_id
CREATE INDEX IF NOT EXISTS idx_sales_logs_restaurant_id ON sales_logs(restaurant_id);

-- Index for faster queries by sale_date
CREATE INDEX IF NOT EXISTS idx_sales_logs_sale_date ON sales_logs(sale_date);

-- Composite index for restaurant + item queries
CREATE INDEX IF NOT EXISTS idx_sales_logs_restaurant_item ON sales_logs(restaurant_id, item_name);

-- Table: retrain_log
-- Tracks model retraining runs
CREATE TABLE IF NOT EXISTS retrain_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_at timestamptz DEFAULT now(),
    model_version text NOT NULL,
    rows_used integer,
    mae_score float,
    status text CHECK (status IN ('success', 'failed')),
    error_msg text
);

-- Table: restaurants
-- Stores restaurant metadata used by cold-start prediction and clustering
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id text PRIMARY KEY,
    restaurant_name text,
    latitude double precision,
    longitude double precision,
    cuisine_type text,
    avg_daily_quantity double precision,
    created_at timestamptz DEFAULT now()
);

-- Table: profiles
-- Stores user profile information and app settings
CREATE TABLE IF NOT EXISTS profiles (
    id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email text,
    name text,
    role text CHECK (role IN ('restaurant', 'ngo', 'individual', 'admin')),
    city text,
    lat double precision,
    lng double precision,
    created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE sales_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE retrain_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE restaurants ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Allow anon access (adjust for production)
CREATE POLICY "Allow anon read" ON sales_logs FOR SELECT USING (true);
CREATE POLICY "Allow anon insert" ON sales_logs FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow anon read" ON retrain_log FOR SELECT USING (true);
CREATE POLICY "Allow anon insert" ON retrain_log FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow anon read" ON restaurants FOR SELECT USING (true);
CREATE POLICY "Allow anon insert" ON restaurants FOR INSERT WITH CHECK (true);

-- Profile Policies
CREATE POLICY "Users can manage their own profile" ON profiles 
FOR ALL 
TO authenticated 
USING (auth.uid() = id) 
WITH CHECK (auth.uid() = id);

CREATE POLICY "Profiles are viewable by all authenticated users" ON profiles 
FOR SELECT 
TO authenticated 
USING (true);
"""


def seed_from_csv(csv_path: str) -> None:
    """
    Load Kaggle CSV and seed Supabase sales_logs table.

    Args:
        csv_path: Path to the Kaggle CSV file
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("[NutriLoop] ERROR: SUPABASE_URL and SUPABASE_KEY must be set")
        print("[NutriLoop] Copy .env.example to .env and fill in your Supabase credentials")
        sys.exit(1)

    # First, load and clean the Kaggle data
    from training.load_kaggle_data import load_kaggle_data

    df = load_kaggle_data(csv_path)

    print(f"[NutriLoop] Connecting to Supabase")
    client = create_client(supabase_url, supabase_key)

    # Seed restaurant metadata so clustering and fallback inference can use
    # deterministic, queryable location and cuisine features.
    restaurant_rows = []
    for restaurant_id, group_df in df.groupby("restaurant_id"):
        avg_daily_quantity = float(group_df["quantity"].mean())
        metadata = deterministic_restaurant_metadata(restaurant_id, avg_daily_quantity=avg_daily_quantity)
        restaurant_rows.append({
            "restaurant_id": metadata.restaurant_id,
            "restaurant_name": metadata.restaurant_id,
            "latitude": metadata.latitude,
            "longitude": metadata.longitude,
            "cuisine_type": metadata.cuisine_type,
            "avg_daily_quantity": metadata.avg_daily_quantity,
        })

    try:
        if restaurant_rows:
            client.table("restaurants").upsert(restaurant_rows, on_conflict="restaurant_id").execute()
            print(f"[NutriLoop] Upserted {len(restaurant_rows)} restaurant metadata rows")
    except Exception as e:
        print(f"[NutriLoop] WARNING: Could not upsert restaurant metadata: {e}")

    # Prepare rows for bulk insert
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "restaurant_id": str(row["restaurant_id"]),
            "item_name": str(row["item_name"]),
            "quantity": int(row["quantity"]),
            "sale_date": row["sale_date"].strftime("%Y-%m-%d"),
            "location": "POINT(76.2673 9.9312)",  # Default to Kochi, PostGIS format is lng lat
        })

    # Insert in batches to avoid payload limits
    batch_size = 500
    total_inserted = 0

    print(f"[NutriLoop] Inserting {len(rows)} rows into Supabase (batch size: {batch_size})")

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            response = client.table("sales_logs").insert(batch).execute()
            inserted = len(response.data) if response.data else len(batch)
            total_inserted += inserted
            print(f"[NutriLoop] Batch {i // batch_size + 1}: inserted {inserted} rows")
        except Exception as e:
            print(f"[NutriLoop] Batch {i // batch_size + 1} failed: {e}")
            # Try inserting one by one
            for row in batch:
                try:
                    client.table("sales_logs").insert(row).execute()
                    total_inserted += 1
                except Exception as row_err:
                    print(f"[NutriLoop] Row insert failed: {row_err}")

    print(f"[NutriLoop] Seed complete: {total_inserted} rows inserted into sales_logs")
    print("[NutriLoop] Run training/train_prophet.py next to train Prophet models")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed Supabase with Kaggle restaurant data")
    parser.add_argument("csv_path", nargs="?", help="Path to the Kaggle CSV file")
    parser.add_argument("--print-sql", action="store_true", help="Print SQL schema instead of seeding")
    args = parser.parse_args()

    if args.print_sql:
        print(create_tables_sql())
    else:
        if not args.csv_path:
            parser.error("csv_path is required unless --print-sql is used")

        if not Path(args.csv_path).exists():
            print(f"[NutriLoop] ERROR: File not found: {args.csv_path}")
            print("[NutriLoop] Download the Kaggle dataset from: https://www.kaggle.com/datasets/mer-sun/restaurant-sales")
            sys.exit(1)
        seed_from_csv(args.csv_path)