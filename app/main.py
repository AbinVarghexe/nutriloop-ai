"""
FastAPI application entry point for NutriLoop AI.
Provides /health, /predict, and /cold-start endpoints.
"""
from contextlib import asynccontextmanager
import os
import pandas as pd
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.schemas import (
    ColdStartRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionPoint,
)
from app.predict import get_model_mae, load_model, run_forecast
from app.cold_start import cold_start_forecast
from app.news_adjuster import get_news_multiplier
from app.restaurant_metadata import create_supabase_client_from_env, load_restaurant_metadata

# Load env vars on startup
load_dotenv()

# Ensure models directory exists
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Version
VERSION = "0.1.0"

_supabase_client = create_supabase_client_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - runs on startup and shutdown."""
    print("[NutriLoop] Starting NutriLoop AI server")
    print(f"[NutriLoop] Models directory: {MODELS_DIR}")
    yield
    print("[NutriLoop] Shutting down NutriLoop AI server")


app = FastAPI(
    title="NutriLoop AI",
    description="Food demand forecasting, cold-start clustering, and news-adjusted predictions",
    version=VERSION,
    lifespan=lifespan,
)

# Allow CORS for Next.js dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns server status, model count, last retrain time, and version.
    """
    prophet_model_count = 0
    cluster_model_present = False
    if MODELS_DIR.exists():
        prophet_model_count = len(list(MODELS_DIR.glob("*__*.pkl")))
        cluster_model_present = (MODELS_DIR / "cluster_model.pkl").exists()

    config_valid = bool(
        os.environ.get("SUPABASE_URL") and
        os.environ.get("SUPABASE_KEY") and
        os.environ.get("HF_TOKEN") and
        os.environ.get("HF_REPO_ID")
    )

    last_retrain_val = None
    last_retrain_file = MODELS_DIR / "last_retrain.txt"
    if last_retrain_file.exists():
        try:
            last_retrain_val = last_retrain_file.read_text().strip()
        except Exception:
            pass

    global_model_path = Path("models/global_model.pkl")
    global_model_present = global_model_path.exists()
    
    cluster_model_path = Path("models/cluster_model.pkl")
    cluster_model_present = cluster_model_path.exists()

    return HealthResponse(
        status="ok",
        global_model_present=global_model_present,
        cluster_model_present=cluster_model_present,
        config_valid=config_valid,
        last_retrain=last_retrain_val,
        version=VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate a demand forecast for a restaurant item.

    Logic:
    1. Try to load a trained Prophet model for restaurant_id + item_name
    2. If not found, fall back to cold_start logic
    3. Run Prophet prediction for `days` horizon
    4. Apply news-based adjustment multiplier
    """
    print(f"[NutriLoop] /predict for restaurant={request.restaurant_id}, item={request.item_name}")

    # Try Global Model first
    model = load_model()
    source = "global_model"
    mae = 0.0

    metadata = load_restaurant_metadata(_supabase_client, request.restaurant_id)

    if model is None:
        # Fall back to cold-start clustering if no global model exists
        print(f"[NutriLoop] No Global model available, using cold-start for {request.restaurant_id}/{request.item_name}")
        source = "cold_start"
        try:
            cold_preds = cold_start_forecast(
                latitude=metadata.latitude,
                longitude=metadata.longitude,
                cuisine_type=metadata.cuisine_type,
                avg_daily_quantity=metadata.avg_daily_quantity,
                item_name=request.item_name,
                days=request.days,
            )
        except Exception as e:
            print(f"[NutriLoop] Error running fallback cold start: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate cold-start forecast: {e}")

        if not cold_preds:
            raise HTTPException(status_code=500, detail="Cold-start forecast returned empty results.")

        news_mult = get_news_multiplier(request.city)

        predictions = []
        for p in cold_preds:
            adj_qty = max(1, round(p["quantity"] * news_mult))
            predictions.append(PredictionPoint(
                date=p["date"],
                quantity=p["quantity"],
                adjusted_quantity=adj_qty,
            ))

        return PredictResponse(
            restaurant_id=request.restaurant_id,
            item_name=request.item_name,
            predictions=predictions,
            news_multiplier=news_mult,
            model_mae=mae,
            source=source,
        )
    else:
        # We have the global model
        try:
            mae = get_model_mae()
            preds_df = run_forecast(
                model=model, 
                days=request.days, 
                restaurant_id=request.restaurant_id, 
                item_name=request.item_name,
                latitude=metadata.latitude,
                longitude=metadata.longitude,
                cuisine_type=metadata.cuisine_type,
                avg_daily_quantity=metadata.avg_daily_quantity
            )
        except Exception as e:
            print(f"[NutriLoop] Error running global forecast: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate multivariate forecast: {e}")

        if preds_df is None or preds_df.empty:
            raise HTTPException(status_code=500, detail="Multivariate forecasting logic failed completely.")
            
        news_mult = get_news_multiplier(request.city)
        predictions = []
        for _, row in preds_df.iterrows():
            d = row["date"].strftime("%Y-%m-%d") if pd.notnull(row["date"]) else "1970-01-01"
            qty = max(1, int(round(float(row["quantity"]))) if pd.notnull(row["quantity"]) else 1)
            adj_qty = max(1, round(qty * news_mult))
            predictions.append(PredictionPoint(
                date=d,
                quantity=qty,
                adjusted_quantity=adj_qty,
            ))

    return PredictResponse(
        restaurant_id=request.restaurant_id,
        item_name=request.item_name,
        predictions=predictions,
        news_multiplier=news_mult,
        model_mae=mae,
        source=source,
    )


@app.post("/cold-start", response_model=PredictResponse)
async def cold_start(request: ColdStartRequest):
    """
    Generate a forecast for a new restaurant using KMeans clustering.
    The restaurant is assigned to a cluster and gets the cluster's average forecast.
    """
    print(f"[NutriLoop] /cold-start for lat={request.latitude}, lng={request.longitude}")

    news_mult = get_news_multiplier(request.city)

    try:
        cold_preds = cold_start_forecast(
            latitude=request.latitude,
            longitude=request.longitude,
            cuisine_type=request.cuisine_type,
            avg_daily_quantity=request.avg_daily_quantity,
            item_name=request.item_name,
            days=request.days,
        )
    except Exception as e:
        print(f"[NutriLoop] Error running cold start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate cold-start forecast: {e}")

    if not cold_preds:
        raise HTTPException(status_code=500, detail="Cold-start forecast returned empty results.")

    predictions = []
    for p in cold_preds:
        adj_qty = max(1, round(p["quantity"] * news_mult))
        predictions.append(PredictionPoint(
            date=p["date"],
            quantity=p["quantity"],
            adjusted_quantity=adj_qty,
        ))

    return PredictResponse(
        restaurant_id="cold_start",
        item_name=request.item_name,
        predictions=predictions,
        news_multiplier=news_mult,
        model_mae=0.0,
        source="cold_start",
    )


# Allow running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)