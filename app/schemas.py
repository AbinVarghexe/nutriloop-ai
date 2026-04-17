"""
Pydantic schemas for NutriLoop AI request/response models.
"""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    restaurant_id: str = Field(..., description="Unique restaurant identifier")
    item_name: str = Field(..., description="Menu item name")
    city: str = Field(..., description="City name for news adjustment")
    days: int = Field(default=7, ge=1, le=30, description="Forecast horizon in days")


class ColdStartRequest(BaseModel):
    """Request body for /cold-start endpoint."""
    latitude: float = Field(..., description="Restaurant latitude")
    longitude: float = Field(..., description="Restaurant longitude")
    cuisine_type: str = Field(..., description="Type of cuisine")
    avg_daily_quantity: float = Field(..., description="Average daily order quantity")
    item_name: str = Field(..., description="Menu item to forecast")
    city: str = Field(default="Unknown", description="City name for news adjustment")
    days: int = Field(default=7, ge=1, le=30, description="Forecast horizon in days")


class PredictionPoint(BaseModel):
    """Single day forecast point."""
    date: str  # YYYY-MM-DD
    quantity: int
    adjusted_quantity: int


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    restaurant_id: str
    item_name: str
    predictions: list[PredictionPoint]
    news_multiplier: float
    model_mae: float
    source: str = Field(..., description="'prophet' or 'cold_start'")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    global_model_present: bool
    cluster_model_present: bool
    config_valid: bool
    last_retrain: Optional[str]
    version: str