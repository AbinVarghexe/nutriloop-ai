"""
Restaurant metadata helpers for NutriLoop AI.

This module provides deterministic fallback metadata when a restaurants table
is not available, and also supports loading real metadata from Supabase when
the optional table exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


KNOWN_CUISINES = [
    "indian", "chinese", "italian", "mexican", "american",
    "thai", "japanese", "korean", "mediterranean", "fast_food", "cafe", "bakery"
]


@dataclass(frozen=True)
class RestaurantMetadata:
    restaurant_id: str
    latitude: float
    longitude: float
    cuisine_type: str
    avg_daily_quantity: float


def normalize_cuisine(cuisine: str | None) -> str:
    """Normalize cuisine text to a known category."""
    if not cuisine:
        return "indian"

    c = cuisine.lower().strip()
    for known in KNOWN_CUISINES:
        if known in c or c in known:
            return known
    return "indian"


def cuisine_to_label(cuisine: str | None) -> int:
    """Convert cuisine text to a stable integer label."""
    normalized = normalize_cuisine(cuisine)
    try:
        return KNOWN_CUISINES.index(normalized)
    except ValueError:
        return 0


def _hash_bytes(restaurant_id: str) -> bytes:
    return hashlib.sha256(restaurant_id.encode("utf-8")).digest()


def deterministic_restaurant_metadata(
    restaurant_id: str,
    avg_daily_quantity: float | None = None,
) -> RestaurantMetadata:
    """Create stable fallback metadata from the restaurant identifier."""
    digest = _hash_bytes(restaurant_id)

    # Keep values in a plausible India range while remaining deterministic.
    lat_seed = int.from_bytes(digest[0:4], "big") / 2**32
    lng_seed = int.from_bytes(digest[4:8], "big") / 2**32
    cuisine_index = digest[8] % len(KNOWN_CUISINES)
    qty_seed = digest[9] % 40

    latitude = round(8.0 + (lat_seed * 15.0), 6)
    longitude = round(68.0 + (lng_seed * 25.0), 6)
    cuisine_type = KNOWN_CUISINES[cuisine_index]
    quantity = float(avg_daily_quantity) if avg_daily_quantity is not None else float(10 + qty_seed)

    return RestaurantMetadata(
        restaurant_id=str(restaurant_id),
        latitude=latitude,
        longitude=longitude,
        cuisine_type=cuisine_type,
        avg_daily_quantity=quantity,
    )


def parse_location_point(value: object) -> tuple[Optional[float], Optional[float]]:
    """Parse a geography POINT value when Supabase returns location text."""
    if value is None:
        return None, None

    text = str(value).strip()
    match = re.search(r"POINT\s*\(\s*([\-\d\.]+)\s+([\-\d\.]+)\s*\)", text, re.IGNORECASE)
    if not match:
        return None, None

    longitude = float(match.group(1))
    latitude = float(match.group(2))
    return latitude, longitude


def load_restaurant_metadata(client, restaurant_id: str) -> RestaurantMetadata:
    """Load restaurant metadata from Supabase, with deterministic fallback."""
    fallback = deterministic_restaurant_metadata(restaurant_id)

    if client is None:
        return fallback

    try:
        response = client.table("restaurants").select("*").eq("restaurant_id", restaurant_id).limit(1).execute()
    except Exception:
        return fallback

    rows = getattr(response, "data", None) or []
    if not rows:
        return fallback

    row = rows[0]
    latitude = row.get("latitude")
    longitude = row.get("longitude")

    if latitude is None or longitude is None:
        parsed_latitude, parsed_longitude = parse_location_point(row.get("location"))
        latitude = parsed_latitude if latitude is None else latitude
        longitude = parsed_longitude if longitude is None else longitude

    cuisine_type = row.get("cuisine_type") or fallback.cuisine_type
    avg_daily_quantity = row.get("avg_daily_quantity")

    try:
        latitude = float(latitude) if latitude is not None else fallback.latitude
        longitude = float(longitude) if longitude is not None else fallback.longitude
        avg_daily_quantity = float(avg_daily_quantity) if avg_daily_quantity is not None else fallback.avg_daily_quantity
    except (TypeError, ValueError):
        return fallback

    return RestaurantMetadata(
        restaurant_id=str(row.get("restaurant_id") or restaurant_id),
        latitude=latitude,
        longitude=longitude,
        cuisine_type=normalize_cuisine(cuisine_type),
        avg_daily_quantity=avg_daily_quantity,
    )


def create_supabase_client_from_env():
    """Create a Supabase client if credentials are available."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        return None

    from supabase import create_client

    return create_client(supabase_url, supabase_key)
