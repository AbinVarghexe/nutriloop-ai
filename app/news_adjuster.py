"""
News-based demand multiplier using NewsAPI for NutriLoop AI.
"""
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx

# In-memory cache: key = city, value = (multiplier, timestamp)
_cache: dict[str, tuple[float, float]] = {}
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


def _get_cached(city: str) -> Optional[float]:
    """Return cached multiplier if still fresh, else None."""
    if city not in _cache:
        return None
    multiplier, ts = _cache[city]
    if time.time() - ts < CACHE_TTL_SECONDS:
        return multiplier
    return None


def _cache_result(city: str, multiplier: float) -> None:
    """Store result in memory cache."""
    _cache[city] = (multiplier, time.time())


def _score_headline(headline: str) -> float:
    """
    Score a headline and return delta multiplier.
    Rules:
        +0.20: festival, onam, christmas, vishu, new year, fair, carnival
        +0.15: holiday, weekend, event
        -0.30: flood, strike, shutdown, curfew, lockdown
        -0.15: rain, storm, bandh
    """
    text = headline.lower()
    score = 0.0

    positive = ["festival", "onam", "christmas", "vishu", "new year", "fair", "carnival"]
    neutral_positive = ["holiday", "weekend", "event"]
    negative = ["flood", "strike", "shutdown", "curfew", "lockdown"]
    neutral_negative = ["rain", "storm", "bandh"]

    for kw in positive:
        if kw in text:
            score += 0.20
    for kw in neutral_positive:
        if kw in text:
            score += 0.15
    for kw in negative:
        if kw in text:
            score -= 0.30
    for kw in neutral_negative:
        if kw in text:
            score -= 0.15

    return score


def get_news_multiplier(city: str) -> float:
    """
    Fetch recent news for a city and compute a demand multiplier.
    Results are cached for 6 hours.

    Args:
        city: City name (e.g., 'Kochi')

    Returns:
        Multiplier between 0.5 and 2.0 (default 1.0 if API fails)
    """
    # Check cache first
    cached = _get_cached(city)
    if cached is not None:
        print(f"[NutriLoop] News multiplier for '{city}' from cache: {cached}")
        return cached

    print(f"[NutriLoop] Fetching news for city: {city}")
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        print("[NutriLoop] NEWSAPI_KEY not set, returning neutral multiplier 1.0")
        return 1.0

    query = f'"{city}" food festival event strike flood'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": 10,
        "sortBy": "relevancy",
        "apiKey": api_key,
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)

        if response.status_code != 200:
            print(f"[NutriLoop] NewsAPI returned {response.status_code}, returning 1.0")
            return 1.0

        data = response.json()
        articles = data.get("articles", [])
        total_score = 0.0

        for article in articles:
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            content = f"{title} {description}"
            total_score += _score_headline(content)

        # Convert score to multiplier: score of 0 = 1.0, each +/-0.15 step = +/-0.1
        multiplier = 1.0 + (total_score * 0.1)
        multiplier = max(0.5, min(2.0, multiplier))

        _cache_result(city, multiplier)
        print(f"[NutriLoop] News multiplier for '{city}': {multiplier} (score={total_score:.2f})")
        return multiplier

    except Exception as e:
        print(f"[NutriLoop] NewsAPI call failed: {e}, returning 1.0")
        return 1.0