"""
End-to-end test script for NutriLoop AI API.
Tests /health, /predict, /cold-start, and news adjuster.
"""
import httpx
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


BASE_URL = "http://localhost:7860"


def test_health():
    """Test GET /health endpoint."""
    print("\n[TEST] GET /health")
    response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
    data = response.json()
    print(f"  Status: {response.status_code}")
    print(f"  Response: {data}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["status"] == "ok", f"Expected status='ok', got {data['status']}"
    print("  PASS")


def test_predict_global():
    """Test POST /predict with known restaurant_id and item."""
    print("\n[TEST] POST /predict")
    # Use a restaurant that might have a model, or test cold-start fallback
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "restaurant_id": "test_restaurant_001",
            "item_name": "biriyani",
            "city": "Kochi",
            "days": 7,
        },
        timeout=10.0,
    )
    data = response.json()
    print(f"  Status: {response.status_code}")
    print(f"  Response: {data}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "predictions" in data, "Missing 'predictions' in response"
    assert len(data["predictions"]) == 7, f"Expected 7 predictions, got {len(data['predictions'])}"
    assert data["source"] in ["global_model", "cold_start"], f"Invalid source: {data['source']}"
    print("  PASS")


def test_cold_start():
    """Test POST /cold-start endpoint with Kochi coordinates."""
    print("\n[TEST] POST /cold-start")
    response = httpx.post(
        f"{BASE_URL}/cold-start",
        json={
            "latitude": 9.9312,
            "longitude": 76.2673,
            "cuisine_type": "indian",
            "avg_daily_quantity": 25.0,
            "item_name": "puttu",
            "days": 7,
        },
        timeout=10.0,
    )
    data = response.json()
    print(f"  Status: {response.status_code}")
    print(f"  Response: {data}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["source"] == "cold_start", f"Expected source='cold_start', got {data['source']}"
    assert len(data["predictions"]) == 7, f"Expected 7 predictions, got {len(data['predictions'])}"
    assert data["news_multiplier"] > 0, "Invalid news_multiplier"
    print("  PASS")


def test_news_adjuster():
    """Test news adjuster directly."""
    print("\n[TEST] News adjuster (Kochi)")
    from app.news_adjuster import get_news_multiplier
    multiplier = get_news_multiplier("Kochi")
    print(f"  Multiplier: {multiplier}")
    assert 0.5 <= multiplier <= 2.0, f"Multiplier out of range: {multiplier}"
    print("  PASS")


def main():
    """Run all tests."""
    print("=" * 50)
    print("NutriLoop AI - End-to-End API Tests")
    print("=" * 50)

    # Check if server is running
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5.0)
    except httpx.ConnectError:
        print(f"\nERROR: Cannot connect to {BASE_URL}")
        print("Make sure the server is running:")
        print("  uv run uvicorn app.main:app --reload --port 7860")
        sys.exit(1)

    passed = 0
    failed = 0

    tests = [
        ("Health", test_health),
        ("Predict", test_predict_global),
        ("Cold-Start", test_cold_start),
        ("News Adjuster", test_news_adjuster),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()