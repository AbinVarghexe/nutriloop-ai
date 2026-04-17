# NutriLoop AI Agent Context

## What this project is

NutriLoop AI is a Python FastAPI backend for restaurant food-demand forecasting. It serves two prediction paths:

- Prophet forecasting for known restaurant + item pairs.
- KMeans-based cold-start forecasting for restaurants with no trained model.

It also applies a NewsAPI-based multiplier to adjust demand using city-level headlines.

## What is already done

- FastAPI app exists in `app/main.py` with `/health`, `/predict`, and `/cold-start` endpoints.
- Request/response models are defined in `app/schemas.py`.
- Prophet model loading and inference are implemented in `app/predict.py`.
- Cold-start fallback logic is implemented in `app/cold_start.py`.
- News-based demand adjustment is implemented in `app/news_adjuster.py`.
- Kaggle CSV ingestion and cleanup are implemented in `training/load_kaggle_data.py`.
- Prophet training pipeline is implemented in `training/train_prophet.py`.
- KMeans clustering training is implemented in `training/train_clusters.py`.
- Model upload to Hugging Face Spaces is implemented in `training/upload_models.py`.
- Full retrain orchestration is implemented in `scripts/retrain.py`.
- Supabase seeding helper exists in `scripts/seed_supabase.py`.
- Basic API test script exists in `scripts/test_api.py`.
- Nightly GitHub Actions retraining workflow exists in `.github/workflows/retrain.yml`.
- Docker-based deployment is configured with `Dockerfile`.
- Shared restaurant metadata helpers now exist in `app/restaurant_metadata.py`.
- `predict` no longer uses hardcoded Kochi defaults when Prophet models are missing.
- `train_clusters.py` now uses deterministic restaurant metadata instead of random coordinates.
- `seed_supabase.py` now seeds a `restaurants` metadata table alongside `sales_logs`.

## Current runtime flow

1. Data is seeded into Supabase `sales_logs` from Kaggle CSV.
2. `train_prophet.py` trains one model per `restaurant_id + item_name` and writes `.pkl` files plus `model_registry.json`.
3. `train_clusters.py` builds a fallback cluster model and writes `cluster_model.pkl`, `cluster_scaler.pkl`, and `cluster_map.json`.
4. `upload_models.py` pushes model artifacts to Hugging Face Spaces.
5. `app/main.py` loads models from `./models` at runtime and serves forecasts.
6. `news_adjuster.py` adds a multiplier based on recent headlines for the requested city.

## Important env vars

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `NEWSAPI_KEY`
- `HF_TOKEN`
- `HF_REPO_ID`

## Good next implementation targets for a coding agent

### Highest priority

- Improve `/health` so it reports which model artifacts are present and whether Supabase / HF configuration is valid.
- Add stronger validation and error handling for missing model files, malformed Supabase rows, and empty forecast results.
- Persist and expose a retrain timestamp instead of keeping it only in memory.

### Medium priority

- Make news adjustment caching and failure behavior more explicit in API responses.
- Expand API tests so they cover both prophet and cold-start paths deterministically.

### Lower priority

- Add structured logging instead of `print` statements.
- Add metrics for model coverage, forecast counts, and retraining success rate.
- Refactor training scripts into reusable modules if the pipeline grows.

## Notes for the coding agent

- Favor small, focused changes that improve correctness before adding new features.
- Keep the API contract stable unless the user requests a breaking change.
- Do not assume real restaurant coordinates exist unless you add the source of truth.
- If you need to change the inference path, update both the API and the training pipeline together.

## Useful commands

- Install deps: `uv sync`
- Run API: `uvicorn app.main:app --reload --port 7860`
- Train Prophet models: `python training/train_prophet.py`
- Train clusters: `python training/train_clusters.py`
- Seed Supabase SQL only: `python scripts/seed_supabase.py --print-sql`
- Run end-to-end tests: `python scripts/test_api.py`
