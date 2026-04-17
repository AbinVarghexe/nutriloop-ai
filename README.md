---
title: NutriLoop AI
emoji: 🍱
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# NutriLoop AI

Food demand forecasting backend using Facebook Prophet, KMeans clustering for cold-start restaurants, and NewsAPI-based demand adjustment.

## Features

- **Prophet Forecasting**: Trained per restaurant+item combination with Indian public holidays
- **Cold-Start Clustering**: KMeans-based fallback for new restaurants without historical data
- **Restaurant Metadata**: Optional `restaurants` table stores location and cuisine data for deterministic fallback behavior
- **News Adjustment**: Real-time demand multiplier from news headlines (festival, weather, events)
- **Nightly Retraining**: GitHub Actions cron job pulls latest Supabase data and retrains models
- **Hugging Face Spaces**: Containerized FastAPI app deploys to `*.hf.space`

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
# Use Python 3.13 for this project
# Create or refresh the venv if needed, then install dependencies
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
# - SUPABASE_URL / SUPABASE_KEY from Supabase project settings
# - NEWSAPI_KEY from https://newsapi.org (free tier)
# - HF_TOKEN / HF_REPO_ID from Hugging Face
```

### 3. Seed Supabase with Kaggle Data

Download the Kaggle dataset from: https://www.kaggle.com/datasets/mer-sun/restaurant-sales

```bash
# Print the SQL schema to run in Supabase dashboard
python scripts/seed_supabase.py --print-sql

# Seed data from CSV
python scripts/seed_supabase.py /path/to/kaggle/sales.csv
```

### 4. Train Models

```bash
# Train Prophet models from Supabase data
python training/train_prophet.py

# Train KMeans clustering model
python training/train_clusters.py

# Upload models to Hugging Face Hub
python training/upload_models.py
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 7860
```

API will be available at: http://localhost:7860

### 6. Test the API

```bash
python scripts/test_api.py
```

## API Endpoints

### GET /health

Health check with model count and version.

```bash
curl http://localhost:7860/health
```

### POST /predict

Forecast demand for an existing restaurant with trained Prophet model.

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": "rest_001",
    "item_name": "biriyani",
    "city": "Kochi",
    "days": 7
  }'
```

### POST /cold-start

Forecast demand for a new restaurant using cluster-based averaging.

```bash
curl -X POST http://localhost:7860/cold-start \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 9.9312,
    "longitude": 76.2673,
    "cuisine_type": "indian",
    "avg_daily_quantity": 25.0,
    "item_name": "puttu",
    "days": 7
  }'
```

## Calling from Next.js

```typescript
// In your Next.js kiosk dashboard page:
const HF_API = process.env.NEXT_PUBLIC_NUTRILOOP_AI_URL
// e.g. "https://your-username-nutriloop-ai.hf.space"

export async function getForecast(restaurantId: string, itemName: string, city: string) {
  const res = await fetch(`${HF_API}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      restaurant_id: restaurantId,
      item_name: itemName,
      city: city,
      days: 7
    })
  })
  return res.json()
  // Returns: { predictions: [{date, quantity, adjusted_quantity}], news_multiplier, source, ... }
}
```

## Deployment to Hugging Face Spaces

### 1. Create a new Space

Go to https://huggingface.co/new-space and create a Docker-based Space named `nutriloop-ai`.

### 2. Push to HF

```bash
git init
git add .
git commit -m "NutriLoop AI initial commit"
git remote add origin https://huggingface.co/spaces/<your-username>/nutriloop-ai
git push origin main
```

### 3. Set Secrets in HF Space Settings

In your HF Space settings, add these secrets:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `NEWSAPI_KEY`
- `HF_TOKEN`
- `HF_REPO_ID`

### 4. Access your deployed API

Your API will be live at: `https://<your-username>-nutriloop-ai.hf.space`

## Supabase Schema

Run this SQL in your Supabase SQL editor:

```sql
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Table: sales_logs
CREATE TABLE sales_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    restaurant_id text NOT NULL,
    item_name text NOT NULL,
    quantity integer NOT NULL,
    sale_date date NOT NULL,
    location geography(Point, 4326),
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_sales_logs_restaurant_id ON sales_logs(restaurant_id);
CREATE INDEX idx_sales_logs_sale_date ON sales_logs(sale_date);
CREATE INDEX idx_sales_logs_restaurant_item ON sales_logs(restaurant_id, item_name);

-- Table: restaurants
CREATE TABLE restaurants (
  restaurant_id text PRIMARY KEY,
  restaurant_name text,
  latitude double precision,
  longitude double precision,
  cuisine_type text,
  avg_daily_quantity double precision,
  created_at timestamptz DEFAULT now()
);

-- Table: retrain_log
CREATE TABLE retrain_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_at timestamptz DEFAULT now(),
    model_version text NOT NULL,
    rows_used integer,
    mae_score float,
    status text CHECK (status IN ('success', 'failed')),
    error_msg text
);
```

## Nightly Retraining (GitHub Actions)

The `.github/workflows/retrain.yml` workflow runs automatically at 1:00 AM IST daily.
It executes `scripts/retrain.py` which:
1. Pulls latest data from Supabase
2. Retrains all Prophet models
3. Retrains KMeans clusters
4. Uploads updated models to Hugging Face Hub

You can also trigger manually from the GitHub Actions tab.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/public key |
| `NEWSAPI_KEY` | NewsAPI key (free tier at newsapi.org) |
| `HF_TOKEN` | Hugging Face access token |
| `HF_REPO_ID` | HF Space repo ID (e.g. `username/nutriloop-ai`) |

## Project Structure

```
nutriloop-ai/
├── app/
│   ├── main.py          # FastAPI app entry point
│   ├── predict.py       # Prophet model loading and inference
│   ├── cold_start.py    # KMeans cold-start forecasting
│   ├── news_adjuster.py # NewsAPI-based demand multiplier
│   └── schemas.py       # Pydantic request/response models
├── training/
│   ├── train_prophet.py # Full Prophet training pipeline
│   ├── train_clusters.py# KMeans clustering training
│   ├── load_kaggle_data.py # Kaggle CSV loader and cleaner
│   └── upload_models.py # Hugging Face model upload
├── scripts/
│   ├── retrain.py       # Master retraining orchestrator
│   ├── seed_supabase.py # First-run data seeding
│   └── test_api.py      # End-to-end API tests
├── models/              # Trained .pkl files (gitignored)
├── data/                # Raw Kaggle CSV (gitignored)
├── .github/workflows/
│   └── retrain.yml      # GitHub Actions nightly cron
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech Stack

- **Runtime**: Python 3.11
- **Web Framework**: FastAPI + Uvicorn
- **Forecasting**: Facebook Prophet (with Indian holidays)
- **Clustering**: scikit-learn KMeans
- **Database**: Supabase (PostgreSQL + PostGIS)
- **Model Registry**: joblib + JSON
- **Model Storage**: Hugging Face Spaces
- **CI/CD**: GitHub Actions
- **Hosting**: Hugging Face Spaces (Docker)## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes and add tests if applicable
4. Submit a pull request with a clear description

Please follow the existing code style and add documentation where needed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact & Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainer:

- GitHub Issues: https://github.com/your-username/nutriloop-ai/issues
- Email: your.email@example.com

## Acknowledgements

- [Facebook Prophet](https://facebook.github.io/prophet/)
- [scikit-learn](https://scikit-learn.org/)
- [Supabase](https://supabase.com/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [NewsAPI](https://newsapi.org/)
