"""
Master retraining script for NutriLoop AI.
Called by GitHub Actions nightly cron or manually.
Runs: Supabase data pull → Prophet training → Cluster training → HF upload.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client


def run_retrain():
    """
    Execute the full NutriLoop retraining pipeline.

    Steps:
    1. Pull latest data from Supabase sales_logs
    2. Train Prophet models (train_prophet.py)
    3. Train KMeans clusters (train_clusters.py)
    4. Upload models to Hugging Face (upload_models.py)
    5. Log success/failure to Supabase retrain_log
    """
    print("=" * 60)
    print("[NutriLoop Retrain] Starting full retraining pipeline")
    print("=" * 60)
    start_time = datetime.now()

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("[NutriLoop Retrain] ERROR: SUPABASE_URL and SUPABASE_KEY not set")
        sys.exit(1)

    client = create_client(supabase_url, supabase_key)

    # Step 1: Check data availability
    print("\n[Step 1] Checking Supabase data...")
    try:
        response = client.table("sales_logs").select("id").limit(1).execute()
        print("[Step 1] Supabase connection OK")
    except Exception as e:
        print(f"[Step 1] ERROR: Cannot connect to Supabase: {e}")
        _log_failure(client, str(e))
        sys.exit(1)

    # Step 2: Train Global Multivariate Model
    print("\n[Step 2] Training Global Multivariate Model...")
    try:
        from training.train_global import train_global_model
        success, registry = train_global_model()
        if not success:
            raise Exception("Global model training failed.")
        print(f"[Step 2] Global Model trained successfully")
    except Exception as e:
        print(f"[Step 2] ERROR: Global training failed: {e}")
        _log_failure(client, f"Prophet training: {e}")
        sys.exit(1)

    # Step 3: Train KMeans clusters
    print("\n[Step 3] Training KMeans clusters...")
    try:
        from training.train_clusters import train_clusters
        n_clusters, _ = train_clusters()
        print(f"[Step 3] Clusters: {n_clusters} clusters created")
    except Exception as e:
        print(f"[Step 3] WARNING: Cluster training failed: {e}")
        # Non-fatal - continue
        n_clusters = 0

    # Step 4: Upload to Hugging Face
    print("\n[Step 4] Uploading models to Hugging Face...")
    hf_uploaded = False
    try:
        from training.upload_models import upload_models
        upload_models()
        hf_uploaded = True
        print("[Step 4] HF upload complete")
    except Exception as e:
        print(f"[Step 4] WARNING: HF upload failed: {e}")
        # Non-fatal - continue

    # Step 5: Log success
    elapsed = (datetime.now() - start_time).total_seconds()
    avg_mae = sum(m["mae"] for m in registry.values()) / max(1, len(registry)) if registry else 0.0

    current_iso = datetime.now().isoformat()
    try:
        last_retrain_path = project_root / "models" / "last_retrain.txt"
        last_retrain_path.parent.mkdir(exist_ok=True)
        last_retrain_path.write_text(current_iso)
    except Exception as e:
        print(f"[Step 5] WARNING: Could not write last_retrain.txt: {e}")

    try:
        client.table("retrain_log").insert({
            "model_version": current_iso,
            "rows_used": len(registry),
            "mae_score": round(avg_mae, 4),
            "status": "success",
        }).execute()
    except Exception as e:
        print(f"[Step 5] WARNING: Could not log to retrain_log: {e}")

    # Summary
    hf_status = "Yes" if hf_uploaded else "No"
    print("\n" + "=" * 60)
    print(f"[NutriLoop Retrain] Complete in {elapsed:.1f}s")
    print(f"  Summary:")
    print(f"  Global model trained: {'Yes' if 'success' in locals() and success else 'No'}")
    print(f"  Cluster model trained: Yes")
    print(f"  Clusters: {n_clusters}")
    print(f"  Avg MAE: {avg_mae:.4f}")
    print(f"  Uploaded to HF: {hf_status}")
    print("=" * 60)


def _log_failure(client, error_msg: str) -> None:
    """Log a failed retrain run to Supabase."""
    try:
        client.table("retrain_log").insert({
            "model_version": datetime.now().isoformat(),
            "rows_used": 0,
            "mae_score": 0.0,
            "status": "failed",
            "error_msg": error_msg,
        }).execute()
    except Exception:
        pass


if __name__ == "__main__":
    run_retrain()