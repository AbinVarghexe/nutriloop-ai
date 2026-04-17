"""
Upload trained model .pkl files to Hugging Face Hub.
Uploads all files in models/ directory to the specified HF_REPO_ID.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

MODELS_DIR = Path(__file__).parent.parent / "models"


def upload_models():
    """
    Upload all .pkl and .json files from models/ to Hugging Face Hub.
    Requires HF_TOKEN and HF_REPO_ID environment variables.
    """
    print("[NutriLoop] Starting model upload to Hugging Face Hub")

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")

    if not token:
        print("[NutriLoop] ERROR: HF_TOKEN not set in environment")
        return
    if not repo_id:
        print("[NutriLoop] ERROR: HF_REPO_ID not set in environment")
        return

    # Check models directory
    if not MODELS_DIR.exists():
        print(f"[NutriLoop] ERROR: Models directory {MODELS_DIR} does not exist")
        return

    model_files = list(MODELS_DIR.glob("*"))
    if not model_files:
        print("[NutriLoop] WARNING: No model files found in models/ directory")
        return

    print(f"[NutriLoop] Found {len(model_files)} files to upload")
    for f in model_files:
        print(f"  - {f.name}")

    api = HfApi()

    try:
        print(f"[NutriLoop] Uploading to https://huggingface.co/spaces/{repo_id}")
        api.upload_folder(
            folder_path=str(MODELS_DIR),
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
        print(f"[NutriLoop] Successfully uploaded all models to {repo_id}")
    except Exception as e:
        print(f"[NutriLoop] Upload failed: {e}")
        raise


if __name__ == "__main__":
    upload_models()