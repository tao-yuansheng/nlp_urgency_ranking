"""Download the fine-tuned model from HuggingFace Hub into model_output/."""

import os
import subprocess
import sys

REPO_ID  = "yuansheng-tao/emotion_urgency_classifier"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import snapshot_download

print(f"Downloading '{REPO_ID}' into '{OUT_DIR}' ...")
os.makedirs(OUT_DIR, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    local_dir=OUT_DIR,
    ignore_patterns=["*.gitattributes", ".gitattributes"],
)

print("\nDownload complete.")
print("Next steps:")
print("  Run adversarial test : python model_training/adversarial_test.py")
print("  Run training         : python model_training/train_deberta.py")
