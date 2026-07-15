"""
app.py  —  Task 6: System Demonstration (CLI)
---------------------------------------------
Simulates the User Identity & Product Recommendation flow:

    face image --> [Face Recognition] --ok--> voice clip --> [Voiceprint
    Verification] --ok--> [Product Recommendation for the customer]
    (any failed step -> ACCESS DENIED)

Usage:
    # full authorized transaction
    python app.py --image data/images/raw/member1_neutral.jpg \
                  --audio data/audio/raw/member1_yes_approve.wav \
                  --customer A176

    # unauthorized attempt (impostor face)
    python app.py --image data/images/raw/impostor1_neutral.jpg \
                  --audio data/audio/raw/member1_yes_approve.wav
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from image_pipeline import extract_features as image_features            # noqa: E402
from audio_pipeline import extract_features as audio_features            # noqa: E402

THRESHOLD = 0.60  # min probability to authorize


def banner(msg, ok=True):
    print(("  [OK]   " if ok else "  [DENY] ") + msg)


def load(name):
    bundle = joblib.load(ROOT / "models" / name)
    return bundle["model"], bundle["columns"]


def check(model, cols, feats, step):
    X = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0.0)
    p = float(model.predict_proba(X)[0, 1])
    ok = p >= THRESHOLD
    banner(f"{step}: P(authorized) = {p:.2f} (threshold {THRESHOLD})", ok)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Multimodal auth + product recommendation")
    ap.add_argument("--image", required=True, help="path to face image")
    ap.add_argument("--audio", required=True, help="path to voice clip (wav)")
    ap.add_argument("--customer", default=None, help="customer_id_new for recommendation")
    args = ap.parse_args()

    print("=" * 60)
    print(" USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)

    # ---- Step 1: face recognition ----
    print("\nStep 1/3  Facial recognition")
    face_model, face_cols = load("face_model.joblib")
    f = image_features(Path(args.image))
    if not check(face_model, face_cols, f, "Face"):
        print("\n*** ACCESS DENIED — face not recognized. ***")
        sys.exit(1)

    # ---- Step 2: voiceprint verification ----
    print("\nStep 2/3  Voiceprint verification")
    voice_model, voice_cols = load("voice_model.joblib")
    a = audio_features(Path(args.audio))
    if not check(voice_model, voice_cols, a, "Voice"):
        print("\n*** ACCESS DENIED — voice not approved. ***")
        sys.exit(1)

    # ---- Step 3: product recommendation ----
    print("\nStep 3/3  Product recommendation")
    product_model, prod_cols = load("product_model.joblib")
    merged = pd.read_csv(ROOT / "data/processed/merged_dataset.csv")
    cust = args.customer or merged["customer_id_new"].iloc[0]
    rows = merged[merged["customer_id_new"] == cust]
    if rows.empty:
        print(f"  customer {cust} not found in merged dataset"); sys.exit(1)
    X = rows.reindex(columns=prod_cols, fill_value=0).iloc[[0]]
    pred = product_model.predict(X)[0]
    proba = product_model.predict_proba(X)[0]
    top3 = sorted(zip(product_model.classes_, proba), key=lambda t: -t[1])[:3]

    print(f"\n  Authenticated user verified. Recommendation for customer {cust}:")
    print(f"  >>> RECOMMENDED PRODUCT: {pred} <<<")
    print("  Top-3 probabilities: " + ", ".join(f"{c} ({p:.2f})" for c, p in top3))
    print("\nTransaction complete.")


if __name__ == "__main__":
    main()
