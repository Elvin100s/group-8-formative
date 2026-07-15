"""
train_models.py  —  Task 4: Model Creation & Evaluation
-------------------------------------------------------
Trains and evaluates three models, saving each to models/ with joblib:

1. Facial Recognition   (RandomForest)      : image features -> authorized (member1) vs not
2. Voiceprint Verification (LogisticRegression) : audio features -> authorized vs not
3. Product Recommendation (RandomForest)    : merged tabular features -> product_category

Metrics reported per model: Accuracy, weighted F1-score, log Loss.
Multimodal logic: at inference (app.py) a prediction is only served when
face_model AND voice_model both authorize, each above a probability threshold.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data/processed", ROOT / "models"
MODELS.mkdir(exist_ok=True)


def evaluate(name, model, Xte, yte, labels=None):
    proba = model.predict_proba(Xte)
    pred = model.predict(Xte)
    m = {"accuracy": round(accuracy_score(yte, pred), 4),
         "f1_weighted": round(f1_score(yte, pred, average="weighted"), 4),
         "log_loss": round(log_loss(yte, proba, labels=labels), 4)}
    print(f"[{name}] {m}")
    return m


def main():
    metrics = {}

    # ---------- 1. Facial recognition ----------
    img = pd.read_csv(PROC / "image_features.csv")
    Xi = img.drop(columns=["file", "member", "expression", "label_authorized"])
    yi = img["label_authorized"]
    Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=42)
    face = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    metrics["face_recognition"] = evaluate("face", face, Xte, yte, labels=[0, 1])
    joblib.dump({"model": face, "columns": list(Xi.columns)}, MODELS / "face_model.joblib")

    # ---------- 2. Voiceprint verification ----------
    aud = pd.read_csv(PROC / "audio_features.csv")
    Xa = aud.drop(columns=["file", "member", "phrase", "label_authorized"])
    ya = aud["label_authorized"]
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=42)
    voice = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=42)).fit(Xtr, ytr)
    metrics["voice_verification"] = evaluate("voice", voice, Xte, yte, labels=[0, 1])
    joblib.dump({"model": voice, "columns": list(Xa.columns)}, MODELS / "voice_model.joblib")

    # ---------- 3. Product recommendation ----------
    merged = pd.read_csv(PROC / "merged_dataset.csv")
    drop = ["transaction_id", "customer_id_legacy", "customer_id_new",
            "purchase_date", "product_category"]
    Xp = merged.drop(columns=[c for c in drop if c in merged.columns])
    yp = merged["product_category"]
    Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.25, stratify=yp, random_state=42)
    product = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    metrics["product_recommendation"] = evaluate("product", product, Xte, yte,
                                                 labels=sorted(yp.unique()))
    joblib.dump({"model": product, "columns": list(Xp.columns)}, MODELS / "product_model.joblib")

    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll 3 models + metrics.json saved -> {MODELS}")


if __name__ == "__main__":
    main()
