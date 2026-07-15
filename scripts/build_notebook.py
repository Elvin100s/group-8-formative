"""Builds Formative2_Multimodal_Pipeline.ipynb with the full pipeline narrative."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Formative 2 — Multimodal Data Preprocessing
**User Identity & Product Recommendation System**

Pipeline: tabular merge → image auth features → audio auth features → 3 models → CLI simulation.

> Tabular data: the **real course datasets** (`customer_transactions.csv`, `customer_social_profiles.csv`, `id_mapping.csv`) are included in `data/raw/`.
> Faces and voices remain synthetic placeholders — drop real photos/recordings into `data/images/raw/` and `data/audio/raw/` and re-run; the code detects and uses them automatically."""))

C.append(code("""import sys, subprocess
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display"""))

C.append(md("""## Task 1 — Data Merge, Cleaning, EDA & Feature Engineering
Two sources use different ID schemes (`customer_id_legacy` vs `customer_id_new`), bridged by `id_mapping.csv`.
An **inner join** is used because a supervised training row needs both the label source (transactions) and the predictors (social profile). Cleaning: duplicates dropped, dtypes fixed, numeric nulls → median, categorical nulls → mode. Post-merge checks assert 0 nulls and no duplicated transactions."""))

C.append(code("""out = subprocess.run([sys.executable, "scripts/generate_data.py"], capture_output=True, text=True)
print(out.stdout)
out = subprocess.run([sys.executable, "scripts/data_merge.py"], capture_output=True, text=True)
print(out.stdout[-1500:])"""))

C.append(code("""merged = pd.read_csv("data/processed/merged_dataset.csv")
display(merged.head())
print(merged.dtypes.to_string())
merged.describe().T"""))

C.append(md("### EDA plots (distribution, outliers, correlations)"))
C.append(code("""for p in ["eda_purchase_distribution.png", "eda_outliers_boxplot.png", "eda_correlation_heatmap.png"]:
    display(IPImage("outputs/plots/" + p))"""))

C.append(md("""**Interpretation.** Purchase amounts are spread almost uniformly between ~\\$80 and \\$500 — there is no typical spend level, and the per-category boxplots show similar medians and ranges with no extreme outliers. The correlation heatmap (IDs excluded; engineered features included) confirms this flatness: apart from mechanical relationships among the engineered aggregates (`total_spend` ≈ `avg_spend` × `n_transactions`), every off-diagonal correlation is near zero. In particular the social-side predictors (`engagement_score`, `purchase_interest_score`) are uncorrelated with purchase behaviour — an early warning, consistent with the baseline check in Task 4, that this dataset carries little signal for product prediction."""))

C.append(md("""## Task 2 — Image Collection, Augmentation & Features
3 expressions per identity (neutral / smiling / surprised); 3 augmentations per image (rotation+brightness, horizontal flip, grayscale); features = 32-bin grayscale histogram + 16×16 pixel embedding → `image_features.csv`."""))

C.append(code("""out = subprocess.run([sys.executable, "scripts/image_pipeline.py"], capture_output=True, text=True)
print(out.stdout)
display(IPImage("outputs/plots/image_samples_grid.png"))"""))

C.append(code("""img_feats = pd.read_csv("data/processed/image_features.csv")
print(img_feats.shape)
img_feats[["file","member","expression","label_authorized","hist_0","emb_0"]].head(8)"""))

C.append(md("""### Augmentation examples"""))
C.append(code("""from PIL import Image
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
names = ["raw/member1_smiling.jpg", "augmented/member1_smiling_rot.jpg",
         "augmented/member1_smiling_flip.jpg", "augmented/member1_smiling_gray.jpg"]
for ax, n in zip(axes, names):
    ax.imshow(Image.open("data/images/" + n)); ax.set_title(n.split("/")[-1], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()"""))

C.append(md("""## Task 3 — Audio Collection, Visualization, Augmentation & Features
Two phrases per identity ("Yes, approve", "Confirm transaction"); waveform + spectrogram per raw sample; 3 augmentations (pitch shift +2 st, time stretch 0.85×, additive noise); features = 13 MFCC means + 13 MFCC stds + spectral roll-off + spectral centroid + RMS energy + zero-crossing rate → `audio_features.csv`."""))

C.append(code("""out = subprocess.run([sys.executable, "scripts/audio_pipeline.py"], capture_output=True, text=True)
print(out.stdout)
display(IPImage("outputs/plots/audio_member1_yes_approve.png"))
display(IPImage("outputs/plots/audio_member1_confirm_transaction.png"))"""))

C.append(md("""**Interpretation.** The waveform shows the two-word phrase as distinct energy bursts separated by silence; the spectrogram shows voiced-speech harmonics concentrated below ~4 kHz. MFCCs summarize this spectral envelope, which is what makes voiceprints separable between speakers."""))

C.append(code("""aud_feats = pd.read_csv("data/processed/audio_features.csv")
print(aud_feats.shape)
aud_feats[["file","member","phrase","label_authorized","mfcc_mean_0","spectral_rolloff","rms_energy","zcr"]].head(8)"""))

C.append(md("""## Task 4 — Model Creation & Evaluation
| Model | Algorithm | Input | Target |
|---|---|---|---|
| Facial recognition | RandomForest | image features | authorized vs not |
| Voiceprint verification | Scaler + LogisticRegression | audio features | authorized vs not |
| Product recommendation | RandomForest | merged tabular features | product_category |

Metrics: Accuracy, weighted F1, log Loss."""))

C.append(code("""out = subprocess.run([sys.executable, "scripts/train_models.py"], capture_output=True, text=True)
print(out.stdout)
import json
pd.DataFrame(json.load(open("models/metrics.json"))).T"""))

C.append(md("""### Honest baseline check — product model
With only ~97 merged rows over 5 fairly balanced classes, we compare against a majority-class dummy to test whether the tabular features carry real signal."""))

C.append(code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
m = pd.read_csv("data/processed/merged_dataset.csv")
drop = ["transaction_id","customer_id_legacy","customer_id_new","purchase_date","product_category"]
X = m.drop(columns=[c for c in drop if c in m]); y = m["product_category"]
for name, clf in [("Dummy (majority class)", DummyClassifier(strategy="most_frequent")),
                  ("RandomForest", RandomForestClassifier(300, random_state=0))]:
    s = cross_val_score(clf, X, y, cv=5)
    print(f"{name:24s} CV accuracy: {s.mean():.3f} ± {s.std():.3f}")"""))

C.append(md("""**Interpretation.** The RandomForest does not beat the majority-class baseline. Given the small merged sample (~97 rows after the inner join across the many-to-many ID mapping) and near-uniform class balance, the social-profile features in this dataset carry little to no predictive signal about the purchased product category. We report this honestly rather than overfitting to the test split; the pipeline, evaluation, and multimodal gating logic remain fully functional, and the same code would exploit signal if richer behavioral data were supplied."""))

C.append(md("""**Multimodal logic.** The product model is only ever invoked after two independent modalities agree: `P(face authorized) ≥ 0.6` **AND** `P(voice authorized) ≥ 0.6`. Either failure short-circuits to ACCESS DENIED — matching the assignment flow diagram."""))

C.append(md("""## Task 6 — System Simulation (CLI)
Full authorized transaction, then an unauthorized attempt (impostor face + authorized voice)."""))

C.append(code("""cust = pd.read_csv("data/processed/merged_dataset.csv")["customer_id_new"].iloc[0]
out = subprocess.run([sys.executable, "app.py",
    "--image", "data/images/raw/member1_smiling.jpg",
    "--audio", "data/audio/raw/member1_yes_approve.wav",
    "--customer", str(cust)], capture_output=True, text=True)
print(out.stdout)"""))

C.append(code("""out = subprocess.run([sys.executable, "app.py",
    "--image", "data/images/raw/impostor2_neutral.jpg",
    "--audio", "data/audio/raw/member1_yes_approve.wav"], capture_output=True, text=True)
print(out.stdout)  # expected: ACCESS DENIED at Step 1"""))

C.append(md("""## Conclusion
All rubric items are covered: cleaned + validated merge with justified join logic, ≥3 labeled EDA plots with interpretation, 3 expressions per identity with ≥2 augmentations and saved `image_features.csv`, 2 phrases with waveform/spectrogram plots, ≥2 audio augmentations and saved `audio_features.csv`, three trained models each evaluated on accuracy/F1/loss, and a working CLI simulation including an unauthorized attempt.

**Next step for the team:** replace synthetic faces/voices/CSVs with real data and re-run top-to-bottom."""))

nb["cells"] = C
nbf.write(nb, "Formative2_Multimodal_Pipeline.ipynb")
print("notebook written")
