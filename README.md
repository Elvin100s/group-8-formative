# Formative 2 — Multimodal Data Preprocessing
User Identity & Product Recommendation System: face recognition → voiceprint verification → product recommendation, with ACCESS DENIED pathways on any failed step.

## ⚠️ Placeholder data — replace before submission
This repo currently runs on **synthetic stand-ins** so the full pipeline is testable end-to-end. Swap in real data (the code auto-detects real files and skips generation):

| Replace with | Location | Naming |
|---|---|---|
|  ~~Course CSVs~~ ✅ **real course datasets already included** (sourced from the shared assignment data) | `data/raw/` | keep the same filenames & columns |
| Your face photos (neutral, smiling, surprised) | `data/images/raw/` | `member1_neutral.jpg`, `member1_smiling.jpg`, `member1_surprised.jpg` (add `member2_…` etc. for teammates) |
| Your voice recordings | `data/audio/raw/` | `member1_yes_approve.wav`, `member1_confirm_transaction.wav` |

Keep at least one non-member identity (impostor) in images/audio so the auth models have negative examples.

## Repo structure
```
├── app.py                          # CLI simulation (Task 6)
├── Formative2_Multimodal_Pipeline.ipynb  # executed notebook, full narrative
├── data/
│   ├── raw/                        # source CSVs
│   ├── images/{raw,augmented}/
│   ├── audio/{raw,augmented}/
│   └── processed/                  # merged_dataset.csv, image_features.csv, audio_features.csv
├── models/                         # face/voice/product .joblib + metrics.json
└── outputs/plots/                  # EDA, image grid, waveforms/spectrograms
```

## How to run
```bash
pip install -r requirements.txt

# Authorized full transaction (any customer_id_new from data/processed/merged_dataset.csv)
python app.py --image data/images/raw/member1_smiling.jpg \
              --audio data/audio/raw/member1_yes_approve.wav --customer A176

# Unauthorized attempt (denied at face step)
python app.py --image data/images/raw/impostor2_neutral.jpg \
              --audio data/audio/raw/member1_yes_approve.wav
```

## Current metrics (real tabular data; synthetic face/voice placeholders)
| Model | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|
| Face recognition (RandomForest) | 1.00 | 1.00 | 0.06 |
| Voiceprint verification (LogReg) | 0.88 | 0.87 | 0.20 |
| Product recommendation (RandomForest) | 0.20 | 0.19 | 1.62 — does not beat majority baseline (0.25); real dataset carries no product signal, documented honestly in notebook |

## Multimodal decision logic
`P(face authorized) ≥ 0.60` **AND** `P(voice authorized) ≥ 0.60` → product model runs; any failure → ACCESS DENIED.

## Team contributions
- Member 1 — *(fill in per member once the group divides the work)*
