# Formative 2 — Multimodal Data Preprocessing
User Identity & Product Recommendation System: face recognition → voiceprint verification → product recommendation, with ACCESS DENIED pathways on any failed step.

**Demo video:** https://youtu.be/5quay27JCxw

## Data
- **Tabular:** course datasets (`customer_transactions.csv`, `customer_social_profiles.csv`, `id_mapping.csv`) in `data/raw/`.
- **Images:** 3 expressions (neutral / smiling / surprised) per team member + 2 impostor identities, in `data/images/raw/`.
- **Audio:** 2 phrases ("Yes, approve", "Confirm transaction") per member + 4 impostor identities, normalized to 22.05 kHz mono WAV, in `data/audio/raw/`.

## Repo structure
```
├── app.py                          # CLI simulation (Task 6)
├── Formative2_Multimodal_Pipeline.ipynb  # executed notebook, full narrative
├── scripts/
│   ├── generate_data.py            # synthetic CSV fallback (skips itself; real data present)
│   ├── data_merge.py               # Task 1: clean, merge, EDA, feature engineering
│   ├── image_pipeline.py           # Task 2: collect, augment, extract → image_features.csv
│   ├── audio_pipeline.py           # Task 3: collect, visualize, augment → audio_features.csv
│   ├── train_models.py             # Task 4: 3 models + accuracy/F1/loss → models/
│   └── build_notebook.py           # regenerates the notebook
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

python scripts/data_merge.py
python scripts/image_pipeline.py
python scripts/audio_pipeline.py
python scripts/train_models.py

# Authorized full transaction (any customer_id_new from data/processed/merged_dataset.csv)
python app.py --image data/images/raw/member1_smiling.jpg \
              --audio data/audio/raw/member1_yes_approve.wav --customer A176

# Unauthorized attempt (denied at face step)
python app.py --image data/images/raw/impostor2_neutral.png \
              --audio data/audio/raw/member1_yes_approve.wav
```

## Metrics
| Model | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|
| Face recognition (RandomForest) | 1.00 | 1.00 | 0.11 |
| Voiceprint verification (LogReg) | 1.00 | 1.00 | 0.15 |
| Product recommendation (RandomForest) | 0.20 | 0.19 | 1.62 — does not beat majority baseline (0.25); dataset carries no product signal, documented honestly in notebook |

## Multimodal decision logic
`P(face authorized) ≥ 0.60` **AND** `P(voice authorized) ≥ 0.60` → product model runs; any failure → ACCESS DENIED.

## Team contributions
| Member | GitHub | Contributions |
|---|---|---|
| Elvin Cyubahiro (member1) | @Elvin100s | Repo setup & architecture; Task 1 merge/EDA script; CLI app (Task 6); media normalization (WAV/JPG); impostor face images; pipeline execution & integration; report compilation & fact checking; docs |
| Eddy Irasetsa (member2) | @Eddydev-ALU | Task 3 audio pipeline; notebook builder; repo hygiene; demo video recording; member2 photos & recordings; impostor2 recordings |
| Iriza Larissa (member3) | @Larissa4-droid | Task 2 image pipeline; Task 4 model training & evaluation; report writing & review; member3 photos & recordings; impostor3 recordings |
| Mutumwinka Heroine (member4) | @h-mutumwinka | data_merge & generate_data scripts; EDA write-up; member4 photos & recordings; impostor4 recordings |

Individual work is verifiable in the [commit history](https://github.com/Elvin100s/group-8-formative/commits/main).
