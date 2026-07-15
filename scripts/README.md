# scripts — pipeline (run in this order)
`generate_data.py` (skips itself; real CSVs present) → `data_merge.py` → `image_pipeline.py` → `audio_pipeline.py` → `train_models.py`. `build_notebook.py` regenerates the notebook.
