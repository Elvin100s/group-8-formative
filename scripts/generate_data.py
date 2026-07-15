"""
generate_data.py
----------------
Creates synthetic stand-ins for the two course datasets:
  - customer_transactions.csv   (legacy system IDs)
  - customer_social_profiles.csv (new system IDs)
  - id_mapping.csv               (links legacy -> new IDs, justifying the join)

REPLACE the files in data/raw/ with the real course CSVs when available —
the rest of the pipeline reads from data/raw/ and will pick them up.
Deliberately injects nulls and duplicates so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 150
N_TX = 400

PRODUCTS = ["Electronics", "Clothing", "Home & Garden", "Beauty", "Sports"]
PLATFORMS = ["Instagram", "TikTok", "Facebook", "Twitter/X"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]


def main():
    if (RAW / "customer_transactions.csv").exists():
        print("Real datasets already present in data/raw/ — skipping synthetic generation.")
        return
    legacy_ids = [f"L{100 + i}" for i in range(N_CUSTOMERS)]
    new_ids = [f"A{5000 + i}" for i in range(N_CUSTOMERS)]

    # --- id mapping (legacy -> new) ---
    id_map = pd.DataFrame({"customer_id_legacy": legacy_ids, "customer_id_new": new_ids})

    # per-customer latent taste driven by (later) social features -> learnable signal
    interest = RNG.uniform(0, 10, N_CUSTOMERS)
    engagement = RNG.uniform(10, 100, N_CUSTOMERS)
    taste_idx = np.clip(((interest / 10) * 2.5 + (engagement / 100) * 2.5).astype(int), 0, 4)
    cust_pref = {lid: PRODUCTS[t] for lid, t in zip(legacy_ids, taste_idx)}

    # --- transactions (legacy IDs) ---
    buyers = RNG.choice(legacy_ids, N_TX)
    products = [cust_pref[b] if RNG.random() < 0.75 else RNG.choice(PRODUCTS) for b in buyers]
    tx = pd.DataFrame({
        "transaction_id": [f"T{2000 + i}" for i in range(N_TX)],
        "customer_id_legacy": buyers,
        "purchase_amount": np.round(RNG.gamma(3.0, 40.0, N_TX), 2),
        "purchase_date": pd.to_datetime("2025-01-01")
                         + pd.to_timedelta(RNG.integers(0, 180, N_TX), unit="D"),
        "product_category": products,
        "customer_rating": RNG.integers(1, 6, N_TX).astype(float),
    })
    # inject messiness: nulls + duplicates + a bad type
    tx.loc[RNG.choice(N_TX, 12, replace=False), "customer_rating"] = np.nan
    tx.loc[RNG.choice(N_TX, 6, replace=False), "purchase_amount"] = np.nan
    tx = pd.concat([tx, tx.sample(8, random_state=1)], ignore_index=True)  # dup rows
    tx["customer_rating"] = tx["customer_rating"].astype(object)  # wrong dtype on purpose

    # --- social profiles (new IDs); engagement correlates with product taste ---
    sp = pd.DataFrame({
        "customer_id_new": new_ids,
        "social_media_platform": RNG.choice(PLATFORMS, N_CUSTOMERS),
        "engagement_score": np.round(engagement, 1),
        "purchase_interest_score": np.round(interest, 2),
        "review_sentiment": RNG.choice(SENTIMENTS, N_CUSTOMERS, p=[0.5, 0.3, 0.2]),
    })
    sp.loc[RNG.choice(N_CUSTOMERS, 5, replace=False), "engagement_score"] = np.nan
    sp = pd.concat([sp, sp.sample(4, random_state=2)], ignore_index=True)

    tx.to_csv(RAW / "customer_transactions.csv", index=False)
    sp.to_csv(RAW / "customer_social_profiles.csv", index=False)
    id_map.to_csv(RAW / "id_mapping.csv", index=False)
    print(f"Wrote {len(tx)} transactions, {len(sp)} social profiles, {len(id_map)} id mappings -> {RAW}")


if __name__ == "__main__":
    main()
