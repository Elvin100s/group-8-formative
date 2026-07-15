"""
data_merge.py  —  Task 1: Data Merge + EDA + Feature Engineering
----------------------------------------------------------------
1. Load customer_transactions.csv (legacy IDs) and customer_social_profiles.csv (new IDs)
2. Clean: drop duplicates, fix dtypes, impute nulls (median for numeric, mode for categorical)
3. Merge: transactions -> id_mapping -> social profiles (inner join, justified below)
4. EDA: summary stats + 3 labeled plots (distribution, outliers, correlation)
5. Feature engineering: per-customer aggregates + encoded categoricals
6. Save data/processed/merged_dataset.csv

Join logic justification: the two sources use different customer ID schemes
(legacy vs new). id_mapping.csv is the bridge table. An INNER join is used
because a training row is only useful if it has BOTH transactional behaviour
(the label source) AND social features (the predictors); unmatched rows would
be all-null on one side and add nothing to a supervised model.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, PLOTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "outputs/plots"
PROC.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)


def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"[{name}] dropped {before - len(df)} duplicate rows")
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")) and \
               pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                fill = df[col].median()
            else:
                fill = df[col].mode().iloc[0]
            n = df[col].isna().sum()
            df[col] = df[col].fillna(fill)
            print(f"[{name}] imputed {n} nulls in '{col}' with {fill}")
    return df


def main():
    tx = pd.read_csv(RAW / "customer_transactions.csv", parse_dates=["purchase_date"])
    sp = pd.read_csv(RAW / "customer_social_profiles.csv")
    id_map = pd.read_csv(RAW / "id_mapping.csv")

    print("=== dtypes before cleaning ===")
    print(tx.dtypes, "\n", sp.dtypes, sep="")

    tx["customer_rating"] = pd.to_numeric(tx["customer_rating"], errors="coerce")  # fix dtype
    tx, sp = clean(tx, "transactions"), clean(sp, "social_profiles")

    # ---- resolve many-to-many ID mapping ----
    # The mapping table contains duplicated legacy AND new IDs (many-to-many).
    # Joining it as-is would fan out transactions (row explosion) and double-count
    # purchases. Resolution: enforce one canonical new ID per legacy ID (keep the
    # first mapping seen). This preserves every transaction exactly once.
    dup_legacy = id_map["customer_id_legacy"].duplicated().sum()
    if dup_legacy:
        print(f"[id_mapping] many-to-many detected: {dup_legacy} duplicate legacy IDs "
              f"-> deduplicated to one new ID per legacy ID")
        id_map = id_map.drop_duplicates(subset="customer_id_legacy", keep="first")

    # Social profiles can also contain multiple rows per customer_id_new
    # (e.g., several platforms). Collapse to one profile per customer:
    # numeric -> mean, categorical -> mode, so the tx join stays one-to-one.
    if sp["customer_id_new"].duplicated().any():
        n = sp["customer_id_new"].duplicated().sum()
        num_cols = sp.select_dtypes("number").columns.tolist()
        cat_cols = [c for c in sp.columns if c not in num_cols + ["customer_id_new"]]
        sp = sp.groupby("customer_id_new").agg(
            {**{c: "mean" for c in num_cols},
             **{c: (lambda s: s.mode().iloc[0]) for c in cat_cols}}).reset_index()
        print(f"[social_profiles] collapsed {n} extra rows -> one profile per customer")

    # ---- merge: tx (legacy) -> mapping -> social (new) ----
    merged = tx.merge(id_map, on="customer_id_legacy", how="inner") \
               .merge(sp, on="customer_id_new", how="inner")

    # ---- post-merge validation ----
    print("\n=== post-merge checks ===")
    print(f"rows: tx={len(tx)}  merged={len(merged)}  (match rate {len(merged)/len(tx):.1%})")
    print(f"null cells after merge: {merged.isna().sum().sum()}")
    print(f"unique customers: {merged['customer_id_new'].nunique()}")
    assert merged.isna().sum().sum() == 0, "nulls survived the merge!"
    assert not merged.duplicated(subset=["transaction_id"]).any(), "duplicate transactions!"

    # ---- EDA ----
    print("\n=== summary statistics ===")
    print(merged.describe(include="all").T.head(15))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(merged["purchase_amount"], bins=30, edgecolor="black")
    ax.set(title="Distribution of Purchase Amount", xlabel="Purchase amount ($)", ylabel="Frequency")
    fig.tight_layout(); fig.savefig(PLOTS / "eda_purchase_distribution.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    merged.boxplot(column="purchase_amount", by="product_category", ax=ax)
    ax.set(title="Purchase Amount by Product (outlier view)", xlabel="", ylabel="Purchase amount ($)")
    plt.suptitle(""); fig.tight_layout()
    fig.savefig(PLOTS / "eda_outliers_boxplot.png", dpi=120); plt.close(fig)

    # ---- feature engineering ----
    merged["purchase_month"] = merged["purchase_date"].dt.month
    merged["purchase_dow"] = merged["purchase_date"].dt.dayofweek
    agg = merged.groupby("customer_id_new").agg(
        total_spend=("purchase_amount", "sum"),
        avg_spend=("purchase_amount", "mean"),
        n_transactions=("transaction_id", "count"),
        avg_rating=("customer_rating", "mean"),
    ).reset_index()
    merged = merged.merge(agg, on="customer_id_new")
    merged["engagement_x_interest"] = merged["engagement_score"] * merged["purchase_interest_score"]

    # correlation heatmap: meaningful numerics only (IDs excluded), engineered features included
    corr_cols = ["purchase_amount", "customer_rating", "engagement_score",
                 "purchase_interest_score", "total_spend", "avg_spend",
                 "n_transactions", "avg_rating", "engagement_x_interest"]
    corr = merged[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)), corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title("Correlation Matrix — raw + engineered features")
    fig.tight_layout(); fig.savefig(PLOTS / "eda_correlation_heatmap.png", dpi=120); plt.close(fig)

    merged = pd.get_dummies(merged, columns=["social_media_platform", "review_sentiment"], dtype=int)

    merged.to_csv(PROC / "merged_dataset.csv", index=False)
    print(f"\nSaved merged + engineered dataset: {merged.shape} -> {PROC/'merged_dataset.csv'}")


if __name__ == "__main__":
    main()
