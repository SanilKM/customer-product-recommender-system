"""
Cluster-First kNN Recommender

Latest version includes:
- CLI args: --dataset-name, --outlier-method, --n-clusters, --n-neighbors
- scalable approximate kNN reference sampling for large 1-5 lac datasets
- exact cluster feature values and top defining features
- outlier treatment summary
- cumulative Hit@1, Hit@2, Hit@3
- rank-1 confusion matrix
- runtime / tech specs by step
- report-ready visualizations

Example:
python src/cluster_first_knn_recommender.py \
    --raw-dir data/01_raw/Syn_01 \
    --conversion-data data/01_raw/Syn_01/conversion_data_july_2026.csv \
    --output-dir data/06_cluster_first_knn_outputs/Syn_01 \
    --dataset-name Syn_01 \
    --outlier-method iqr
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PRODUCTS = ["PL", "CC", "HL", "SA", "RD", "MF"]

HOLDING_COLS = {
    "PL": "PL_count",
    "CC": "CC_count",
    "HL": "HL_count",
    "SA": "SA_count",
    "RD": "RD_count",
    "MF": "MF_count",
}

PRODUCT_NAME_MAP = {
    "PL": "Personal Loan",
    "CC": "Credit Card",
    "HL": "Home Loan",
    "SA": "Savings Account",
    "RD": "Recurring Deposit",
    "MF": "Mutual Fund",
}

FEATURE_GROUP_WEIGHTS = {
    "product_holdings": 0.35,
    "customer_numeric_profile": 0.25,
    "customer_categorical_profile": 0.20,
    "digital_behavior": 0.10,
    "transaction_behavior": 0.10,
}

PREFERRED_NUMERIC_FEATURES = [
    "Age",
    "Monthly income proxy (sn)",
    "Risk profile score",
    "Days since last login (sn)",
    "Campaign exposure count 90D (sn)",
    "Products ignored count 90D (sn)",
    "Mobile no.",
]

PREFERRED_CATEGORICAL_FEATURES = [
    "Persona",
    "Lifestage",
    "Income band (sn)",
    "Tier Map",
    "Occupation",
    "Occupation desc",
    "Preferred channel (sn)",
    "Gender",
    "City",
    "Constitution",
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-first kNN recommender for one dataset folder.")

    parser.add_argument("--raw-dir", type=str, required=True, help="Folder containing raw CSV files.")
    parser.add_argument("--conversion-data", type=str, default=None, help="Optional conversion CSV path. Defaults to raw-dir/conversion_data_july_2026.csv.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Name to store in outputs, e.g. Syn_01.")

    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-onehot-cardinality", type=int, default=40)
    parser.add_argument("--outlier-method", type=str, default="iqr", choices=["none", "iqr", "quantile"])
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)

    # Large data safety settings. Full brute-force kNN on 5 lac customers can be very slow.
    parser.add_argument("--knn-reference-limit", type=int, default=10000,
                        help="Max reference customers per cluster for kNN. If a cluster is larger, a reproducible sample is used.")
    parser.add_argument("--knn-batch-size", type=int, default=5000,
                        help="Batch size for kNN queries.")
    parser.add_argument("--visualization-sample-size", type=int, default=3000)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Loading and basic cleaning
# ---------------------------------------------------------------------


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def normalize_customer_id_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    customer_aliases = ["Customer ID", "CustID", "Customer_ID", "customer_id", "cust_id"]
    for col in customer_aliases:
        if col in df.columns and col != "Customer ID":
            return df.rename(columns={col: "Customer ID"})
    return df


def add_month_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Month period" in df.columns:
        df["Month period"] = df["Month period"].astype(str)
        return df

    if "Monthly period" in df.columns:
        df = df.rename(columns={"Monthly period": "Month period"})
        df["Month period"] = df["Month period"].astype(str)
        return df

    if "Year" in df.columns and "Month" in df.columns:
        df["Month period"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
        return df

    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        df["Month period"] = parsed.dt.to_period("M").astype(str)
        return df

    df["Month period"] = "unknown"
    return df


def latest_month_filter(df: pd.DataFrame, target_month: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    df = add_month_period(df)
    months = sorted([m for m in df["Month period"].dropna().astype(str).unique() if m != "NaT"])
    if not months:
        return df, "unknown"

    chosen = target_month if target_month in months else months[-1]
    return df[df["Month period"].astype(str) == chosen].copy(), chosen


def product_code_from_text(value: str) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in PRODUCTS:
        return upper
    for code, name in PRODUCT_NAME_MAP.items():
        if upper == name.upper():
            return code
    return None


def load_customer_data(raw_dir: Path) -> pd.DataFrame:
    holdings = normalize_customer_id_col(read_csv_if_exists(raw_dir / "product_holdings.csv"))
    demographics = normalize_customer_id_col(read_csv_if_exists(raw_dir / "customer_demographics.csv"))
    digital_login = normalize_customer_id_col(read_csv_if_exists(raw_dir / "digital_login.csv"))
    transaction_aggregates = normalize_customer_id_col(read_csv_if_exists(raw_dir / "transaction_aggregates.csv"))

    if holdings.empty or "Customer ID" not in holdings.columns:
        raise FileNotFoundError(f"Missing or invalid product_holdings.csv in {raw_dir}")

    holdings, feature_month = latest_month_filter(holdings)

    for product, col in HOLDING_COLS.items():
        if col in holdings.columns:
            holdings[col] = pd.to_numeric(holdings[col], errors="coerce").fillna(0)
            holdings[product] = (holdings[col] > 0).astype(int)
        elif product in holdings.columns:
            holdings[product] = pd.to_numeric(holdings[product], errors="coerce").fillna(0).clip(0, 1).astype(int)
        else:
            holdings[product] = 0

    customer_df = holdings[["Customer ID", "Month period"] + PRODUCTS].drop_duplicates("Customer ID").copy()
    customer_df["feature_month"] = feature_month

    if not demographics.empty and "Customer ID" in demographics.columns:
        demographics = add_month_period(demographics)
        if feature_month in set(demographics["Month period"].dropna().astype(str)):
            demographics = demographics[demographics["Month period"].astype(str) == feature_month].copy()
        demographics = demographics.drop_duplicates("Customer ID")
        demographics = demographics.drop(columns=["Month period"], errors="ignore")
        customer_df = customer_df.merge(demographics, on="Customer ID", how="left")

    if not digital_login.empty and "Customer ID" in digital_login.columns:
        digital_login = add_month_period(digital_login)
        numeric_cols = digital_login.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["Month", "Year"]]
        if numeric_cols:
            dig = digital_login.groupby("Customer ID", as_index=False)[numeric_cols].mean()
            rename = {c: f"digital_{c}" for c in numeric_cols if c != "Customer ID"}
            dig = dig.rename(columns=rename)
            customer_df = customer_df.merge(dig, on="Customer ID", how="left")

    if not transaction_aggregates.empty and "Customer ID" in transaction_aggregates.columns:
        transaction_aggregates = add_month_period(transaction_aggregates)
        numeric_cols = transaction_aggregates.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["Month", "Year"]]
        if numeric_cols:
            txn = transaction_aggregates.groupby("Customer ID", as_index=False)[numeric_cols].mean()
            rename = {c: f"txn_{c}" for c in numeric_cols if c != "Customer ID"}
            txn = txn.rename(columns=rename)
            customer_df = customer_df.merge(txn, on="Customer ID", how="left")

    return customer_df


# ---------------------------------------------------------------------
# Feature matrix and outlier treatment
# ---------------------------------------------------------------------


def choose_feature_columns(customer_df: pd.DataFrame, max_onehot_cardinality: int):
    id_cols = {"Customer ID", "Month period", "feature_month"}
    holding_cols = [c for c in PRODUCTS if c in customer_df.columns]

    numeric_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in id_cols and c not in holding_cols]

    digital_cols = [c for c in numeric_cols if c.startswith("digital_")]
    txn_cols = [c for c in numeric_cols if c.startswith("txn_")]
    customer_numeric_cols = [c for c in numeric_cols if c not in digital_cols and c not in txn_cols]

    object_cols = customer_df.select_dtypes(include=["object", "category"]).columns.tolist()
    object_cols = [c for c in object_cols if c not in id_cols]

    categorical_cols = []
    for col in object_cols:
        nunique = customer_df[col].nunique(dropna=True)
        if 1 < nunique <= max_onehot_cardinality:
            categorical_cols.append(col)

    # Prefer known business columns first, then all additional usable columns.
    customer_numeric_cols = [c for c in PREFERRED_NUMERIC_FEATURES if c in customer_numeric_cols] + [
        c for c in customer_numeric_cols if c not in PREFERRED_NUMERIC_FEATURES
    ]
    categorical_cols = [c for c in PREFERRED_CATEGORICAL_FEATURES if c in categorical_cols] + [
        c for c in categorical_cols if c not in PREFERRED_CATEGORICAL_FEATURES
    ]

    return holding_cols, customer_numeric_cols, categorical_cols, digital_cols, txn_cols


def treat_outliers(block: pd.DataFrame, method: str, lower_q: float, upper_q: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    treated = block.copy()

    for col in treated.columns:
        s = pd.to_numeric(treated[col], errors="coerce")
        min_before = s.min()
        max_before = s.max()

        if method == "none" or s.dropna().empty:
            lower = np.nan
            upper = np.nan
            low_count = 0
            high_count = 0
            clipped = s
        elif method == "quantile":
            lower = s.quantile(lower_q)
            upper = s.quantile(upper_q)
            low_count = int((s < lower).sum())
            high_count = int((s > upper).sum())
            clipped = s.clip(lower=lower, upper=upper)
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            low_count = int((s < lower).sum())
            high_count = int((s > upper).sum())
            clipped = s.clip(lower=lower, upper=upper)

        treated[col] = clipped
        summary_rows.append({
            "feature": col,
            "method": method,
            "lower_cap": lower,
            "upper_cap": upper,
            "values_capped_low": low_count,
            "values_capped_high": high_count,
            "min_before": min_before,
            "max_before": max_before,
            "min_after": treated[col].min(),
            "max_after": treated[col].max(),
        })

    return treated, pd.DataFrame(summary_rows)


def scale_numeric_block(
    df: pd.DataFrame,
    cols: List[str],
    outlier_method: str,
    lower_q: float,
    upper_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cols:
        return pd.DataFrame(index=df.index), pd.DataFrame()

    block = df[cols].copy()
    for col in cols:
        block[col] = pd.to_numeric(block[col], errors="coerce")

    block = block.replace([np.inf, -np.inf], np.nan)
    block = block.fillna(block.median(numeric_only=True)).fillna(0)

    treated, outlier_summary = treat_outliers(block, outlier_method, lower_q, upper_q)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(treated)
    scaled_df = pd.DataFrame(scaled, columns=cols, index=df.index)

    return scaled_df, outlier_summary


def build_customer_feature_matrix(
    customer_df: pd.DataFrame,
    max_onehot_cardinality: int,
    outlier_method: str,
    lower_q: float,
    upper_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holding_cols, numeric_cols, categorical_cols, digital_cols, txn_cols = choose_feature_columns(customer_df, max_onehot_cardinality)

    blocks = []
    metadata_rows = []
    outlier_summaries = []

    def add_block(group_name: str, block: pd.DataFrame):
        weight = FEATURE_GROUP_WEIGHTS[group_name]
        if block.empty:
            metadata_rows.append({
                "feature_group": group_name,
                "feature_count": 0,
                "group_weight": weight,
                "columns_used": "",
            })
            return

        weighted = block.astype(float) * np.sqrt(weight)
        weighted.columns = [f"{group_name}__{c}" for c in weighted.columns]
        blocks.append(weighted)
        metadata_rows.append({
            "feature_group": group_name,
            "feature_count": block.shape[1],
            "group_weight": weight,
            "columns_used": ", ".join(block.columns.astype(str).tolist()[:150]),
        })

    holding_block = customer_df[holding_cols].fillna(0).astype(float)

    numeric_block, numeric_outliers = scale_numeric_block(customer_df, numeric_cols, outlier_method, lower_q, upper_q)
    digital_block, digital_outliers = scale_numeric_block(customer_df, digital_cols, outlier_method, lower_q, upper_q)
    txn_block, txn_outliers = scale_numeric_block(customer_df, txn_cols, outlier_method, lower_q, upper_q)

    for group_name, out_df in [
        ("customer_numeric_profile", numeric_outliers),
        ("digital_behavior", digital_outliers),
        ("transaction_behavior", txn_outliers),
    ]:
        if not out_df.empty:
            out_df = out_df.copy()
            out_df.insert(0, "feature_group", group_name)
            outlier_summaries.append(out_df)

    if categorical_cols:
        cat = customer_df[categorical_cols].fillna("Missing").astype(str)
        categorical_block = pd.get_dummies(cat, columns=categorical_cols).astype(float)
    else:
        categorical_block = pd.DataFrame(index=customer_df.index)

    add_block("product_holdings", holding_block)
    add_block("customer_numeric_profile", numeric_block)
    add_block("customer_categorical_profile", categorical_block)
    add_block("digital_behavior", digital_block)
    add_block("transaction_behavior", txn_block)

    if not blocks:
        raise ValueError("No usable customer similarity features were found.")

    feature_matrix = pd.concat(blocks, axis=1)
    feature_matrix.insert(0, "Customer ID", customer_df["Customer ID"].values)

    metadata = pd.DataFrame(metadata_rows)
    outlier_summary = pd.concat(outlier_summaries, ignore_index=True) if outlier_summaries else pd.DataFrame()

    return feature_matrix, metadata, outlier_summary


def build_product_matrix(customer_df: pd.DataFrame) -> pd.DataFrame:
    return customer_df.set_index("Customer ID")[PRODUCTS].fillna(0).astype(float)


# ---------------------------------------------------------------------
# Clustering and cluster profiling
# ---------------------------------------------------------------------


def create_kmeans_clusters(
    feature_matrix: pd.DataFrame,
    customer_df: pd.DataFrame,
    n_clusters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = feature_matrix.set_index("Customer ID").values.astype(np.float32)
    customer_ids = feature_matrix["Customer ID"].values
    n_clusters = min(n_clusters, len(customer_ids))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        batch_size=4096,
        max_iter=100,
    )
    cluster_ids = kmeans.fit_predict(X)

    clusters = pd.DataFrame({"Customer ID": customer_ids, "cluster_id": cluster_ids})
    clustered_customers = customer_df.merge(clusters, on="Customer ID", how="left")

    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers.insert(0, "cluster_id", range(n_clusters))

    return clustered_customers, centers


def summarize_clusters(clustered_customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_numeric = [c for c in PREFERRED_NUMERIC_FEATURES if c in clustered_customers.columns]
    selected_numeric += [c for c in clustered_customers.columns if c.startswith("digital_") or c.startswith("txn_")]

    selected_categorical = [c for c in PREFERRED_CATEGORICAL_FEATURES if c in clustered_customers.columns]

    profile_rows = []
    exact_rows = []

    for cluster_id, group in clustered_customers.groupby("cluster_id"):
        row = {
            "cluster_id": cluster_id,
            "customer_count": len(group),
            "customer_share": len(group) / len(clustered_customers),
        }

        for product in PRODUCTS:
            rate = pd.to_numeric(group[product], errors="coerce").fillna(0).mean()
            row[f"{product}_holding_rate"] = rate
            exact_rows.append({
                "cluster_id": cluster_id,
                "feature_type": "product_holding_rate",
                "feature": product,
                "mean": rate,
                "min": group[product].min(),
                "p25": group[product].quantile(0.25),
                "median": group[product].median(),
                "p75": group[product].quantile(0.75),
                "max": group[product].max(),
                "mode_or_top_value": "",
                "top_value_share": np.nan,
            })

        for col in selected_numeric:
            s = pd.to_numeric(group[col], errors="coerce")
            row[f"avg_{col}"] = s.mean()
            row[f"min_{col}"] = s.min()
            row[f"p25_{col}"] = s.quantile(0.25)
            row[f"median_{col}"] = s.median()
            row[f"p75_{col}"] = s.quantile(0.75)
            row[f"max_{col}"] = s.max()
            exact_rows.append({
                "cluster_id": cluster_id,
                "feature_type": "numeric",
                "feature": col,
                "mean": s.mean(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
                "mode_or_top_value": "",
                "top_value_share": np.nan,
            })

        for col in selected_categorical:
            series = group[col].fillna("Missing").astype(str)
            mode = series.mode()
            top_value = mode.iloc[0] if not mode.empty else ""
            top_share = float(series.eq(top_value).mean()) if top_value else np.nan
            row[f"top_{col}"] = top_value
            row[f"top_{col}_share"] = top_share
            exact_rows.append({
                "cluster_id": cluster_id,
                "feature_type": "categorical",
                "feature": col,
                "mean": np.nan,
                "min": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "max": np.nan,
                "mode_or_top_value": top_value,
                "top_value_share": top_share,
            })

        profile_rows.append(row)

    cluster_profiles = pd.DataFrame(profile_rows).sort_values("cluster_id")
    exact_values = pd.DataFrame(exact_rows).sort_values(["cluster_id", "feature_type", "feature"])

    product_rate_cols = [f"{p}_holding_rate" for p in PRODUCTS]
    product_rates_long = cluster_profiles[["cluster_id", "customer_count"] + product_rate_cols].melt(
        id_vars=["cluster_id", "customer_count"],
        value_vars=product_rate_cols,
        var_name="product_code",
        value_name="cluster_product_holding_rate",
    )
    product_rates_long["product_code"] = product_rates_long["product_code"].str.replace("_holding_rate", "", regex=False)
    product_rates_long["Product_Name"] = product_rates_long["product_code"].map(PRODUCT_NAME_MAP)

    return cluster_profiles, exact_values, product_rates_long


def compute_defining_features(clustered_customers: pd.DataFrame, top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for product in PRODUCTS:
        overall = pd.to_numeric(clustered_customers[product], errors="coerce").fillna(0).mean()
        for cluster_id, group in clustered_customers.groupby("cluster_id"):
            val = pd.to_numeric(group[product], errors="coerce").fillna(0).mean()
            diff = val - overall
            rows.append({
                "cluster_id": cluster_id,
                "feature_type": "product_holding",
                "feature": f"{product} holding rate",
                "cluster_exact_value": val,
                "overall_exact_value": overall,
                "difference": diff,
                "abs_difference": abs(diff),
                "standardized_difference_for_ranking": abs(diff),
                "direction": "higher than overall" if diff > 0 else "lower than overall",
            })

    numeric_cols = [c for c in PREFERRED_NUMERIC_FEATURES if c in clustered_customers.columns]
    numeric_cols += [c for c in clustered_customers.columns if c.startswith("digital_") or c.startswith("txn_")]

    for col in numeric_cols:
        all_s = pd.to_numeric(clustered_customers[col], errors="coerce")
        overall = all_s.mean()
        overall_std = all_s.std()
        overall_std = overall_std if pd.notna(overall_std) and overall_std > 0 else 1.0

        for cluster_id, group in clustered_customers.groupby("cluster_id"):
            val = pd.to_numeric(group[col], errors="coerce").mean()
            diff = val - overall
            rows.append({
                "cluster_id": cluster_id,
                "feature_type": "numeric",
                "feature": col,
                "cluster_exact_value": val,
                "overall_exact_value": overall,
                "difference": diff,
                "abs_difference": abs(diff),
                "standardized_difference_for_ranking": abs(diff / overall_std),
                "direction": "higher than overall" if diff > 0 else "lower than overall",
            })

    cat_cols = [c for c in PREFERRED_CATEGORICAL_FEATURES if c in clustered_customers.columns]
    for col in cat_cols:
        full_series = clustered_customers[col].fillna("Missing").astype(str)
        for cluster_id, group in clustered_customers.groupby("cluster_id"):
            series = group[col].fillna("Missing").astype(str)
            if series.empty:
                continue
            top_value = series.mode().iloc[0]
            cluster_share = series.eq(top_value).mean()
            overall_share = full_series.eq(top_value).mean()
            diff = cluster_share - overall_share
            rows.append({
                "cluster_id": cluster_id,
                "feature_type": "categorical",
                "feature": f"{col} = {top_value}",
                "cluster_exact_value": cluster_share,
                "overall_exact_value": overall_share,
                "difference": diff,
                "abs_difference": abs(diff),
                "standardized_difference_for_ranking": abs(diff),
                "direction": "more common than overall" if diff > 0 else "less common than overall",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    df = df.sort_values(["cluster_id", "standardized_difference_for_ranking"], ascending=[True, False])
    top = df.groupby("cluster_id", as_index=False, group_keys=False).head(top_k)
    return df, top


# ---------------------------------------------------------------------
# kNN recommendation
# ---------------------------------------------------------------------


def run_cluster_first_knn(
    feature_matrix: pd.DataFrame,
    product_matrix: pd.DataFrame,
    clustered_customers: pd.DataFrame,
    n_neighbors: int,
    reference_limit: int,
    batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_indexed = feature_matrix.set_index("Customer ID")
    product_matrix = product_matrix.loc[feature_indexed.index]
    cluster_lookup = clustered_customers.set_index("Customer ID")["cluster_id"]

    rng = np.random.default_rng(seed)
    signal_frames = []
    neighbor_rows = []

    for cluster_id in sorted(cluster_lookup.dropna().unique()):
        ids = cluster_lookup[cluster_lookup == cluster_id].index.to_numpy()
        if len(ids) == 0:
            continue

        X_all = feature_indexed.loc[ids].values.astype(np.float32)
        H_all = product_matrix.loc[ids].values.astype(np.float32)
        n_customers = len(ids)

        # Use all customers if cluster is small. For large clusters, use a sample as kNN reference.
        if n_customers > reference_limit:
            ref_positions = np.sort(rng.choice(np.arange(n_customers), size=reference_limit, replace=False))
            approximate_knn_used = True
        else:
            ref_positions = np.arange(n_customers)
            approximate_knn_used = False

        ref_ids = ids[ref_positions]
        X_ref = X_all[ref_positions]
        H_ref = H_all[ref_positions]

        if len(ref_ids) <= 1:
            scores = np.zeros_like(H_all)
            support_counts = np.zeros_like(H_all)
            avg_sims = np.zeros(n_customers)
            top_sims = np.zeros(n_customers)
        else:
            k = min(n_neighbors + 1, len(ref_ids))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            nn.fit(X_ref)

            scores = np.zeros_like(H_all, dtype=np.float32)
            support_counts = np.zeros_like(H_all, dtype=np.int32)
            avg_sims = np.zeros(n_customers, dtype=np.float32)
            top_sims = np.zeros(n_customers, dtype=np.float32)

            ref_id_to_pos = {cid: pos for pos, cid in enumerate(ref_ids)}

            for start in range(0, n_customers, batch_size):
                end = min(start + batch_size, n_customers)
                X_batch = X_all[start:end]
                batch_ids = ids[start:end]
                distances, indices = nn.kneighbors(X_batch)

                for local_i, cust_id in enumerate(batch_ids):
                    neigh_idx = indices[local_i]
                    neigh_dist = distances[local_i]

                    # Exclude self only when self is part of reference sample.
                    if cust_id in ref_id_to_pos:
                        self_ref_pos = ref_id_to_pos[cust_id]
                        mask = neigh_idx != self_ref_pos
                        neigh_idx = neigh_idx[mask]
                        neigh_dist = neigh_dist[mask]

                    neigh_idx = neigh_idx[:n_neighbors]
                    neigh_dist = neigh_dist[:n_neighbors]
                    sims = np.clip(1 - neigh_dist, 0, None).astype(np.float32)

                    global_i = start + local_i
                    if len(sims) and sims.sum() > 0:
                        weighted_scores = sims.reshape(1, -1).dot(H_ref[neigh_idx]).ravel() / sims.sum()
                    else:
                        weighted_scores = np.zeros(len(PRODUCTS), dtype=np.float32)

                    # Never recommend products the customer already holds.
                    weighted_scores = np.where(H_all[global_i] == 1, 0.0, weighted_scores)
                    scores[global_i] = weighted_scores
                    support_counts[global_i] = H_ref[neigh_idx].sum(axis=0).astype(np.int32) if len(neigh_idx) else 0
                    avg_sims[global_i] = float(sims.mean()) if len(sims) else 0.0
                    top_sims[global_i] = float(sims[0]) if len(sims) else 0.0

                    # Keep only a small neighbor sample for explainability.
                    if len(neighbor_rows) < 5000:
                        for rank, (ref_j, sim) in enumerate(zip(neigh_idx[:5], sims[:5]), start=1):
                            neighbor_rows.append({
                                "Customer ID": cust_id,
                                "cluster_id": cluster_id,
                                "neighbor_rank": rank,
                                "neighbor_customer_id": ref_ids[ref_j],
                                "within_cluster_cosine_similarity": float(sim),
                                "approximate_knn_used": approximate_knn_used,
                            })

        base = pd.DataFrame({
            "Customer ID": np.repeat(ids, len(PRODUCTS)),
            "cluster_id": cluster_id,
            "product_code": PRODUCTS * n_customers,
            "Product_Name": [PRODUCT_NAME_MAP[p] for p in PRODUCTS] * n_customers,
            "has_product": H_all.reshape(-1).astype(int),
            "cluster_first_knn_score": scores.reshape(-1),
            "neighbor_support_count": support_counts.reshape(-1),
            "avg_neighbor_similarity": np.repeat(avg_sims, len(PRODUCTS)),
            "top_neighbor_similarity": np.repeat(top_sims, len(PRODUCTS)),
            "neighbors_requested": n_neighbors,
            "neighbors_available_in_cluster": min(max(len(ref_ids) - 1, 0), n_neighbors),
            "knn_reference_customers_in_cluster": len(ref_ids),
            "cluster_customer_count": n_customers,
            "approximate_knn_used": approximate_knn_used,
        })

        base["eligible_for_recommendation"] = (base["has_product"] == 0).astype(int)
        base["cluster_first_knn_recommendation_score"] = np.where(
            base["eligible_for_recommendation"] == 1,
            base["cluster_first_knn_score"],
            0.0,
        )
        signal_frames.append(base)

    signal_df = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    neighbor_df = pd.DataFrame(neighbor_rows)

    return signal_df, neighbor_df


def create_top_n_recommendations(signal_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = signal_df[signal_df["eligible_for_recommendation"] == 1].copy()
    candidates = candidates.sort_values(
        ["Customer ID", "cluster_first_knn_recommendation_score", "neighbor_support_count"],
        ascending=[True, False, False],
    )
    candidates["recommendation_rank"] = candidates.groupby("Customer ID").cumcount() + 1
    top = candidates[candidates["recommendation_rank"] <= top_n].copy()
    top["recommendation_model"] = "Cluster-First kNN"

    keep = [
        "Customer ID",
        "cluster_id",
        "recommendation_model",
        "recommendation_rank",
        "product_code",
        "Product_Name",
        "cluster_first_knn_recommendation_score",
        "neighbor_support_count",
        "avg_neighbor_similarity",
        "top_neighbor_similarity",
        "neighbors_requested",
        "neighbors_available_in_cluster",
        "knn_reference_customers_in_cluster",
        "cluster_customer_count",
        "approximate_knn_used",
    ]
    return top[keep]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def load_conversions(conversion_path: Optional[Path]) -> pd.DataFrame:
    if conversion_path is None or not conversion_path.exists():
        return pd.DataFrame()

    conv = normalize_customer_id_col(pd.read_csv(conversion_path))
    if "Product converted" not in conv.columns:
        for alt in ["Product_converted", "product_converted", "product_code", "Product"]:
            if alt in conv.columns:
                conv = conv.rename(columns={alt: "Product converted"})
                break

    if "Product converted" not in conv.columns or "Customer ID" not in conv.columns:
        return pd.DataFrame()

    conv = conv[["Customer ID", "Product converted"]].dropna().drop_duplicates()
    conv["Product converted"] = conv["Product converted"].map(lambda x: product_code_from_text(x))
    conv = conv[conv["Product converted"].isin(PRODUCTS)].copy()
    return conv


def evaluate_recommendations(top_recs: pd.DataFrame, conversion_path: Optional[Path], top_n: int):
    conversions = load_conversions(conversion_path)
    if conversions.empty:
        note = pd.DataFrame([{
            "model": "Cluster-First kNN",
            "metric": "evaluation_status",
            "value": np.nan,
            "note": "No valid conversion data supplied; evaluation skipped.",
        }])
        return note, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    recs = top_recs[["Customer ID", "product_code", "recommendation_rank"]].copy()
    joined = conversions.merge(
        recs,
        left_on=["Customer ID", "Product converted"],
        right_on=["Customer ID", "product_code"],
        how="left",
    )

    eval_rows = []
    for k in range(1, top_n + 1):
        eval_rows.append({
            "model": "Cluster-First kNN",
            "metric": f"cumulative_hit_rate_at_{k}",
            "value": float(joined["recommendation_rank"].le(k).sum() / len(conversions)) if len(conversions) else np.nan,
            "converted_customers": conversions["Customer ID"].nunique(),
            "evaluated_conversion_rows": len(conversions),
        })

    ranks = joined["recommendation_rank"].dropna()
    eval_rows.append({
        "model": "Cluster-First kNN",
        "metric": "mrr",
        "value": float((1 / ranks).sum() / len(conversions)) if len(conversions) else 0.0,
        "converted_customers": conversions["Customer ID"].nunique(),
        "evaluated_conversion_rows": len(conversions),
    })

    eval_long = pd.DataFrame(eval_rows)

    wide = {
        "model": "Cluster-First kNN",
        "converted_customers": conversions["Customer ID"].nunique(),
        "evaluated_conversion_rows": len(conversions),
        "mrr": float((1 / ranks).sum() / len(conversions)) if len(conversions) else 0.0,
        "avg_rank_if_hit": float(ranks.mean()) if len(ranks) else np.nan,
    }
    for k in range(1, top_n + 1):
        wide[f"cumulative_hit_rate_at_{k}"] = float(joined["recommendation_rank"].le(k).sum() / len(conversions)) if len(conversions) else np.nan
    eval_wide = pd.DataFrame([wide])

    # Rank-1 confusion matrix.
    rank1 = top_recs[top_recs["recommendation_rank"] == 1][["Customer ID", "product_code"]].rename(
        columns={"product_code": "predicted_rank1_product"}
    )
    conf_base = conversions.merge(rank1, on="Customer ID", how="left")
    conf_base["predicted_rank1_product"] = conf_base["predicted_rank1_product"].fillna("NO_RECOMMENDATION")

    labels = PRODUCTS + ["NO_RECOMMENDATION"]
    cm = confusion_matrix(conf_base["Product converted"], conf_base["predicted_rank1_product"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"predicted_{x}" for x in labels])
    cm_df = cm_df.reset_index().rename(columns={"index": "actual_product"})

    # Product-wise top-3 hit rate.
    topn_join = conversions.merge(
        top_recs[top_recs["recommendation_rank"] <= top_n][["Customer ID", "product_code"]],
        left_on=["Customer ID", "Product converted"],
        right_on=["Customer ID", "product_code"],
        how="left",
    )
    product_hit = topn_join.groupby("Product converted", as_index=False).agg(
        conversions=("Customer ID", "count"),
        topn_hits=("product_code", lambda s: int(s.notna().sum())),
    )
    product_hit["topn_hit_rate"] = product_hit["topn_hits"] / product_hit["conversions"]

    return eval_long, eval_wide, cm_df, product_hit


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------


def create_visualizations(
    output_dir: Path,
    dataset_name: str,
    feature_metadata: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    defining_top: pd.DataFrame,
    product_rates_long: pd.DataFrame,
    signal_df: pd.DataFrame,
    top_recs: pd.DataFrame,
    eval_wide: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    clustered_customers: pd.DataFrame,
    sample_size: int,
    seed: int,
):
    viz_dir = output_dir / "visualizations"
    report_dir = output_dir / "report_assets"
    viz_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 01 Feature group weights.
    if not feature_metadata.empty:
        meta = feature_metadata.sort_values("group_weight")
        plt.figure(figsize=(8, 4))
        plt.barh(meta["feature_group"], meta["group_weight"])
        plt.xlabel("Feature group weight")
        plt.title(f"{dataset_name}: customer similarity feature weights")
        plt.tight_layout()
        plt.savefig(viz_dir / "01_feature_group_weights.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 02 Cluster sizes.
    if "customer_count" in cluster_profiles.columns:
        plt.figure(figsize=(8, 4))
        plt.bar(cluster_profiles["cluster_id"].astype(str), cluster_profiles["customer_count"])
        plt.xlabel("Cluster ID")
        plt.ylabel("Customers")
        plt.title(f"{dataset_name}: K-Means cluster sizes")
        plt.tight_layout()
        plt.savefig(viz_dir / "02_cluster_sizes.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 03 Product holding rates by cluster.
    if not product_rates_long.empty:
        pivot = product_rates_long.pivot(index="cluster_id", columns="product_code", values="cluster_product_holding_rate").reindex(columns=PRODUCTS)
        plt.figure(figsize=(8, 4))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(PRODUCTS)), PRODUCTS)
        plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
        plt.xlabel("Product")
        plt.ylabel("Cluster ID")
        plt.title(f"{dataset_name}: product holding rates by cluster")
        plt.colorbar(label="Holding rate")
        plt.tight_layout()
        plt.savefig(viz_dir / "03_cluster_product_holding_rates.png", dpi=150, bbox_inches="tight")
        plt.close()
        pivot.reset_index().to_csv(report_dir / "report_cluster_product_holding_rates_wide.csv", index=False)

    # 04 kNN score distribution.
    if not signal_df.empty and "cluster_first_knn_recommendation_score" in signal_df.columns:
        scores = signal_df[signal_df["eligible_for_recommendation"] == 1]["cluster_first_knn_recommendation_score"]
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=30)
        plt.xlabel("Cluster-first kNN recommendation score")
        plt.ylabel("Customer-product rows")
        plt.title(f"{dataset_name}: kNN score distribution")
        plt.tight_layout()
        plt.savefig(viz_dir / "04_cluster_first_knn_score_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 05 Recommendation mix.
    if not top_recs.empty:
        mix = top_recs["product_code"].value_counts().reindex(PRODUCTS).fillna(0)
        plt.figure(figsize=(8, 4))
        plt.bar(mix.index, mix.values)
        plt.xlabel("Product")
        plt.ylabel("Top-3 recommendation count")
        plt.title(f"{dataset_name}: recommendation mix")
        plt.tight_layout()
        plt.savefig(viz_dir / "05_recommendation_mix_cluster_first_knn.png", dpi=150, bbox_inches="tight")
        plt.close()
        mix.rename("recommendation_count").reset_index().rename(columns={"index": "product_code"}).to_csv(
            report_dir / "report_recommendation_mix.csv", index=False
        )

    # 06 Cumulative hit rates.
    if not eval_wide.empty and "cumulative_hit_rate_at_3" in eval_wide.columns:
        cols = [c for c in ["cumulative_hit_rate_at_1", "cumulative_hit_rate_at_2", "cumulative_hit_rate_at_3"] if c in eval_wide.columns]
        vals = [float(eval_wide[c].iloc[0]) for c in cols]
        labels = [c.replace("cumulative_hit_rate_at_", "Hit@") for c in cols]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, vals)
        plt.ylabel("Cumulative hit rate")
        plt.title(f"{dataset_name}: cumulative top-3 hit rates")
        plt.tight_layout()
        plt.savefig(viz_dir / "06_cumulative_hit_rates.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 07 PCA sample plot.
    feature_indexed = feature_matrix.set_index("Customer ID")
    n = min(sample_size, len(feature_indexed))
    if n >= 2 and feature_indexed.shape[1] >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_ids = feature_indexed.sample(n=n, random_state=seed).index
            coords = PCA(n_components=2, random_state=seed).fit_transform(feature_indexed.loc[sample_ids].values)
        cluster_lookup = clustered_customers.set_index("Customer ID")["cluster_id"]
        sample_clusters = cluster_lookup.loc[sample_ids]
        plt.figure(figsize=(7, 5))
        for cluster_id in sorted(sample_clusters.dropna().unique()):
            mask = sample_clusters.values == cluster_id
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.7, label=f"Cluster {cluster_id}")
        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.title(f"{dataset_name}: customer clusters in PCA view")
        plt.legend(fontsize=8, markerscale=1.5)
        plt.tight_layout()
        plt.savefig(viz_dir / "07_pca_customer_clusters.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 08 One exact-difference defining-feature chart per cluster.
    if not defining_top.empty:
        for cluster_id, group in defining_top.groupby("cluster_id"):
            plot_df = group.head(6).copy().sort_values("difference")
            plt.figure(figsize=(9, 4))
            plt.barh(plot_df["feature"].astype(str), plot_df["difference"])
            plt.axvline(0, linewidth=1)
            plt.xlabel("Exact difference from overall average/share")
            plt.title(f"{dataset_name}: Cluster {cluster_id} top defining features")
            plt.tight_layout()
            plt.savefig(viz_dir / f"08_cluster_{cluster_id}_top_defining_features_exact.png", dpi=150, bbox_inches="tight")
            plt.close()

    # 09 Age ranges by cluster.
    rows = []
    for cluster_id, group in clustered_customers.groupby("cluster_id"):
        row = {"cluster_id": cluster_id}
        for col in ["Age", "Monthly income proxy (sn)", "Risk profile score", "Days since last login (sn)"]:
            if col in group.columns:
                s = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_min"] = s.min()
                row[f"{col}_median"] = s.median()
                row[f"{col}_max"] = s.max()
        rows.append(row)
    range_df = pd.DataFrame(rows)
    range_df.to_csv(report_dir / "report_exact_ranges_by_cluster.csv", index=False)

    if "Age_min" in range_df.columns:
        plt.figure(figsize=(9, 4))
        plt.plot(range_df["cluster_id"].astype(str), range_df["Age_min"], marker="o", label="Min")
        plt.plot(range_df["cluster_id"].astype(str), range_df["Age_median"], marker="o", label="Median")
        plt.plot(range_df["cluster_id"].astype(str), range_df["Age_max"], marker="o", label="Max")
        plt.xlabel("Cluster ID")
        plt.ylabel("Age")
        plt.title(f"{dataset_name}: age range by cluster")
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "09_age_range_by_cluster.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 10 Rank-1 recommendation counts by cluster and product.
    if not top_recs.empty:
        rank1 = top_recs[top_recs["recommendation_rank"] == 1].copy()
        counts = rank1.groupby(["cluster_id", "product_code"]).size().unstack(fill_value=0).reindex(columns=PRODUCTS, fill_value=0)
        counts.to_csv(report_dir / "report_top1_recommendation_counts_by_cluster.csv")
        ax = counts.plot(kind="bar", stacked=True, figsize=(9, 5))
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Top-1 recommendation count")
        ax.set_title(f"{dataset_name}: top-1 recommendation counts by cluster and product")
        plt.tight_layout()
        plt.savefig(viz_dir / "10_top1_recommendation_counts_by_cluster.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    start_total = time.perf_counter()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or raw_dir.name
    conversion_path = Path(args.conversion_data) if args.conversion_data else raw_dir / "conversion_data_july_2026.csv"
    if not conversion_path.exists():
        conversion_path = None

    tech_steps = []

    def timed_step(step_name, fn):
        t0 = time.perf_counter()
        result = fn()
        seconds = time.perf_counter() - t0
        tech_steps.append({"dataset": dataset_name, "step": step_name, "seconds": seconds})
        print(f"{step_name}: {seconds:.2f} sec")
        return result

    customer_df = timed_step("load_customer_data", lambda: load_customer_data(raw_dir))
    customer_df.to_csv(output_dir / "cluster_first_model_input_table.csv", index=False)

    feature_matrix, feature_metadata, outlier_summary = timed_step(
        "build_customer_feature_matrix",
        lambda: build_customer_feature_matrix(
            customer_df=customer_df,
            max_onehot_cardinality=args.max_onehot_cardinality,
            outlier_method=args.outlier_method,
            lower_q=args.lower_quantile,
            upper_q=args.upper_quantile,
        ),
    )
    feature_matrix.to_csv(output_dir / "cluster_first_customer_feature_matrix.csv", index=False)
    feature_matrix.head(1000).to_csv(output_dir / "cluster_first_customer_feature_matrix_sample.csv", index=False)
    feature_metadata.to_csv(output_dir / "cluster_first_feature_metadata.csv", index=False)
    outlier_summary.to_csv(output_dir / "outlier_treatment_summary.csv", index=False)

    product_matrix = build_product_matrix(customer_df)
    product_matrix.reset_index().to_csv(output_dir / "cluster_first_product_holdings_matrix.csv", index=False)

    clustered_customers, cluster_centers = timed_step(
        "kmeans_clustering",
        lambda: create_kmeans_clusters(feature_matrix, customer_df, args.n_clusters, args.seed),
    )
    clustered_customers.to_csv(output_dir / "cluster_first_customer_clusters.csv", index=False)
    cluster_centers.to_csv(output_dir / "cluster_first_kmeans_cluster_centers.csv", index=False)

    cluster_profiles, exact_values, product_rates_long = timed_step(
        "cluster_profile_summary",
        lambda: summarize_clusters(clustered_customers),
    )
    cluster_profiles.to_csv(output_dir / "cluster_first_cluster_profiles.csv", index=False)
    exact_values.to_csv(output_dir / "cluster_exact_feature_values.csv", index=False)
    product_rates_long.to_csv(output_dir / "cluster_first_cluster_product_holding_rates_long.csv", index=False)

    defining_all, defining_top = timed_step(
        "cluster_defining_features",
        lambda: compute_defining_features(clustered_customers, top_k=10),
    )
    defining_all.to_csv(output_dir / "cluster_defining_features_all_exact_values.csv", index=False)
    defining_top.to_csv(output_dir / "cluster_defining_features_top10_exact_values.csv", index=False)

    signal_df, neighbor_df = timed_step(
        "cluster_first_knn",
        lambda: run_cluster_first_knn(
            feature_matrix=feature_matrix,
            product_matrix=product_matrix,
            clustered_customers=clustered_customers,
            n_neighbors=args.n_neighbors,
            reference_limit=args.knn_reference_limit,
            batch_size=args.knn_batch_size,
            seed=args.seed,
        ),
    )
    signal_df.to_csv(output_dir / "cluster_first_knn_signal_by_customer_product.csv", index=False)
    neighbor_df.to_csv(output_dir / "cluster_first_knn_neighbor_sample.csv", index=False)

    top_recs = timed_step("create_top_n_recommendations", lambda: create_top_n_recommendations(signal_df, args.top_n))
    top_recs.to_csv(output_dir / "final_top_3_recommendations_cluster_first_knn.csv", index=False)

    eval_long, eval_wide, cm_df, product_hit = timed_step(
        "evaluate_recommendations",
        lambda: evaluate_recommendations(top_recs, conversion_path, args.top_n),
    )
    eval_long.to_csv(output_dir / "cumulative_hit_rate_metrics.csv", index=False)
    eval_wide.to_csv(output_dir / "cluster_first_knn_accuracy_against_july_conversions.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix_rank1_actual_vs_predicted.csv", index=False)
    product_hit.to_csv(output_dir / "top3_hit_rate_by_actual_product.csv", index=False)

    timed_step(
        "create_visualizations",
        lambda: create_visualizations(
            output_dir=output_dir,
            dataset_name=dataset_name,
            feature_metadata=feature_metadata,
            cluster_profiles=cluster_profiles,
            defining_top=defining_top,
            product_rates_long=product_rates_long,
            signal_df=signal_df,
            top_recs=top_recs,
            eval_wide=eval_wide,
            feature_matrix=feature_matrix,
            clustered_customers=clustered_customers,
            sample_size=args.visualization_sample_size,
            seed=args.seed,
        ),
    )

    # Raw file row counts.
    raw_rows = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        try:
            raw_rows.append({
                "dataset": dataset_name,
                "file": csv_path.name,
                "rows": len(pd.read_csv(csv_path, usecols=[0])),
            })
        except Exception:
            raw_rows.append({"dataset": dataset_name, "file": csv_path.name, "rows": np.nan})
    pd.DataFrame(raw_rows).to_csv(output_dir / "raw_file_row_counts.csv", index=False)

    total_seconds = time.perf_counter() - start_total
    tech_specs = pd.DataFrame(tech_steps)
    tech_specs["total_seconds"] = total_seconds
    tech_specs["customers"] = customer_df["Customer ID"].nunique()
    tech_specs["feature_columns"] = feature_matrix.shape[1] - 1
    tech_specs["python_version"] = sys.version.split()[0]
    tech_specs["platform"] = platform.platform()
    tech_specs["outlier_method"] = args.outlier_method
    tech_specs["knn_reference_limit"] = args.knn_reference_limit
    tech_specs["knn_batch_size"] = args.knn_batch_size
    tech_specs.to_csv(output_dir / "tech_specs_runtime_by_step.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "customers": customer_df["Customer ID"].nunique(),
        "clusters": args.n_clusters,
        "neighbors": args.n_neighbors,
        "feature_columns": feature_matrix.shape[1] - 1,
        "outlier_method": args.outlier_method,
        "knn_reference_limit": args.knn_reference_limit,
        "knn_batch_size": args.knn_batch_size,
        "total_runtime_seconds": total_seconds,
        "recommendation_rows": len(top_recs),
        "has_conversion_data": conversion_path is not None,
        "approximate_knn_used_any_cluster": bool(signal_df["approximate_knn_used"].any()) if not signal_df.empty and "approximate_knn_used" in signal_df.columns else False,
    }
    if not eval_wide.empty:
        for col in eval_wide.columns:
            if col != "model":
                summary[col] = eval_wide[col].iloc[0]
    pd.DataFrame([summary]).to_csv(output_dir / "dataset_run_summary.csv", index=False)

    sanity = pd.DataFrame([
        {"check": "customers", "value": customer_df["Customer ID"].nunique()},
        {"check": "products", "value": len(PRODUCTS)},
        {"check": "clusters_requested", "value": args.n_clusters},
        {"check": "neighbors_requested", "value": args.n_neighbors},
        {"check": "feature_columns", "value": feature_matrix.shape[1] - 1},
        {"check": "recommendation_rows", "value": len(top_recs)},
        {"check": "already_held_positive_score_rows", "value": int(((signal_df["has_product"] == 1) & (signal_df["cluster_first_knn_recommendation_score"] > 0)).sum()) if not signal_df.empty else 0},
        {"check": "approximate_knn_used_any_cluster", "value": bool(signal_df["approximate_knn_used"].any()) if not signal_df.empty and "approximate_knn_used" in signal_df.columns else False},
    ])
    sanity.to_csv(output_dir / "cluster_first_knn_sanity_checks.csv", index=False)

    print("\nFinished:", dataset_name)
    print("Customers:", customer_df["Customer ID"].nunique())
    print("Total runtime seconds:", round(total_seconds, 2))
    print("Outputs:", output_dir)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
