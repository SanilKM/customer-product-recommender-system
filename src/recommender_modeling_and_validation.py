"""
Axis Bank Product Recommendation - Modeling + Proxy Evaluation

This is the next step after:
02_feature_engineering_and_baseline_recommender.py

Expected folder structure:

Coding main/
├── Python files/
│   └── 03_recommender_modeling_and_validation.py
├── axis_bank_recommender_csv/
└── axis_bank_recommender_outputs/
    ├── customer_level_features.csv
    ├── customer_product_propensity_long.csv
    ├── customer_product_holdings_long.csv
    ├── customer_product_transaction_features.csv
    ├── customer_product_digital_features.csv
    ├── top_3_recommendations_baseline.csv
    └── pipeline_sanity_checks.csv

Run from inside "Python files":
    python 03_recommender_modeling_and_validation.py

Or pass folders manually:
    python recommender_modeling_and_validation.py --input-dir ../data/02_features --output-dir ../data/03_model_outputs

Why this step exists:
- We do not have real conversion/campaign-response labels.
- So this script validates recommendation approaches using a proxy test:
  hide one product the customer already holds, then check whether the recommender ranks that product highly.
- This is not perfect, but it is a useful sanity check before real campaign data is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PRODUCTS = ["PL", "CC", "HL", "SA", "RD", "MF"]

DEFAULT_PRODUCT_NAMES = {
    "PL": "Personal Loan",
    "CC": "Credit Card",
    "HL": "Home Loan",
    "SA": "Savings Account",
    "RD": "Recurring Deposit",
    "MF": "Mutual Fund",
}


REQUIRED_FILES = [
    "customer_level_features.csv",
    "customer_product_propensity_long.csv",
    "customer_product_holdings_long.csv",
    "customer_product_transaction_features.csv",
    "customer_product_digital_features.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate next-step recommendation models."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Folder containing outputs from the previous feature-engineering step.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder where modeling outputs should be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of final recommendations per customer-month.",
    )
    return parser.parse_args()


def resolve_paths(input_dir_arg: str | None, output_dir_arg: str | None) -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent

    candidates = []
    if input_dir_arg:
        candidates.append(Path(input_dir_arg))

    candidates.extend(
        [
            script_dir.parent / "data" / "02_features",
            Path.cwd().parent / "data" / "02_features",
            Path.cwd() / "data" / "02_features",
            script_dir.parent / "axis_bank_recommender_outputs",
            script_dir / "axis_bank_recommender_outputs",
            Path.cwd().parent / "axis_bank_recommender_outputs",
            Path.cwd() / "axis_bank_recommender_outputs",
            Path.cwd(),
            script_dir.parent,
        ]
    )

    input_dir = None
    for candidate in candidates:
        if candidate.exists() and all((candidate / file).exists() for file in REQUIRED_FILES):
            input_dir = candidate.resolve()
            break

    if input_dir is None:
        raise FileNotFoundError(
            "Could not find the previous-step output files. "
            "Expected files include customer_level_features.csv, "
            "customer_product_propensity_long.csv, customer_product_holdings_long.csv, "
            "customer_product_transaction_features.csv, and customer_product_digital_features.csv. "
            "Pass --input-dir if needed."
        )

    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
    else:
        output_dir = script_dir.parent / "data" / "03_model_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def read_required(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_optional(input_dir: Path, filename: str) -> pd.DataFrame | None:
    path = input_dir / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def normalize_within_month(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df.groupby("Month period")[col].transform(normalize_series)


def get_product_names(top_baseline: pd.DataFrame | None) -> dict[str, str]:
    names = DEFAULT_PRODUCT_NAMES.copy()

    if top_baseline is not None and {"product_code", "Product_Name"}.issubset(top_baseline.columns):
        from_top = (
            top_baseline[["product_code", "Product_Name"]]
            .dropna()
            .drop_duplicates("product_code")
            .set_index("product_code")["Product_Name"]
            .to_dict()
        )
        names.update(from_top)

    return names


def build_model_ready(input_dir: Path) -> pd.DataFrame:
    customer = read_required(input_dir, "customer_level_features.csv")
    prop = read_required(input_dir, "customer_product_propensity_long.csv")
    hold = read_required(input_dir, "customer_product_holdings_long.csv")
    tx = read_required(input_dir, "customer_product_transaction_features.csv")
    digital = read_required(input_dir, "customer_product_digital_features.csv")
    top_baseline = read_optional(input_dir, "top_3_recommendations_baseline.csv")

    product_names = get_product_names(top_baseline)

    model = prop.merge(
        hold,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        tx,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        digital,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        customer,
        on=["Customer ID", "Month period"],
        how="left",
    )

    model["Product_Name"] = model["product_code"].map(product_names)

    numeric_cols = [
        "propensity_score",
        "product_count",
        "has_product",
        "product_txn_count",
        "product_txn_amount_sum",
        "product_txn_amount_avg",
        "product_credit_txn_count",
        "product_debit_txn_count",
        "product_salary_txn_count",
        "product_recurring_txn_count",
        "product_mobile_app_txn_count",
        "product_unique_channels",
        "product_unique_txn_categories",
        "click_count",
        "avg_funnel_depth",
        "max_funnel_depth",
        "total_time_on_page_seconds",
        "cta_click_count",
        "application_started_count",
        "booking_completed_count",
        "product_shown_before_count",
        "Age",
        "Risk profile score",
        "Monthly income proxy (sn)",
        "Days since last login (sn)",
        "Campaign exposure count 90D (sn)",
        "Products ignored count 90D (sn)",
        "Average ticket size",
        "Count",
        "Total credit amount (sn)",
        "Total debit amount (sn)",
        "UPI txn count (sn)",
        "MCC diversity count (sn)",
        "Days since last txn (sn)",
        "Avg monthly balance proxy (sn)",
        "Login count 30D (sn)",
        "Login count 90D (sn)",
    ]

    model = safe_numeric(model, numeric_cols)
    model["has_product"] = model["has_product"].fillna(0).astype(int)
    model["product_count"] = model["product_count"].fillna(0).astype(int)

    return model


def product_fit_score(row: pd.Series) -> float:
    """Simple business-rule fit score. This is explainable, not a trained model."""
    product = row.get("product_code")

    age = row.get("Age", 0) or 0
    income = row.get("Monthly income proxy (sn)", 0) or 0
    risk = row.get("Risk profile score", 0) or 0
    avg_balance = row.get("Avg monthly balance proxy (sn)", 0) or 0
    txn_count = row.get("Count", 0) or 0

    persona = str(row.get("Persona", "")).lower()
    occupation = str(row.get("Occupation", "")).lower()
    lifestage = str(row.get("Lifestage", "")).lower()
    tier = str(row.get("Tier Map", "")).lower()
    salary_flag = str(row.get("Salary credit flag (sn)", "")).upper()

    score = 0.30

    if product == "PL":
        score += 0.20 if 24 <= age <= 55 else 0
        score += 0.20 if salary_flag == "Y" or "sal" in occupation else 0
        score += 0.15 if income >= 50000 else 0
        score += 0.10 if txn_count >= 10 else 0

    elif product == "CC":
        score += 0.20 if 21 <= age <= 60 else 0
        score += 0.20 if income >= 40000 else 0
        score += 0.15 if txn_count >= 15 else 0
        score += 0.10 if "metro" in tier else 0

    elif product == "HL":
        score += 0.20 if 28 <= age <= 60 else 0
        score += 0.25 if income >= 100000 else 0
        score += 0.15 if "married" in lifestage else 0
        score += 0.10 if avg_balance >= 200000 else 0

    elif product == "SA":
        score += 0.30
        score += 0.10 if txn_count >= 3 else 0
        score += 0.10 if avg_balance >= 10000 else 0

    elif product == "RD":
        score += 0.20 if age >= 30 else 0
        score += 0.15 if "conservative" in persona or "planner" in persona else 0
        score += 0.15 if avg_balance >= 50000 else 0
        score += 0.10 if income >= 30000 else 0

    elif product == "MF":
        score += 0.20 if income >= 75000 else 0
        score += 0.20 if risk >= 650 else 0
        score += 0.15 if "invest" in persona or "affluent" in persona else 0
        score += 0.10 if 25 <= age <= 60 else 0

    return float(min(score, 1.0))


def add_signal_scores(model: pd.DataFrame) -> pd.DataFrame:
    model = model.copy()

    # Digital intent summarizes app/web product interest.
    model["digital_intent_score"] = (
        0.30 * normalize_within_month(model, "click_count")
        + 0.25 * normalize_within_month(model, "max_funnel_depth")
        + 0.20 * normalize_within_month(model, "total_time_on_page_seconds")
        + 0.15 * normalize_within_month(model, "cta_click_count")
        + 0.10 * normalize_within_month(model, "application_started_count")
    )

    # Transaction interest summarizes related movement in transaction data.
    model["transaction_interest_score"] = (
        0.45 * normalize_within_month(model, "product_txn_count")
        + 0.35 * normalize_within_month(model, "product_txn_amount_sum")
        + 0.20 * normalize_within_month(model, "product_unique_txn_categories")
    )

    model["profile_fit_score"] = model.apply(product_fit_score, axis=1)

    return model


def add_segment_affinity(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = model.copy()

    for col in ["persona_code", "income_band_code", "lifestage_code"]:
        if col not in model.columns:
            model[col] = "UNKNOWN"

    model["segment_key"] = (
        model[["persona_code", "income_band_code", "lifestage_code"]]
        .fillna("UNKNOWN")
        .astype(str)
        .agg("|".join, axis=1)
    )

    product_base_rates = model.groupby("product_code")["has_product"].mean().to_dict()

    segment_counts = (
        model.groupby(["segment_key", "product_code"], as_index=False)
        .agg(
            segment_product_holding_count=("has_product", "sum"),
            segment_product_customer_count=("Customer ID", "nunique"),
        )
    )

    model = model.merge(
        segment_counts,
        on=["segment_key", "product_code"],
        how="left",
    )

    # Leave-one-out segment affinity to reduce direct leakage during proxy validation.
    model["segment_affinity_score"] = np.where(
        model["segment_product_customer_count"] > 1,
        (model["segment_product_holding_count"] - model["has_product"])
        / (model["segment_product_customer_count"] - 1),
        model["product_code"].map(product_base_rates).fillna(0),
    )

    model["segment_affinity_score"] = (
        model["segment_affinity_score"]
        .clip(0, 1)
        .fillna(model["product_code"].map(product_base_rates))
        .fillna(0)
    )

    model["segment_affinity_score_norm"] = normalize_within_month(
        model, "segment_affinity_score"
    )

    segment_output = segment_counts.copy()
    segment_output["segment_product_holding_rate"] = (
        segment_output["segment_product_holding_count"]
        / segment_output["segment_product_customer_count"].replace(0, np.nan)
    ).fillna(0)

    return model, segment_output


def add_cooccurrence_scores(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = model.copy()

    holdings_matrix = model.pivot_table(
        index="Customer ID",
        columns="product_code",
        values="has_product",
        aggfunc="max",
        fill_value=0,
    )

    for product in PRODUCTS:
        if product not in holdings_matrix.columns:
            holdings_matrix[product] = 0

    holdings_matrix = holdings_matrix[PRODUCTS].astype(float)

    co_counts = holdings_matrix.T.dot(holdings_matrix)
    product_counts = holdings_matrix.sum(axis=0)

    # Conditional probability: P(candidate product held | context product held)
    conditional = co_counts.copy()
    for held_product in PRODUCTS:
        denom = product_counts[held_product]
        conditional[held_product] = conditional[held_product] / denom if denom > 0 else 0

    for product in PRODUCTS:
        conditional.loc[product, product] = 0

    base_rates = product_counts / len(holdings_matrix)
    lift = conditional.copy()
    for candidate_product in PRODUCTS:
        denom = base_rates[candidate_product] if base_rates[candidate_product] > 0 else 1
        lift.loc[candidate_product, :] = conditional.loc[candidate_product, :] / denom

    lift = lift.replace([np.inf, -np.inf], 0).fillna(0)

    # Customer-product co-occurrence score:
    # If customer holds products A and B, candidate C gets credit from P(C|A) and P(C|B).
    raw_scores = holdings_matrix.values @ conditional.T.values
    held_counts = holdings_matrix.sum(axis=1).replace(0, np.nan).values.reshape(-1, 1)
    cooccurrence_score_matrix = raw_scores / held_counts

    cooccurrence_score_df = (
        pd.DataFrame(
            cooccurrence_score_matrix,
            index=holdings_matrix.index,
            columns=PRODUCTS,
        )
        .reset_index()
        .melt(
            id_vars="Customer ID",
            var_name="product_code",
            value_name="cooccurrence_score",
        )
    )

    model = model.merge(
        cooccurrence_score_df,
        on=["Customer ID", "product_code"],
        how="left",
    )

    product_base_rates = model.groupby("product_code")["has_product"].mean().to_dict()
    no_context = model.groupby("Customer ID")["has_product"].transform("sum").eq(0)

    model["cooccurrence_score"] = model["cooccurrence_score"].fillna(0)
    model.loc[no_context, "cooccurrence_score"] = (
        model.loc[no_context, "product_code"].map(product_base_rates).fillna(0)
    )

    model["cooccurrence_score_norm"] = normalize_within_month(model, "cooccurrence_score")

    conditional_out = conditional.reset_index().rename(columns={"index": "candidate_product"})
    lift_out = lift.reset_index().rename(columns={"index": "candidate_product"})

    return model, conditional_out, lift_out


def add_model_scores(model: pd.DataFrame) -> pd.DataFrame:
    model = model.copy()

    model["baseline_digital_txn_score"] = (
        0.60 * model["propensity_score"].fillna(0)
        + 0.25 * model["digital_intent_score"].fillna(0)
        + 0.15 * model["transaction_interest_score"].fillna(0)
    )

    model["hybrid_raw_score"] = (
        0.40 * model["propensity_score"].fillna(0)
        + 0.20 * model["cooccurrence_score_norm"].fillna(0)
        + 0.15 * model["segment_affinity_score_norm"].fillna(0)
        + 0.10 * model["digital_intent_score"].fillna(0)
        + 0.10 * model["transaction_interest_score"].fillna(0)
        + 0.05 * model["profile_fit_score"].fillna(0)
    )

    model["eligible_for_recommendation"] = (model["has_product"] == 0).astype(int)

    # Final score excludes products already held by the customer.
    model["hybrid_recommendation_score"] = np.where(
        model["eligible_for_recommendation"] == 1,
        model["hybrid_raw_score"],
        0.0,
    )

    return model


def recommendation_reason(row: pd.Series) -> str:
    if row.get("has_product", 0) == 1:
        return "already held - excluded"

    reasons = []

    if row.get("propensity_score", 0) >= 0.60:
        reasons.append("high propensity")
    elif row.get("propensity_score", 0) >= 0.40:
        reasons.append("medium propensity")

    if row.get("cooccurrence_score_norm", 0) >= 0.50:
        reasons.append("similar to products already held")

    if row.get("segment_affinity_score_norm", 0) >= 0.50:
        reasons.append("strong segment fit")

    if row.get("digital_intent_score", 0) >= 0.25:
        reasons.append("recent digital product interest")

    if row.get("transaction_interest_score", 0) >= 0.20:
        reasons.append("related transaction activity")

    if row.get("profile_fit_score", 0) >= 0.65:
        reasons.append("customer profile fit")

    return "; ".join(reasons) if reasons else "low/neutral signal"


def build_top_n(model: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    candidates = model[model["eligible_for_recommendation"] == 1].copy()
    candidates["recommendation_reason"] = candidates.apply(recommendation_reason, axis=1)

    candidates = candidates.sort_values(
        ["Customer ID", "Month period", "hybrid_recommendation_score", "product_code"],
        ascending=[True, True, False, True],
    )

    candidates["recommendation_rank"] = (
        candidates.groupby(["Customer ID", "Month period"]).cumcount() + 1
    )

    top = candidates[candidates["recommendation_rank"] <= n].copy()

    cols = [
        "Customer ID",
        "Month period",
        "recommendation_rank",
        "product_code",
        "Product_Name",
        "hybrid_recommendation_score",
        "propensity_score",
        "cooccurrence_score_norm",
        "segment_affinity_score_norm",
        "digital_intent_score",
        "transaction_interest_score",
        "profile_fit_score",
        "recommendation_reason",
    ]

    return top[[col for col in cols if col in top.columns]]


def pivot_matrix(model: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = model.pivot_table(
        index=["Customer ID", "Month period"],
        columns="product_code",
        values=value_col,
        aggfunc="first",
        fill_value=0,
    ).reset_index()

    matrix.columns.name = None
    return matrix


def evaluate_with_masked_holdout(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proxy validation:
    1. Find customer-months with at least two held products.
    2. Hide one held product as the target product.
    3. Rank the hidden target against non-held products.
    4. Check whether the hidden product appears near the top.

    Important pandas compatibility note:
    This implementation avoids groupby.apply because newer pandas versions
    may drop grouping columns inside apply, which can cause:
    KeyError: "['Customer ID', 'Month period'] not in index"
    """
    required_cols = ["Customer ID", "Month period", "product_code", "has_product", "propensity_score"]
    missing_cols = [col for col in required_cols if col not in model.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for masked holdout evaluation: {missing_cols}")

    held = model.loc[
        model["has_product"] == 1,
        ["Customer ID", "Month period", "product_code", "propensity_score"],
    ].copy()

    if held.empty:
        empty_metrics = pd.DataFrame(
            columns=["model", "metric", "value", "eval_customer_months"]
        )
        empty_details = pd.DataFrame(
            columns=["Customer ID", "Month period", "target_product", "rank", "target_score", "model"]
        )
        return empty_metrics, empty_details

    held_counts = (
        held.groupby(["Customer ID", "Month period"], as_index=False)
        .agg(held_product_count=("product_code", "nunique"))
    )

    eval_keys = held_counts.loc[
        held_counts["held_product_count"] >= 2,
        ["Customer ID", "Month period"],
    ].copy()

    if eval_keys.empty:
        empty_metrics = pd.DataFrame(
            columns=["model", "metric", "value", "eval_customer_months"]
        )
        empty_details = pd.DataFrame(
            columns=["Customer ID", "Month period", "target_product", "rank", "target_score", "model"]
        )
        return empty_metrics, empty_details

    eval_held = held.merge(eval_keys, on=["Customer ID", "Month period"], how="inner")

    # Deterministic target selection without groupby.apply.
    # Pick one held product per customer-month by sorting and taking the first.
    # The hash creates a stable pseudo-random order while preserving grouping columns.
    stable_key = (
        eval_held["Customer ID"].astype(str)
        + "|"
        + eval_held["Month period"].astype(str)
        + "|"
        + eval_held["product_code"].astype(str)
    )
    eval_held["stable_random_order"] = pd.util.hash_pandas_object(stable_key, index=False).astype("uint64")

    target_rows = (
        eval_held.sort_values(
            ["Customer ID", "Month period", "stable_random_order"],
            ascending=[True, True, True],
        )
        .drop_duplicates(subset=["Customer ID", "Month period"], keep="first")
        [["Customer ID", "Month period", "product_code"]]
        .rename(columns={"product_code": "target_product"})
        .reset_index(drop=True)
    )

    eval_df = model.merge(target_rows, on=["Customer ID", "Month period"], how="inner")

    # Context held products are products the customer holds other than the hidden target.
    # We remove those from the candidate set because they should not be recommended.
    eval_df["context_held"] = (
        (eval_df["has_product"] == 1)
        & (eval_df["product_code"] != eval_df["target_product"])
    )

    eval_df = eval_df[~eval_df["context_held"]].copy()

    score_cols = {
        "propensity_only": "propensity_score",
        "baseline_digital_txn": "baseline_digital_txn_score",
        "segment_affinity": "segment_affinity_score_norm",
        "cooccurrence": "cooccurrence_score_norm",
        "hybrid": "hybrid_raw_score",
    }

    # Only evaluate score columns that exist. This makes the function robust
    # if you remove or rename one experimental scoring method later.
    score_cols = {name: col for name, col in score_cols.items() if col in eval_df.columns}

    metrics = []
    detail_frames = []

    for model_name, score_col in score_cols.items():
        tmp = eval_df[
            ["Customer ID", "Month period", "product_code", "target_product", score_col]
        ].copy()

        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").fillna(0)

        tmp = tmp.sort_values(
            ["Customer ID", "Month period", score_col, "product_code"],
            ascending=[True, True, False, True],
        )

        tmp["rank"] = tmp.groupby(["Customer ID", "Month period"]).cumcount() + 1

        target_rank = tmp[tmp["product_code"] == tmp["target_product"]][
            ["Customer ID", "Month period", "target_product", "rank", score_col]
        ].copy()

        target_rank = target_rank.rename(columns={score_col: "target_score"})
        target_rank["model"] = model_name

        n_eval = len(target_rank)
        if n_eval == 0:
            continue

        for k in [1, 2, 3, 5]:
            metrics.append(
                {
                    "model": model_name,
                    "metric": f"hit_rate_at_{k}",
                    "value": float((target_rank["rank"] <= k).mean()),
                    "eval_customer_months": n_eval,
                }
            )

        metrics.append(
            {
                "model": model_name,
                "metric": "mrr",
                "value": float((1 / target_rank["rank"]).mean()),
                "eval_customer_months": n_eval,
            }
        )

        metrics.append(
            {
                "model": model_name,
                "metric": "mean_target_rank",
                "value": float(target_rank["rank"].mean()),
                "eval_customer_months": n_eval,
            }
        )

        detail_frames.append(target_rank)

    metrics_df = pd.DataFrame(metrics)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    return metrics_df, details_df

def build_sanity_checks(model: pd.DataFrame, top_n: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    checks.append(
        {
            "check": "model_ready_rows",
            "value": len(model),
            "expected": "customers x months x 6 products",
            "status": "PASS" if len(model) > 0 else "FAIL",
        }
    )

    checks.append(
        {
            "check": "unique_customers",
            "value": model["Customer ID"].nunique(),
            "expected": "should match previous outputs",
            "status": "PASS" if model["Customer ID"].nunique() > 0 else "FAIL",
        }
    )

    observed_products = sorted(model["product_code"].dropna().unique())
    checks.append(
        {
            "check": "products",
            "value": ", ".join(observed_products),
            "expected": ", ".join(PRODUCTS),
            "status": "PASS" if observed_products == sorted(PRODUCTS) else "WARN",
        }
    )

    already_held_positive = int(
        ((model["has_product"] == 1) & (model["hybrid_recommendation_score"] > 0)).sum()
    )

    checks.append(
        {
            "check": "already_held_products_with_positive_final_score",
            "value": already_held_positive,
            "expected": "0",
            "status": "PASS" if already_held_positive == 0 else "FAIL",
        }
    )

    checks.append(
        {
            "check": "top_n_rows",
            "value": len(top_n),
            "expected": "up to customer-months x top_n",
            "status": "PASS" if len(top_n) > 0 else "FAIL",
        }
    )

    for col in [
        "propensity_score",
        "digital_intent_score",
        "transaction_interest_score",
        "cooccurrence_score_norm",
        "segment_affinity_score_norm",
        "hybrid_recommendation_score",
    ]:
        if col in model.columns:
            checks.append(
                {
                    "check": f"{col}_range",
                    "value": f"{model[col].min():.4f} to {model[col].max():.4f}",
                    "expected": "roughly 0 to 1",
                    "status": "PASS" if model[col].min() >= -0.001 and model[col].max() <= 1.001 else "WARN",
                }
            )

    if not metrics_df.empty:
        hybrid_hit3 = metrics_df[
            (metrics_df["model"] == "hybrid") & (metrics_df["metric"] == "hit_rate_at_3")
        ]["value"]

        if len(hybrid_hit3) > 0:
            checks.append(
                {
                    "check": "hybrid_proxy_hit_rate_at_3",
                    "value": round(float(hybrid_hit3.iloc[0]), 4),
                    "expected": "higher is better; proxy only",
                    "status": "PASS",
                }
            )

    return pd.DataFrame(checks)


def run_pipeline(input_dir: Path, output_dir: Path, top_n: int = 3) -> None:
    print(f"Reading previous outputs from: {input_dir}")
    print(f"Writing modeling outputs to: {output_dir}")

    print("Building model-ready table...")
    model = build_model_ready(input_dir)

    print("Adding signal scores...")
    model = add_signal_scores(model)

    print("Adding segment-affinity scores...")
    model, segment_output = add_segment_affinity(model)

    print("Adding product co-occurrence scores...")
    model, cooccurrence_conditional, cooccurrence_lift = add_cooccurrence_scores(model)

    print("Adding final hybrid scores...")
    model = add_model_scores(model)

    print("Creating final top recommendations...")
    top_recommendations = build_top_n(model, n=top_n)

    print("Running proxy validation with masked held-product evaluation...")
    metrics_df, details_df = evaluate_with_masked_holdout(model)

    print("Creating matrices and summaries...")
    hybrid_matrix = pivot_matrix(model, "hybrid_recommendation_score")
    propensity_matrix = pivot_matrix(model, "propensity_score")
    cooccurrence_matrix = pivot_matrix(model, "cooccurrence_score_norm")
    segment_matrix = pivot_matrix(model, "segment_affinity_score_norm")

    reason_summary = (
        top_recommendations["recommendation_reason"]
        .value_counts()
        .reset_index()
        .rename(columns={"recommendation_reason": "recommendation_reason", "count": "recommendation_count"})
    )

    # Some pandas versions name the count column differently after value_counts reset.
    if "count" in reason_summary.columns:
        reason_summary = reason_summary.rename(columns={"count": "recommendation_count"})
    if "index" in reason_summary.columns:
        reason_summary = reason_summary.rename(columns={"index": "recommendation_reason"})

    sanity_checks = build_sanity_checks(model, top_recommendations, metrics_df)

    print("Writing CSV outputs...")
    model.to_csv(output_dir / "model_ready_customer_product_features_scored.csv", index=False)
    top_recommendations.to_csv(output_dir / f"final_top_{top_n}_recommendations_hybrid.csv", index=False)
    metrics_df.to_csv(output_dir / "model_comparison_proxy_metrics.csv", index=False)
    details_df.to_csv(output_dir / "masked_product_evaluation_details.csv", index=False)
    hybrid_matrix.to_csv(output_dir / "matrix_hybrid_recommendation_score.csv", index=False)
    propensity_matrix.to_csv(output_dir / "matrix_propensity_score.csv", index=False)
    cooccurrence_matrix.to_csv(output_dir / "matrix_cooccurrence_score.csv", index=False)
    segment_matrix.to_csv(output_dir / "matrix_segment_affinity_score.csv", index=False)
    cooccurrence_conditional.to_csv(output_dir / "product_cooccurrence_conditional_probability_matrix.csv", index=False)
    cooccurrence_lift.to_csv(output_dir / "product_cooccurrence_lift_matrix.csv", index=False)
    segment_output.to_csv(output_dir / "segment_product_affinity.csv", index=False)
    reason_summary.to_csv(output_dir / "recommendation_reason_summary.csv", index=False)
    sanity_checks.to_csv(output_dir / "modeling_sanity_checks.csv", index=False)

    readme = f"""Axis Bank Modeling Outputs

Generated from:
{input_dir}

Main outputs:
1. model_ready_customer_product_features_scored.csv
   Full customer-product table with propensity, digital intent, transaction interest,
   co-occurrence, segment affinity, and final hybrid score.

2. final_top_{top_n}_recommendations_hybrid.csv
   Final top-{top_n} product recommendations per customer-month, excluding products already held.

3. model_comparison_proxy_metrics.csv
   Proxy evaluation results using masked held-product validation.
   This is useful because true conversion labels are not available.

4. product_cooccurrence_conditional_probability_matrix.csv
   Product-to-product conditional probability matrix.
   Example: P(candidate product held | another product already held).

5. segment_product_affinity.csv
   Product holding affinity by persona/income/lifestage segment.

6. matrix_hybrid_recommendation_score.csv
   Customer x product matrix for final hybrid recommendation scores.

Important note:
This is not final production ML. It is a strong next-step experiment:
- propensity-only baseline
- digital/transaction baseline
- segment affinity
- product co-occurrence
- hybrid recommender

The best real evaluation would use historical campaign/conversion labels when available.
"""
    (output_dir / "README_modeling_outputs.txt").write_text(readme, encoding="utf-8")

    print("\nDone. Key files created:")
    for filename in [
        "model_ready_customer_product_features_scored.csv",
        f"final_top_{top_n}_recommendations_hybrid.csv",
        "model_comparison_proxy_metrics.csv",
        "matrix_hybrid_recommendation_score.csv",
        "modeling_sanity_checks.csv",
    ]:
        print(f"- {output_dir / filename}")


def main() -> None:
    args = parse_args()
    input_dir, output_dir = resolve_paths(args.input_dir, args.output_dir)
    run_pipeline(input_dir=input_dir, output_dir=output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
