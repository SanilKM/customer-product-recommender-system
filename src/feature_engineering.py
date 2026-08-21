"""
Axis Bank Product Recommendation - Feature Engineering + Baseline Recommender

Folder structure expected:

Coding main/
├── Python files/
│   └── 02_feature_engineering_and_baseline_recommender.py
└── axis_bank_recommender_csv/
    ├── customer_demographics.csv
    ├── product_propensity.csv
    ├── product_holdings.csv
    ├── transaction_aggregates.csv
    ├── raw_transactions.csv
    ├── digital_login.csv
    ├── digital_clicks.csv
    ├── product_metadata.csv
    └── recommendation_output.csv

Run from inside the src folder:
    python feature_engineering.py

Or pass folders manually:
    python feature_engineering.py --csv-dir ../data/01_raw --output-dir ../data/02_features
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PRODUCTS = ["PL", "CC", "HL", "SA", "RD", "MF"]

PROPENSITY_COLS = {
    "PL": "PL_P",
    "CC": "CC_P",
    "HL": "HL_P",
    "SA": "SA_P",
    "RD": "RD_P",
    "MF": "MF_P",
}

HOLDING_COLS = {
    "PL": "PL_count",
    "CC": "CC_count",
    "HL": "HL_count",
    "SA": "SA_count",
    "RD": "RD_count",
    "MF": "MF_count",
}

TEXT_CODE_COLUMNS = [
    "Gender",
    "Constitution flag",
    "Occupation",
    "Occ desc",
    "City",
    "Tier Map",
    "Persona",
    "Lifestage",
    "Macro cluster (persona)",
    "Pref lang",
    "State",
    "Address type",
    "Cust health",
    "Marital status",
    "Income band (sn)",
    "Preferred channel (sn)",
    "Last product shown (sn)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model-ready recommendation tables from Axis synthetic CSVs."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Folder containing source CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder where output CSVs should be written.",
    )
    return parser.parse_args()


def resolve_paths(csv_dir_arg: str | None, output_dir_arg: str | None) -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent

    candidates = []
    if csv_dir_arg:
        candidates.append(Path(csv_dir_arg))

    candidates.extend(
        [
            script_dir.parent / "data" / "01_raw",
            Path.cwd().parent / "data" / "01_raw",
            Path.cwd() / "data" / "01_raw",
            script_dir / "axis_bank_recommender_csv",
            script_dir.parent / "axis_bank_recommender_csv",
            Path.cwd() / "axis_bank_recommender_csv",
            Path.cwd().parent / "axis_bank_recommender_csv",
        ]
    )

    csv_dir = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            csv_dir = candidate.resolve()
            break

    if csv_dir is None:
        raise FileNotFoundError(
            "Could not find the raw CSV folder. "
            "Expected data/01_raw or pass --csv-dir."
        )

    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
    else:
        output_dir = script_dir.parent / "data" / "02_features"

    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, output_dir


def read_csv(csv_dir: Path, filename: str) -> pd.DataFrame:
    path = csv_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def normalize_01(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to 0-1."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=series.index)
    return (s - mn) / (mx - mn)


def yn_to_int(series: pd.Series) -> pd.Series:
    return (
        series.fillna("N")
        .astype(str)
        .str.upper()
        .map({"Y": 1, "YES": 1, "TRUE": 1, "1": 1})
        .fillna(0)
        .astype(int)
    )


def create_lookup_codes(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Add code columns for string-heavy columns and export lookup files.

    Example:
    Persona -> persona_code
    City -> city_code
    Occupation -> occupation_code
    """
    coded = df.copy()
    lookup_dir = output_dir / "lookups"
    lookup_dir.mkdir(parents=True, exist_ok=True)

    for col in TEXT_CODE_COLUMNS:
        if col not in coded.columns:
            continue

        cleaned = (
            coded[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN"})
        )

        values = sorted(cleaned.unique())

        prefix = (
            col.lower()
            .replace("(sn)", "")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
            .strip("_")
        )

        code_col = f"{prefix}_code"
        mapping = {
            value: f"{prefix.upper()[:8]}_{idx + 1:03d}"
            for idx, value in enumerate(values)
        }

        coded[code_col] = cleaned.map(mapping)

        lookup = pd.DataFrame(
            {
                code_col: list(mapping.values()),
                col: list(mapping.keys()),
            }
        )
        lookup.to_csv(lookup_dir / f"lookup_{prefix}.csv", index=False)

    return coded


def build_propensity_long(product_propensity: pd.DataFrame) -> pd.DataFrame:
    prop = product_propensity[
        ["Customer ID", "Month period"] + list(PROPENSITY_COLS.values())
    ].copy()

    prop_long = prop.melt(
        id_vars=["Customer ID", "Month period"],
        value_vars=list(PROPENSITY_COLS.values()),
        var_name="propensity_column",
        value_name="propensity_score",
    )

    reverse_map = {v: k for k, v in PROPENSITY_COLS.items()}
    prop_long["product_code"] = prop_long["propensity_column"].map(reverse_map)

    return prop_long.drop(columns=["propensity_column"])


def build_holdings_long(product_holdings: pd.DataFrame) -> pd.DataFrame:
    hold = product_holdings.copy()

    if "Month period" not in hold.columns:
        hold["Month period"] = pd.to_datetime(
            hold["Year"].astype(str)
            + "-"
            + hold["Month"].astype(str).str.zfill(2)
            + "-01"
        ).dt.to_period("M").astype(str)

    hold_long = hold[
        ["Customer ID", "Month period"] + list(HOLDING_COLS.values())
    ].melt(
        id_vars=["Customer ID", "Month period"],
        value_vars=list(HOLDING_COLS.values()),
        var_name="holding_column",
        value_name="product_count",
    )

    reverse_map = {v: k for k, v in HOLDING_COLS.items()}
    hold_long["product_code"] = hold_long["holding_column"].map(reverse_map)
    hold_long = hold_long.drop(columns=["holding_column"])

    hold_long["product_count"] = (
        pd.to_numeric(hold_long["product_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    hold_long["has_product"] = (hold_long["product_count"] > 0).astype(int)

    return hold_long


def build_transaction_product_features(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    tx = raw_transactions.copy()

    tx["Txn Amount"] = pd.to_numeric(tx["Txn Amount"], errors="coerce").fillna(0)
    tx["product_code"] = tx["Product linked (sn)"].astype(str).str.strip()
    tx = tx[tx["product_code"].isin(PRODUCTS)].copy()

    tx["is_credit"] = (tx["Cr/De"].astype(str).str.upper() == "CR").astype(int)
    tx["is_debit"] = (tx["Cr/De"].astype(str).str.upper() == "DE").astype(int)
    tx["is_salary"] = yn_to_int(tx["Salary flag"])
    tx["is_recurring"] = yn_to_int(tx["Is recurring txn (sn)"])
    tx["is_mobile_app"] = (tx["Channel"].astype(str).str.upper() == "MOBILE_APP").astype(int)

    grouped = tx.groupby(["CustID", "Month period", "product_code"], as_index=False).agg(
        product_txn_count=("UID", "count"),
        product_txn_amount_sum=("Txn Amount", "sum"),
        product_txn_amount_avg=("Txn Amount", "mean"),
        product_credit_txn_count=("is_credit", "sum"),
        product_debit_txn_count=("is_debit", "sum"),
        product_salary_txn_count=("is_salary", "sum"),
        product_recurring_txn_count=("is_recurring", "sum"),
        product_mobile_app_txn_count=("is_mobile_app", "sum"),
        product_unique_channels=("Channel", "nunique"),
        product_unique_txn_categories=("Txn category derived (sn)", "nunique"),
    )

    grouped = grouped.rename(columns={"CustID": "Customer ID"})
    return grouped


def build_digital_product_features(digital_clicks: pd.DataFrame) -> pd.DataFrame:
    clicks = digital_clicks.copy()

    clicks["product_code"] = clicks["Product code (sn)"].astype(str).str.strip()
    clicks = clicks[clicks["product_code"].isin(PRODUCTS)].copy()

    yn_cols = [
        "CTA clicked flag (sn)",
        "Application started flag (sn)",
        "Booking completed flag (sn)",
        "Product shown before flag (sn)",
    ]

    for col in yn_cols:
        clicks[col + "_int"] = yn_to_int(clicks[col])

    clicks["Funnel depth (sn)"] = pd.to_numeric(
        clicks["Funnel depth (sn)"], errors="coerce"
    ).fillna(0)

    clicks["Time on page seconds (sn)"] = pd.to_numeric(
        clicks["Time on page seconds (sn)"], errors="coerce"
    ).fillna(0)

    grouped = clicks.groupby(["CustID", "Month period", "product_code"], as_index=False).agg(
        click_count=("ClickID", "count"),
        avg_funnel_depth=("Funnel depth (sn)", "mean"),
        max_funnel_depth=("Funnel depth (sn)", "max"),
        total_time_on_page_seconds=("Time on page seconds (sn)", "sum"),
        cta_click_count=("CTA clicked flag (sn)_int", "sum"),
        application_started_count=("Application started flag (sn)_int", "sum"),
        booking_completed_count=("Booking completed flag (sn)_int", "sum"),
        product_shown_before_count=("Product shown before flag (sn)_int", "sum"),
    )

    grouped = grouped.rename(columns={"CustID": "Customer ID"})
    return grouped


def prepare_customer_features(
    customer_demographics: pd.DataFrame,
    transaction_aggregates: pd.DataFrame,
    digital_login: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    demo_coded = create_lookup_codes(customer_demographics, output_dir)

    demo_cols = [
        "Customer ID",
        "Month period",
        "Age",
        "Risk profile score",
        "Monthly income proxy (sn)",
        "Days since last login (sn)",
        "Campaign exposure count 90D (sn)",
        "Products ignored count 90D (sn)",
        "Total active products (sn)",
    ]

    useful_text_cols = [
        "Gender",
        "Occupation",
        "Tier Map",
        "Persona",
        "Lifestage",
        "Income band (sn)",
        "Preferred channel (sn)",
    ]

    code_cols = [col for col in demo_coded.columns if col.endswith("_code")]
    final_demo_cols = [
        col for col in demo_cols + useful_text_cols + code_cols
        if col in demo_coded.columns
    ]

    customer = demo_coded[final_demo_cols].copy()
    customer = customer.drop_duplicates(subset=["Customer ID", "Month period"])

    txn = transaction_aggregates.copy().rename(columns={"CustID": "Customer ID"})

    txn_cols = [
        "Customer ID",
        "Month period",
        "Average ticket size",
        "Count",
        "Total credit amount (sn)",
        "Total debit amount (sn)",
        "UPI txn count (sn)",
        "Salary credit flag (sn)",
        "MCC diversity count (sn)",
        "High value txn flag (sn)",
        "Most common channel (sn)",
        "Days since last txn (sn)",
        "Avg monthly balance proxy (sn)",
    ]
    txn_cols = [col for col in txn_cols if col in txn.columns]

    customer = customer.merge(
        txn[txn_cols],
        on=["Customer ID", "Month period"],
        how="left",
    )

    login = digital_login.copy().rename(columns={"CustID": "Customer ID"})

    if "Month period" not in login.columns:
        login["Month period"] = (
            pd.to_datetime(login["Date of login"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )

    login_cols = [
        "Customer ID",
        "Month period",
        "Login count 30D (sn)",
        "Login count 90D (sn)",
        "Last app version (sn)",
        "Preferred login hour (sn)",
        "Device OS (sn)",
    ]
    login_cols = [col for col in login_cols if col in login.columns]

    login_small = login[login_cols].drop_duplicates(
        subset=["Customer ID", "Month period"]
    )

    customer = customer.merge(
        login_small,
        on=["Customer ID", "Month period"],
        how="left",
    )

    return customer


def product_fit_score(row: pd.Series) -> float:
    """
    Simple explainable product-fit score.

    This is not a trained ML model. It is a baseline rules/scoring layer
    that combines business intuition with synthetic data signals.
    """
    product = row.get("product_code")

    age = row.get("Age", 0) or 0
    income = row.get("Monthly income proxy (sn)", 0) or 0
    risk = row.get("Risk profile score", 0) or 0
    persona = str(row.get("Persona", "")).lower()
    occupation = str(row.get("Occupation", "")).lower()
    lifestage = str(row.get("Lifestage", "")).lower()
    tier = str(row.get("Tier Map", "")).lower()
    salary_flag = str(row.get("Salary credit flag (sn)", "N")).upper()
    txn_count = row.get("Count", 0) or 0
    avg_balance = row.get("Avg monthly balance proxy (sn)", 0) or 0

    score = 0.30

    if product == "PL":
        score += 0.20 if 24 <= age <= 55 else 0.00
        score += 0.20 if salary_flag == "Y" or "sal" in occupation else 0.00
        score += 0.15 if income >= 50000 else 0.00
        score += 0.15 if txn_count >= 10 else 0.00

    elif product == "CC":
        score += 0.20 if 21 <= age <= 60 else 0.00
        score += 0.20 if income >= 40000 else 0.00
        score += 0.15 if txn_count >= 15 else 0.00
        score += 0.10 if "metro" in tier else 0.00

    elif product == "HL":
        score += 0.20 if 28 <= age <= 60 else 0.00
        score += 0.25 if income >= 100000 else 0.00
        score += 0.15 if "married" in lifestage else 0.00
        score += 0.10 if avg_balance >= 200000 else 0.00

    elif product == "SA":
        score += 0.30
        score += 0.15 if txn_count >= 3 else 0.00
        score += 0.10 if avg_balance >= 10000 else 0.00

    elif product == "RD":
        score += 0.20 if age >= 30 else 0.00
        score += 0.15 if "conservative" in persona or "planner" in persona else 0.00
        score += 0.15 if avg_balance >= 50000 else 0.00
        score += 0.10 if income >= 30000 else 0.00

    elif product == "MF":
        score += 0.20 if income >= 75000 else 0.00
        score += 0.20 if risk >= 650 else 0.00
        score += 0.15 if "invest" in persona or "affluent" in persona else 0.00
        score += 0.10 if 25 <= age <= 60 else 0.00

    return float(min(score, 1.0))


def add_scores_and_reasons(model: pd.DataFrame) -> pd.DataFrame:
    model = model.copy()

    numeric_fill_cols = [
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
        "Count",
        "Average ticket size",
        "Total credit amount (sn)",
        "Total debit amount (sn)",
        "UPI txn count (sn)",
        "MCC diversity count (sn)",
        "Days since last txn (sn)",
        "Avg monthly balance proxy (sn)",
        "Login count 30D (sn)",
        "Login count 90D (sn)",
        "Campaign exposure count 90D (sn)",
        "Products ignored count 90D (sn)",
    ]

    for col in numeric_fill_cols:
        if col in model.columns:
            model[col] = pd.to_numeric(model[col], errors="coerce").fillna(0)

    model["digital_intent_score"] = (
        0.30 * normalize_01(model.get("click_count", pd.Series(0, index=model.index)))
        + 0.25 * normalize_01(model.get("max_funnel_depth", pd.Series(0, index=model.index)))
        + 0.20 * normalize_01(model.get("total_time_on_page_seconds", pd.Series(0, index=model.index)))
        + 0.15 * normalize_01(model.get("cta_click_count", pd.Series(0, index=model.index)))
        + 0.10 * normalize_01(model.get("application_started_count", pd.Series(0, index=model.index)))
    )

    model["transaction_interest_score"] = (
        0.45 * normalize_01(model.get("product_txn_count", pd.Series(0, index=model.index)))
        + 0.35 * normalize_01(model.get("product_txn_amount_sum", pd.Series(0, index=model.index)))
        + 0.20 * normalize_01(model.get("product_unique_txn_categories", pd.Series(0, index=model.index)))
    )

    model["customer_activity_score"] = (
        0.50 * normalize_01(model.get("Login count 30D (sn)", pd.Series(0, index=model.index)))
        + 0.30 * normalize_01(model.get("Count", pd.Series(0, index=model.index)))
        + 0.20 * normalize_01(model.get("Campaign exposure count 90D (sn)", pd.Series(0, index=model.index)))
    )

    model["demographic_fit_score"] = model.apply(product_fit_score, axis=1)

    model["raw_recommendation_score"] = (
        0.50 * model["propensity_score"].fillna(0)
        + 0.20 * model["digital_intent_score"]
        + 0.15 * model["transaction_interest_score"]
        + 0.10 * model["demographic_fit_score"]
        + 0.05 * model["customer_activity_score"]
    )

    # Already-held products should not be recommended again in this baseline.
    model["eligible_for_recommendation"] = (model["has_product"] == 0).astype(int)

    model["recommendation_score"] = np.where(
        model["eligible_for_recommendation"] == 1,
        model["raw_recommendation_score"],
        0.0,
    )

    def reason(row: pd.Series) -> str:
        reasons = []

        if row["has_product"] == 1:
            return "already held - excluded from recommendation ranking"

        if row["propensity_score"] >= 0.65:
            reasons.append("high propensity")
        elif row["propensity_score"] >= 0.45:
            reasons.append("medium propensity")

        if row["digital_intent_score"] >= 0.25:
            reasons.append("recent digital intent")

        if row["transaction_interest_score"] >= 0.20:
            reasons.append("related transaction activity")

        if row["demographic_fit_score"] >= 0.65:
            reasons.append("customer profile fit")

        if row["customer_activity_score"] >= 0.40:
            reasons.append("active customer")

        return "; ".join(reasons) if reasons else "low/neutral signal"

    model["recommendation_reason"] = model.apply(reason, axis=1)

    return model


def build_top_n(model: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    candidates = model[model["eligible_for_recommendation"] == 1].copy()

    candidates = candidates.sort_values(
        ["Customer ID", "Month period", "recommendation_score"],
        ascending=[True, True, False],
    )

    candidates["recommendation_rank"] = (
        candidates.groupby(["Customer ID", "Month period"]).cumcount() + 1
    )

    top = candidates[candidates["recommendation_rank"] <= n].copy()

    return top[
        [
            "Customer ID",
            "Month period",
            "recommendation_rank",
            "product_code",
            "Product_Name",
            "recommendation_score",
            "propensity_score",
            "digital_intent_score",
            "transaction_interest_score",
            "demographic_fit_score",
            "recommendation_reason",
        ]
    ]


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


def run_pipeline(csv_dir: Path, output_dir: Path) -> None:
    print(f"Reading source CSVs from: {csv_dir}")
    print(f"Writing outputs to: {output_dir}")

    customer_demographics = read_csv(csv_dir, "customer_demographics.csv")
    product_propensity = read_csv(csv_dir, "product_propensity.csv")
    product_holdings = read_csv(csv_dir, "product_holdings.csv")
    transaction_aggregates = read_csv(csv_dir, "transaction_aggregates.csv")
    raw_transactions = read_csv(csv_dir, "raw_transactions.csv")
    digital_login = read_csv(csv_dir, "digital_login.csv")
    digital_clicks = read_csv(csv_dir, "digital_clicks.csv")
    product_metadata = read_csv(csv_dir, "product_metadata.csv")

    print("Building long customer-product propensity table...")
    prop_long = build_propensity_long(product_propensity)

    print("Building long customer-product holdings table...")
    hold_long = build_holdings_long(product_holdings)

    print("Aggregating product-level transaction signals...")
    tx_product = build_transaction_product_features(raw_transactions)

    print("Aggregating product-level digital click signals...")
    digital_product = build_digital_product_features(digital_clicks)

    print("Preparing customer-level features and lookup codes...")
    customer_features = prepare_customer_features(
        customer_demographics=customer_demographics,
        transaction_aggregates=transaction_aggregates,
        digital_login=digital_login,
        output_dir=output_dir,
    )

    print("Joining into model-ready customer-product table...")
    model = prop_long.merge(
        hold_long,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        tx_product,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        digital_product,
        on=["Customer ID", "Month period", "product_code"],
        how="left",
    )

    model = model.merge(
        customer_features,
        on=["Customer ID", "Month period"],
        how="left",
    )

    product_metadata_small = product_metadata.rename(columns={"Product_Code": "product_code"})

    metadata_cols = [
        "product_code",
        "Product_Name",
        "Product_Family",
        "Product_Category",
        "Digital_Journey_Available",
        "Typical_Customer_Segment",
    ]
    metadata_cols = [col for col in metadata_cols if col in product_metadata_small.columns]

    model = model.merge(
        product_metadata_small[metadata_cols],
        on="product_code",
        how="left",
    )

    print("Adding baseline recommendation scores and reasons...")
    model = add_scores_and_reasons(model)

    print("Creating output matrices and top recommendations...")
    top_3 = build_top_n(model, n=3)
    matrix_propensity = pivot_matrix(model, "propensity_score")
    matrix_holdings = pivot_matrix(model, "has_product")
    matrix_recommendation = pivot_matrix(model, "recommendation_score")

    print("Writing CSV outputs...")

    customer_features.to_csv(output_dir / "customer_level_features.csv", index=False)
    prop_long.to_csv(output_dir / "customer_product_propensity_long.csv", index=False)
    hold_long.to_csv(output_dir / "customer_product_holdings_long.csv", index=False)
    tx_product.to_csv(output_dir / "customer_product_transaction_features.csv", index=False)
    digital_product.to_csv(output_dir / "customer_product_digital_features.csv", index=False)
    model.to_csv(output_dir / "model_ready_customer_product_features.csv", index=False)
    top_3.to_csv(output_dir / "top_3_recommendations_baseline.csv", index=False)
    matrix_propensity.to_csv(output_dir / "matrix_propensity.csv", index=False)
    matrix_holdings.to_csv(output_dir / "matrix_holdings.csv", index=False)
    matrix_recommendation.to_csv(output_dir / "matrix_recommendation_score.csv", index=False)

    checks = [
        {
            "check": "model_ready_rows",
            "value": len(model),
            "expected": "customers x months x 6 products",
        },
        {
            "check": "unique_customers",
            "value": model["Customer ID"].nunique(),
            "expected": "should match customer_demographics",
        },
        {
            "check": "products",
            "value": ", ".join(sorted(model["product_code"].dropna().unique())),
            "expected": "PL, CC, HL, SA, RD, MF",
        },
        {
            "check": "already_held_recommendation_rows",
            "value": int(
                ((model["has_product"] == 1) & (model["recommendation_score"] > 0)).sum()
            ),
            "expected": "0",
        },
        {
            "check": "top_3_rows",
            "value": len(top_3),
            "expected": "up to customers x months x 3",
        },
        {
            "check": "propensity_score_min",
            "value": round(float(model["propensity_score"].min()), 4),
            "expected": ">= 0",
        },
        {
            "check": "propensity_score_max",
            "value": round(float(model["propensity_score"].max()), 4),
            "expected": "<= 1",
        },
        {
            "check": "recommendation_score_min",
            "value": round(float(model["recommendation_score"].min()), 4),
            "expected": ">= 0",
        },
        {
            "check": "recommendation_score_max",
            "value": round(float(model["recommendation_score"].max()), 4),
            "expected": "<= 1-ish, depending on weighted components",
        },
    ]

    pd.DataFrame(checks).to_csv(output_dir / "pipeline_sanity_checks.csv", index=False)

    print("\nDone. Created these key files:")
    for file in [
        "model_ready_customer_product_features.csv",
        "top_3_recommendations_baseline.csv",
        "matrix_propensity.csv",
        "matrix_holdings.csv",
        "matrix_recommendation_score.csv",
        "pipeline_sanity_checks.csv",
    ]:
        print(f"- {output_dir / file}")


def main() -> None:
    args = parse_args()
    csv_dir, output_dir = resolve_paths(args.csv_dir, args.output_dir)
    run_pipeline(csv_dir, output_dir)


if __name__ == "__main__":
    main()
