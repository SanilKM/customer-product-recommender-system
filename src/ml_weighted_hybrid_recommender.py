"""
04 - ML-Weighted Hybrid Recommender

Purpose
-------
This script replaces the earlier heuristic-weight hybrid scoring step for the
conversion/ML stage.

It uses:
- June 2026 customer-product signal table from data/03_model_outputs
- July 2026 conversion labels from data/01_raw/conversion_data_july_2026.csv

Then it:
1. Creates conversion labels at customer-product level.
2. Splits customers into train/test sets.
3. Trains an ML model to predict next-month conversion.
4. Learns data-driven weights for each recommender signal.
5. Builds a new ML-weighted hybrid recommendation score.
6. Compares the ML-weighted hybrid against the previous heuristic hybrid and propensity-only baselines.

Run from the main Coding folder:

python src/ml_weighted_hybrid_recommender.py \
    --model-input-dir data/03_model_outputs \
    --conversion-data data/01_raw/conversion_data_july_2026.csv \
    --output-dir data/04_ml_weighted_hybrid_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRODUCTS = ["PL", "CC", "HL", "SA", "RD", "MF"]

PRODUCT_NAME_MAP = {
    "PL": "Personal Loan",
    "CC": "Credit Card",
    "HL": "Home Loan",
    "SA": "Savings Account",
    "RD": "Recurring Deposit",
    "MF": "Mutual Fund",
}

SIGNAL_COLS = [
    "propensity_score",
    "digital_intent_score",
    "transaction_interest_score",
    "profile_fit_score",
    "segment_affinity_score_norm",
    "cooccurrence_score_norm",
]

BASELINE_SCORE_COLS = [
    "propensity_score",
    "hybrid_recommendation_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML-weighted hybrid recommender.")
    parser.add_argument(
        "--model-input-dir",
        type=str,
        default=None,
        help="Folder containing model_ready_customer_product_features_scored.csv from stage 03.",
    )
    parser.add_argument(
        "--conversion-data",
        type=str,
        default=None,
        help="Path to conversion_data_july_2026.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder to write ML-weighted hybrid outputs.",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    model_candidates = []
    if args.model_input_dir:
        model_candidates.append(Path(args.model_input_dir))
    model_candidates.extend([
        cwd / "data" / "03_model_outputs",
        cwd.parent / "data" / "03_model_outputs",
        script_dir.parent / "data" / "03_model_outputs",
    ])

    model_input_dir = next((p.resolve() for p in model_candidates if p.exists()), None)
    if model_input_dir is None:
        raise FileNotFoundError(
            "Could not find data/03_model_outputs. Run stage 03 first or pass --model-input-dir."
        )

    conversion_candidates = []
    if args.conversion_data:
        conversion_candidates.append(Path(args.conversion_data))
    conversion_candidates.extend([
        cwd / "data" / "01_raw" / "conversion_data_july_2026.csv",
        cwd.parent / "data" / "01_raw" / "conversion_data_july_2026.csv",
        script_dir.parent / "data" / "01_raw" / "conversion_data_july_2026.csv",
    ])

    conversion_path = next((p.resolve() for p in conversion_candidates if p.exists()), None)
    if conversion_path is None:
        raise FileNotFoundError(
            "Could not find conversion_data_july_2026.csv. Pass --conversion-data."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = model_input_dir.parent / "04_ml_weighted_hybrid_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    return model_input_dir, conversion_path, output_dir


def load_inputs(model_input_dir: Path, conversion_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = model_input_dir / "model_ready_customer_product_features_scored.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing required model input: {model_path}")

    model = pd.read_csv(model_path)
    conversions = pd.read_csv(conversion_path)

    return model, conversions


def clean_inputs(model: pd.DataFrame, conversions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = model.copy()
    conversions = conversions.copy()

    if "Product_Name" not in model.columns:
        model["Product_Name"] = model["product_code"].map(PRODUCT_NAME_MAP)

    for col in SIGNAL_COLS + BASELINE_SCORE_COLS:
        if col not in model.columns:
            model[col] = 0.0
        model[col] = pd.to_numeric(model[col], errors="coerce").fillna(0.0)

    model["has_product"] = pd.to_numeric(model["has_product"], errors="coerce").fillna(0).astype(int)

    if "Product converted" not in conversions.columns:
        possible = [c for c in conversions.columns if c.lower().strip() in ["product", "product_code", "converted_product"]]
        if possible:
            conversions = conversions.rename(columns={possible[0]: "Product converted"})
        else:
            raise ValueError("Conversion data must have a 'Product converted' column.")

    if "Customer ID" not in conversions.columns:
        raise ValueError("Conversion data must have a 'Customer ID' column.")

    if "Date" not in conversions.columns:
        raise ValueError("Conversion data must have a 'Date' column.")

    conversions["Product converted"] = conversions["Product converted"].astype(str).str.strip()
    conversions = conversions[conversions["Product converted"].isin(PRODUCTS)].copy()

    return model, conversions


def add_conversion_labels(model: pd.DataFrame, conversions: pd.DataFrame) -> pd.DataFrame:
    """
    Label June customer-product rows using July conversion data.

    converted_next_month = 1 when Customer ID + product_code matches
    a July converted product.
    """
    labels = model.copy()

    conversion_keys = (
        conversions[["Customer ID", "Product converted"]]
        .drop_duplicates()
        .rename(columns={"Product converted": "product_code"})
    )
    conversion_keys["converted_next_month"] = 1

    labels = labels.merge(
        conversion_keys,
        on=["Customer ID", "product_code"],
        how="left",
    )
    labels["converted_next_month"] = labels["converted_next_month"].fillna(0).astype(int)

    return labels


def split_by_customer(
    labels: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by customer, not by row.

    This avoids leakage where the same customer appears in both train and test
    with different product rows.
    """
    customer_labels = labels.groupby("Customer ID", as_index=False).agg(
        customer_converted=("converted_next_month", "max")
    )

    stratify = None
    if customer_labels["customer_converted"].nunique() == 2:
        vc = customer_labels["customer_converted"].value_counts()
        if vc.min() >= 2:
            stratify = customer_labels["customer_converted"]

    train_customers, test_customers = train_test_split(
        customer_labels["Customer ID"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_customers = set(train_customers)
    test_customers = set(test_customers)

    train_df = labels[labels["Customer ID"].isin(train_customers)].copy()
    test_df = labels[labels["Customer ID"].isin(test_customers)].copy()

    split_summary = pd.DataFrame([
        {
            "split": "train",
            "customers": train_df["Customer ID"].nunique(),
            "rows": len(train_df),
            "candidate_rows_has_product_0": int((train_df["has_product"] == 0).sum()),
            "positive_rows": int(train_df["converted_next_month"].sum()),
            "positive_rate_candidate_rows": train_df.loc[train_df["has_product"] == 0, "converted_next_month"].mean(),
        },
        {
            "split": "test",
            "customers": test_df["Customer ID"].nunique(),
            "rows": len(test_df),
            "candidate_rows_has_product_0": int((test_df["has_product"] == 0).sum()),
            "positive_rows": int(test_df["converted_next_month"].sum()),
            "positive_rate_candidate_rows": test_df.loc[test_df["has_product"] == 0, "converted_next_month"].mean(),
        },
    ])

    return train_df, test_df, split_summary


def train_logistic_weight_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Train logistic regression on eligible product candidates.

    The coefficients are used as ML-learned signal weights.
    """
    train_candidates = train_df[train_df["has_product"] == 0].copy()
    test_candidates = test_df[test_df["has_product"] == 0].copy()

    X_train = train_candidates[SIGNAL_COLS]
    y_train = train_candidates["converted_next_month"].astype(int)

    X_test = test_candidates[SIGNAL_COLS]
    y_test = test_candidates["converted_next_month"].astype(int)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    clf.fit(X_train, y_train)

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "model": "ML-weighted hybrid logistic regression",
        "train_candidate_rows": len(X_train),
        "test_candidate_rows": len(X_test),
        "train_positive_rows": int(y_train.sum()),
        "test_positive_rows": int(y_test.sum()),
        "test_positive_rate": float(y_test.mean()),
        "accuracy_threshold_0_5": accuracy_score(y_test, test_pred),
        "precision_threshold_0_5": precision_score(y_test, test_pred, zero_division=0),
        "recall_threshold_0_5": recall_score(y_test, test_pred, zero_division=0),
        "f1_threshold_0_5": f1_score(y_test, test_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_test, test_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = average_precision_score(y_test, test_prob)
    except ValueError:
        metrics["average_precision"] = np.nan

    try:
        metrics["log_loss"] = log_loss(y_test, test_prob)
    except ValueError:
        metrics["log_loss"] = np.nan

    metrics_df = pd.DataFrame([metrics])

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    confusion_df = pd.DataFrame(
        cm,
        index=["actual_0_no_conversion", "actual_1_conversion"],
        columns=["predicted_0_no_conversion", "predicted_1_conversion"],
    ).reset_index().rename(columns={"index": "actual"})

    return clf, metrics_df, confusion_df


def extract_learned_weights(clf: Pipeline) -> pd.DataFrame:
    model = clf.named_steps["model"]

    weights = pd.DataFrame({
        "signal": SIGNAL_COLS,
        "logistic_coefficient": model.coef_[0],
    })

    weights["absolute_coefficient"] = weights["logistic_coefficient"].abs()
    abs_total = weights["absolute_coefficient"].sum()
    weights["abs_normalized_weight"] = np.where(
        abs_total > 0,
        weights["absolute_coefficient"] / abs_total,
        0,
    )

    # For a recommender score, positive coefficients are more intuitive.
    # Negative coefficients are treated as 0 in the weighted hybrid score.
    weights["positive_coefficient"] = weights["logistic_coefficient"].clip(lower=0)
    pos_total = weights["positive_coefficient"].sum()

    if pos_total > 0:
        weights["ml_recommender_weight"] = weights["positive_coefficient"] / pos_total
        weighting_method = "positive_coefficients_only"
    else:
        weights["ml_recommender_weight"] = weights["abs_normalized_weight"]
        weighting_method = "absolute_coefficients_fallback"

    weights["weighting_method"] = weighting_method
    weights["interpretation"] = np.where(
        weights["logistic_coefficient"] >= 0,
        "higher signal increased conversion likelihood in training",
        "higher signal reduced conversion likelihood in training; set to 0 for final recommender weight",
    )

    return weights.sort_values("ml_recommender_weight", ascending=False).reset_index(drop=True)


def apply_ml_weights(
    labels: pd.DataFrame,
    clf: Pipeline,
    learned_weights: pd.DataFrame,
) -> pd.DataFrame:
    scored = labels.copy()

    for col in SIGNAL_COLS:
        scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(0.0)

    scored["ml_conversion_probability"] = clf.predict_proba(scored[SIGNAL_COLS])[:, 1]

    weight_map = dict(zip(learned_weights["signal"], learned_weights["ml_recommender_weight"]))

    scored["ml_weighted_raw_score"] = 0.0
    for col in SIGNAL_COLS:
        scored["ml_weighted_raw_score"] += scored[col] * float(weight_map.get(col, 0.0))

    scored["eligible_for_recommendation"] = (scored["has_product"] == 0).astype(int)

    scored["ml_weighted_hybrid_score"] = np.where(
        scored["eligible_for_recommendation"] == 1,
        scored["ml_weighted_raw_score"],
        0.0,
    )

    scored["ml_conversion_probability_recommendation_score"] = np.where(
        scored["eligible_for_recommendation"] == 1,
        scored["ml_conversion_probability"],
        0.0,
    )

    # Keep reasons fast and simple. Avoid row-wise apply on large data.
    top_weighted_signals = (
        learned_weights[learned_weights["ml_recommender_weight"] > 0.05]
        .sort_values("ml_recommender_weight", ascending=False)["signal"]
        .tolist()
    )
    readable_signals = [
        s.replace("_score_norm", "").replace("_score", "").replace("_", " ")
        for s in top_weighted_signals[:3]
    ]
    base_reason = "ML-weighted score driven by: " + ", ".join(readable_signals) if readable_signals else "ML-weighted score"

    scored["ml_recommendation_reason"] = base_reason
    scored.loc[scored["propensity_score"] >= 0.60, "ml_recommendation_reason"] += "; high propensity"
    high_prob_cutoff = scored["ml_conversion_probability"].quantile(0.80)
    scored.loc[scored["ml_conversion_probability"] >= high_prob_cutoff, "ml_recommendation_reason"] += "; high ML conversion probability"
    scored.loc[scored["has_product"] == 1, "ml_recommendation_reason"] = "already held - excluded"

    return scored

def top_n_recommendations(
    scored: pd.DataFrame,
    score_col: str,
    top_n: int,
    rank_label: str,
) -> pd.DataFrame:
    candidates = scored[scored["eligible_for_recommendation"] == 1].copy()

    candidates = candidates.sort_values(
        ["Customer ID", "Month period", score_col],
        ascending=[True, True, False],
    )

    candidates["recommendation_rank"] = (
        candidates.groupby(["Customer ID", "Month period"]).cumcount() + 1
    )

    out = candidates[candidates["recommendation_rank"] <= top_n].copy()
    out["recommendation_model"] = rank_label

    cols = [
        "Customer ID",
        "Month period",
        "recommendation_model",
        "recommendation_rank",
        "product_code",
        "Product_Name",
        score_col,
        "ml_conversion_probability",
        "propensity_score",
        "digital_intent_score",
        "transaction_interest_score",
        "profile_fit_score",
        "segment_affinity_score_norm",
        "cooccurrence_score_norm",
        "ml_recommendation_reason",
    ]

    cols = [c for c in cols if c in out.columns]
    return out[cols]


def evaluate_topn(
    scored: pd.DataFrame,
    score_col: str,
    model_name: str,
    top_n: int,
    customer_scope: set | None = None,
) -> dict:
    data = scored.copy()
    if customer_scope is not None:
        data = data[data["Customer ID"].isin(customer_scope)].copy()

    positives = data[
        (data["converted_next_month"] == 1) &
        (data["has_product"] == 0)
    ][["Customer ID", "product_code"]].drop_duplicates()

    if positives.empty:
        return {
            "model": model_name,
            "converted_customers": 0,
            "hit_rate_at_1": np.nan,
            f"hit_rate_at_{top_n}": np.nan,
            "mrr": np.nan,
            "avg_rank_if_hit": np.nan,
        }

    candidates = data[data["has_product"] == 0].copy()
    candidates = candidates.sort_values(
        ["Customer ID", "Month period", score_col],
        ascending=[True, True, False],
    )
    candidates["rank"] = candidates.groupby(["Customer ID", "Month period"]).cumcount() + 1

    recs = candidates[candidates["rank"] <= top_n][
        ["Customer ID", "product_code", "rank"]
    ]

    joined = positives.merge(
        recs,
        on=["Customer ID", "product_code"],
        how="left",
    )

    hit_1 = (joined["rank"] == 1).sum()
    hit_n = joined["rank"].le(top_n).sum()
    ranks = joined["rank"].dropna()

    return {
        "model": model_name,
        "converted_customers": positives["Customer ID"].nunique(),
        "hit_rate_at_1": hit_1 / len(positives),
        f"hit_rate_at_{top_n}": hit_n / len(positives),
        "mrr": (1 / ranks).sum() / len(positives) if len(ranks) else 0.0,
        "avg_rank_if_hit": ranks.mean() if len(ranks) else np.nan,
    }


def run_pipeline(
    model_input_dir: Path,
    conversion_path: Path,
    output_dir: Path,
    top_n: int,
    test_size: float,
    seed: int,
) -> None:
    print(f"Reading stage 03 model outputs from: {model_input_dir}")
    print(f"Reading conversion data from: {conversion_path}")
    print(f"Writing ML-weighted hybrid outputs to: {output_dir}")

    model, conversions = load_inputs(model_input_dir, conversion_path)
    model, conversions = clean_inputs(model, conversions)

    labels = add_conversion_labels(model, conversions)
    labels.to_csv(output_dir / "conversion_training_labels_june_to_july.csv", index=False)

    train_df, test_df, split_summary = split_by_customer(labels, test_size=test_size, seed=seed)
    split_summary.to_csv(output_dir / "ml_train_test_split_summary.csv", index=False)

    clf, test_metrics, confusion_df = train_logistic_weight_model(train_df, test_df)
    test_metrics.to_csv(output_dir / "ml_model_test_metrics.csv", index=False)
    confusion_df.to_csv(output_dir / "ml_test_confusion_matrix.csv", index=False)

    learned_weights = extract_learned_weights(clf)
    learned_weights.to_csv(output_dir / "ml_learned_signal_weights.csv", index=False)

    scored = apply_ml_weights(labels, clf, learned_weights)
    scored.to_csv(output_dir / "model_ready_ml_weighted_hybrid_scored.csv", index=False)

    # Top recommendations using the ML-learned weighted hybrid score.
    top_ml_weighted = top_n_recommendations(
        scored,
        score_col="ml_weighted_hybrid_score",
        top_n=top_n,
        rank_label="ML Weighted Hybrid",
    )
    top_ml_weighted.to_csv(output_dir / "final_top_3_recommendations_ml_weighted_hybrid.csv", index=False)

    # Also output probability-ranked recommendations for comparison.
    top_ml_probability = top_n_recommendations(
        scored,
        score_col="ml_conversion_probability_recommendation_score",
        top_n=top_n,
        rank_label="ML Conversion Probability",
    )
    top_ml_probability.to_csv(output_dir / "final_top_3_recommendations_ml_probability.csv", index=False)

    test_customers = set(test_df["Customer ID"].unique())

    comparison = pd.DataFrame([
        evaluate_topn(scored, "propensity_score", "Propensity Only", top_n, test_customers),
        evaluate_topn(scored, "hybrid_recommendation_score", "Previous Heuristic Hybrid", top_n, test_customers),
        evaluate_topn(scored, "ml_weighted_hybrid_score", "ML Weighted Hybrid", top_n, test_customers),
        evaluate_topn(scored, "ml_conversion_probability_recommendation_score", "ML Conversion Probability", top_n, test_customers),
    ])
    comparison.to_csv(output_dir / "ml_weighted_hybrid_vs_baselines_test_accuracy.csv", index=False)

    # Score-level evaluation on test candidate rows.
    test_scored = scored[
        (scored["Customer ID"].isin(test_customers)) &
        (scored["has_product"] == 0)
    ].copy()

    score_rows = []
    y = test_scored["converted_next_month"].astype(int)

    for col in [
        "propensity_score",
        "hybrid_recommendation_score",
        "ml_weighted_hybrid_score",
        "ml_conversion_probability_recommendation_score",
    ]:
        if col not in test_scored.columns:
            continue
        s = pd.to_numeric(test_scored[col], errors="coerce").fillna(0.0)

        try:
            auc = roc_auc_score(y, s)
        except ValueError:
            auc = np.nan

        try:
            ap = average_precision_score(y, s)
        except ValueError:
            ap = np.nan

        score_rows.append({
            "score_column": col,
            "roc_auc_test_candidate_rows": auc,
            "average_precision_test_candidate_rows": ap,
            "mean_score_converted": s[y == 1].mean() if y.sum() else np.nan,
            "mean_score_not_converted": s[y == 0].mean() if (y == 0).sum() else np.nan,
        })

    pd.DataFrame(score_rows).to_csv(output_dir / "score_level_test_metrics.csv", index=False)

    conversion_summary = conversions.groupby("Product converted", as_index=False).agg(
        conversions=("Customer ID", "count"),
        unique_customers=("Customer ID", "nunique"),
    )
    conversion_summary["Product converted name"] = conversion_summary["Product converted"].map(PRODUCT_NAME_MAP)
    conversion_summary = conversion_summary.sort_values("conversions", ascending=False)
    conversion_summary.to_csv(output_dir / "conversion_product_summary.csv", index=False)

    sanity = pd.DataFrame([
        {"check": "input_model_rows", "value": len(model)},
        {"check": "unique_customers", "value": model["Customer ID"].nunique()},
        {"check": "conversion_rows", "value": len(conversions)},
        {"check": "positive_labels", "value": int(labels["converted_next_month"].sum())},
        {"check": "train_customers", "value": train_df["Customer ID"].nunique()},
        {"check": "test_customers", "value": test_df["Customer ID"].nunique()},
        {"check": "already_held_positive_ml_weighted_score_rows", "value": int(((scored["has_product"] == 1) & (scored["ml_weighted_hybrid_score"] > 0)).sum())},
        {"check": "top3_rows_ml_weighted", "value": len(top_ml_weighted)},
    ])
    sanity.to_csv(output_dir / "ml_weighted_hybrid_sanity_checks.csv", index=False)

    print("\nDone. Key outputs:")
    for f in [
        "ml_learned_signal_weights.csv",
        "ml_model_test_metrics.csv",
        "ml_weighted_hybrid_vs_baselines_test_accuracy.csv",
        "final_top_3_recommendations_ml_weighted_hybrid.csv",
        "model_ready_ml_weighted_hybrid_scored.csv",
    ]:
        print(f"- {output_dir / f}")


def main() -> None:
    args = parse_args()
    model_input_dir, conversion_path, output_dir = resolve_paths(args)
    run_pipeline(
        model_input_dir=model_input_dir,
        conversion_path=conversion_path,
        output_dir=output_dir,
        top_n=args.top_n,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
