"""
Create better cluster definition visuals from existing cluster-first kNN outputs.

Run from Coding/ folder:

python src/create_better_cluster_visualizations.py \
  --output-root data/06_cluster_first_knn_outputs \
  --datasets Syn_01 Syn_02 Syn_03 Syn_04

This does not rerun clustering or kNN. It only reads the existing output CSVs and creates
better report-ready visuals + a cluster definition summary CSV/Markdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


PRODUCTS = ["PL", "CC", "HL", "SA", "RD", "MF"]
PRODUCT_LABELS = {
    "PL": "Personal Loan",
    "CC": "Credit Card",
    "HL": "Home Loan",
    "SA": "Savings Account",
    "RD": "Recurring Deposit",
    "MF": "Mutual Fund",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="data/06_cluster_first_knn_outputs")
    parser.add_argument("--datasets", nargs="+", default=["Syn_01", "Syn_02", "Syn_03", "Syn_04"])
    return parser.parse_args()


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.1f}%"


def num(x):
    if pd.isna(x):
        return "NA"
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def short_feature_name(feature: str) -> str:
    replacements = {
        "Monthly income proxy (sn)": "Monthly income",
        "Avg monthly balance proxy (sn)": "Avg balance",
        "Risk profile score": "Risk score",
        "Days since last login (sn)": "Days since login",
        "digital_Days since last login (sn)": "Digital days since login",
        "digital_Login count 30D (sn)": "30D logins",
        "digital_Login count 90D (sn)": "90D logins",
        "txn_Average ticket size": "Avg ticket size",
        "txn_Count": "Txn count",
        "txn_Total credit amount (sn)": "Credit amount",
        "txn_Total debit amount (sn)": "Debit amount",
        "txn_UPI txn count (sn)": "UPI count",
        "txn_MCC diversity count (sn)": "MCC diversity",
        "txn_Days since last txn (sn)": "Days since txn",
        "txn_Avg monthly balance proxy (sn)": "Avg balance",
        "Campaign exposure count 90D (sn)": "Campaign exposure",
        "Products ignored count 90D (sn)": "Products ignored",
    }
    for old, new in replacements.items():
        feature = feature.replace(old, new)
    return feature.replace(" holding rate", " holding")


def value_for_feature(row):
    ft = row.get("feature_type", "")
    feature = str(row.get("feature", ""))
    val = row.get("cluster_exact_value")
    overall = row.get("overall_exact_value")

    if ft in ["product_holding", "categorical"] or "holding rate" in feature or " = " in feature:
        return pct(val), pct(overall)
    return num(val), num(overall)


def get_col(row, candidates):
    for c in candidates:
        if c in row.index and not pd.isna(row[c]):
            return row[c]
    return np.nan


def cluster_title(row: pd.Series) -> str:
    age = get_col(row, ["avg_Age"])
    income_band = str(get_col(row, ["top_Income band (sn)", "top_Income band"])).strip()
    persona = str(get_col(row, ["top_Persona"])).strip()
    lifestage = str(get_col(row, ["top_Lifestage"])).strip()

    cc = row.get("CC_holding_rate", np.nan)
    rd = row.get("RD_holding_rate", np.nan)
    mf = row.get("MF_holding_rate", np.nan)
    pl = row.get("PL_holding_rate", np.nan)

    digital_days = get_col(row, ["avg_digital_Days since last login (sn)", "avg_Days since last login (sn)"])
    income = get_col(row, ["avg_Monthly income proxy (sn)"])

    if "HNI" in income_band or "HNI" in persona or (not pd.isna(income) and income >= 400000):
        if not pd.isna(mf) and mf >= 0.30:
            return "HNI / wealth-oriented segment"
        return "High-income premium segment"
    if not pd.isna(age) and age <= 30:
        if not pd.isna(cc) and cc >= 0.40:
            return "Young digitally active / credit-ready segment"
        return "Young digitally active segment"
    if not pd.isna(digital_days) and digital_days >= 70:
        return "Dormant / low digital activity segment"
    if not pd.isna(rd) and rd >= 0.30:
        return "Mature savings / RD-oriented segment"
    if not pd.isna(cc) and cc >= 0.50:
        return "Credit-card-heavy engaged segment"
    if not pd.isna(pl) and pl >= 0.22:
        return "Personal-loan opportunity segment"
    if "Value" in persona or "Family" in persona or "Family" in lifestage:
        return "Family / value-seeker segment"
    return "General mixed customer segment"


def build_cluster_summary(dataset: str, profiles: pd.DataFrame, defining: pd.DataFrame, rec_counts: pd.DataFrame | None):
    rows = []

    for _, row in profiles.sort_values("cluster_id").iterrows():
        cluster_id = int(row["cluster_id"])
        title = cluster_title(row)
        cluster_defs = defining[defining["cluster_id"] == cluster_id].head(5).copy()

        differentiators = []
        for _, d in cluster_defs.iterrows():
            cluster_val, overall_val = value_for_feature(d)
            differentiators.append(
                f"{short_feature_name(str(d['feature']))}: {cluster_val} vs overall {overall_val} ({d['direction']})"
            )

        top_rec = "NA"
        top_rec_share = np.nan
        if rec_counts is not None and cluster_id in rec_counts.index:
            counts = rec_counts.loc[cluster_id, PRODUCTS].fillna(0)
            if counts.sum() > 0:
                top_rec = counts.idxmax()
                top_rec_share = counts.max() / counts.sum()

        rows.append({
            "dataset": dataset,
            "cluster_id": cluster_id,
            "cluster_name": title,
            "customers": int(row.get("customer_count", 0)),
            "customer_share": row.get("customer_share", np.nan),
            "age_avg": row.get("avg_Age", np.nan),
            "age_range": f"{num(row.get('min_Age'))}-{num(row.get('max_Age'))}",
            "income_avg": row.get("avg_Monthly income proxy (sn)", np.nan),
            "top_income_band": row.get("top_Income band (sn)", ""),
            "top_persona": row.get("top_Persona", ""),
            "top_lifestage": row.get("top_Lifestage", ""),
            "PL_holding": row.get("PL_holding_rate", np.nan),
            "CC_holding": row.get("CC_holding_rate", np.nan),
            "HL_holding": row.get("HL_holding_rate", np.nan),
            "SA_holding": row.get("SA_holding_rate", np.nan),
            "RD_holding": row.get("RD_holding_rate", np.nan),
            "MF_holding": row.get("MF_holding_rate", np.nan),
            "top_rank1_recommendation": top_rec,
            "top_rank1_recommendation_share": top_rec_share,
            "top_differentiators": " | ".join(differentiators),
        })

    return pd.DataFrame(rows)


def create_cluster_profile_card(dataset: str, row: pd.Series, defining: pd.DataFrame, rec_counts: pd.DataFrame | None, out_path: Path):
    cluster_id = int(row["cluster_id"])
    title = cluster_title(row)

    fig = plt.figure(figsize=(12, 7.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 1.05], width_ratios=[1.1, 1.0])

    # Text summary panel
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis("off")

    summary_lines = [
        f"Cluster {cluster_id}: {title}",
        "",
        f"Customers: {num(row.get('customer_count'))} ({pct(row.get('customer_share'))})",
        f"Age: avg {num(row.get('avg_Age'))}, range {num(row.get('min_Age'))}-{num(row.get('max_Age'))}",
        f"Income: avg {num(row.get('avg_Monthly income proxy (sn)'))}",
        f"Income band: {row.get('top_Income band (sn)', 'NA')}",
        f"Persona: {row.get('top_Persona', 'NA')}",
        f"Lifestage: {row.get('top_Lifestage', 'NA')}",
    ]
    ax_text.text(
        0.0, 1.0, "\n".join(summary_lines),
        va="top", ha="left", fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.8"},
    )

    # Product holding rates
    ax_prod = fig.add_subplot(gs[0, 1])
    prod_vals = [row.get(f"{p}_holding_rate", np.nan) * 100 for p in PRODUCTS]
    ax_prod.barh([PRODUCT_LABELS[p] for p in PRODUCTS], prod_vals)
    ax_prod.set_xlabel("Holding rate (%)")
    ax_prod.set_title("Product holding mix")
    ax_prod.set_xlim(0, max(100, np.nanmax(prod_vals) + 5))
    for i, v in enumerate(prod_vals):
        if not pd.isna(v):
            ax_prod.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)

    # Top differentiators as exact-value table.
    # This is clearer than one bar chart because features have different units
    # such as years, rupees, percentages, login counts, and transaction values.
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_diff.axis("off")
    d = defining[defining["cluster_id"] == cluster_id].head(5).copy()
    if not d.empty:
        table_rows = []
        for _, dd in d.iterrows():
            cluster_val, overall_val = value_for_feature(dd)
            table_rows.append([
                textwrap.shorten(short_feature_name(str(dd["feature"])), width=30, placeholder="…"),
                cluster_val,
                overall_val,
                str(dd.get("direction", "")).replace("higher than overall", "Higher").replace("lower than overall", "Lower").replace("more common than overall", "More common").replace("less common than overall", "Less common"),
            ])

        table = ax_diff.table(
            cellText=table_rows,
            colLabels=["Defining feature", "Cluster", "Overall", "Direction"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        ax_diff.set_title("Top differentiators with exact values", pad=12)
    else:
        ax_diff.text(0, 0.5, "No defining feature table found", va="center")

    # Recommendation mix
    ax_rec = fig.add_subplot(gs[1, 1])
    if rec_counts is not None and cluster_id in rec_counts.index:
        counts = rec_counts.loc[cluster_id, PRODUCTS].fillna(0)
        shares = counts / counts.sum() * 100 if counts.sum() > 0 else counts
        ax_rec.bar([PRODUCT_LABELS[p] for p in PRODUCTS], shares.values)
        ax_rec.set_ylabel("Share of rank-1 recs (%)")
        ax_rec.set_title("Top-1 recommendation mix")
        ax_rec.tick_params(axis="x", rotation=35)
        for i, v in enumerate(shares.values):
            ax_rec.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
    else:
        ax_rec.axis("off")
        ax_rec.text(0, 0.5, "No recommendation count table found", va="center")

    fig.suptitle(f"{dataset} cluster profile card", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_cluster_compare_dashboard(dataset: str, profiles: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Age vs income scatter
    ax = fig.add_subplot(gs[0, 0])
    x = profiles["avg_Age"]
    y = profiles["avg_Monthly income proxy (sn)"]
    sizes = profiles["customer_share"] * 3000 + 100
    ax.scatter(x, y, s=sizes, alpha=0.7)
    for _, r in profiles.iterrows():
        ax.text(r["avg_Age"], r["avg_Monthly income proxy (sn)"], str(int(r["cluster_id"])), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Average age")
    ax.set_ylabel("Average monthly income")
    ax.set_title("Cluster position: age vs income")

    # Digital recency
    ax = fig.add_subplot(gs[0, 1])
    digital_col = "avg_digital_Days since last login (sn)"
    if digital_col not in profiles.columns:
        digital_col = "avg_Days since last login (sn)"
    ax.bar(profiles["cluster_id"].astype(str), profiles[digital_col])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Avg days since login")
    ax.set_title("Digital recency by cluster")

    # Product heatmap without extra deps
    ax = fig.add_subplot(gs[1, 0])
    product_cols = [f"{p}_holding_rate" for p in PRODUCTS]
    product_matrix = profiles.set_index("cluster_id")[product_cols] * 100
    img = ax.imshow(product_matrix.values, aspect="auto")
    ax.set_xticks(range(len(PRODUCTS)))
    ax.set_xticklabels(PRODUCTS)
    ax.set_yticks(range(len(product_matrix.index)))
    ax.set_yticklabels(product_matrix.index.astype(str))
    ax.set_xlabel("Product")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Product holding rates (%)")
    for i in range(product_matrix.shape[0]):
        for j in range(product_matrix.shape[1]):
            ax.text(j, i, f"{product_matrix.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    # Simple table-like labels
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    lines = []
    for _, r in profiles.sort_values("cluster_id").iterrows():
        cid = int(r["cluster_id"])
        lines.append(f"C{cid}: {cluster_title(r)}")
    ax.text(0, 1, "Cluster labels\n\n" + "\n".join(lines), va="top", fontsize=10)

    fig.suptitle(f"{dataset}: cluster overview dashboard", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown_section(dataset: str, summary: pd.DataFrame, out_path: Path):
    lines = [f"## {dataset} Cluster Definitions", ""]
    lines.append("Use the cluster profile cards below instead of the older single normalized-bar plots. These cards combine the customer profile, product mix, top differentiators, and recommendation mix in one place.")
    lines.append("")
    lines.append(f"Paste here: `{dataset}/visualizations_better/00_cluster_overview_dashboard.png`")
    lines.append("")

    for _, r in summary.sort_values("cluster_id").iterrows():
        cid = int(r["cluster_id"])
        lines.append(f"### Cluster {cid}: {r['cluster_name']}")
        lines.append("")
        lines.append(
            f"This cluster has {num(r['customers'])} customers ({pct(r['customer_share'])}). "
            f"The average age is {num(r['age_avg'])}, with an age range of {r['age_range']}. "
            f"The average monthly income proxy is {num(r['income_avg'])}. "
            f"The dominant income band is {r['top_income_band']}, persona is {r['top_persona']}, and lifestage is {r['top_lifestage']}."
        )
        lines.append("")
        lines.append(
            f"Product mix: PL {pct(r['PL_holding'])}, CC {pct(r['CC_holding'])}, HL {pct(r['HL_holding'])}, "
            f"SA {pct(r['SA_holding'])}, RD {pct(r['RD_holding'])}, MF {pct(r['MF_holding'])}. "
            f"The most common rank-1 recommendation is {r['top_rank1_recommendation']} "
            f"({pct(r['top_rank1_recommendation_share'])} of rank-1 recommendations in this cluster)."
        )
        lines.append("")
        if isinstance(r["top_differentiators"], str) and r["top_differentiators"]:
            lines.append("Key differentiators: " + r["top_differentiators"] + ".")
            lines.append("")
        lines.append(f"Paste here: `{dataset}/visualizations_better/cluster_{cid}_profile_card.png`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def process_dataset(output_root: Path, dataset: str):
    dataset_dir = output_root / dataset
    profiles_path = dataset_dir / "cluster_first_cluster_profiles.csv"
    defining_path = dataset_dir / "cluster_defining_features_top10_exact_values.csv"
    rec_path = dataset_dir / "report_assets" / "report_top1_recommendation_counts_by_cluster.csv"

    if not profiles_path.exists() or not defining_path.exists():
        print(f"Skipping {dataset}: missing profile or defining feature file")
        return

    profiles = pd.read_csv(profiles_path)
    defining = pd.read_csv(defining_path)
    rec_counts = None
    if rec_path.exists():
        rec_counts = pd.read_csv(rec_path).set_index("cluster_id")

    viz_dir = dataset_dir / "visualizations_better"
    report_dir = dataset_dir / "report_assets_better"
    viz_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = build_cluster_summary(dataset, profiles, defining, rec_counts)
    summary.to_csv(report_dir / "better_cluster_definition_summary.csv", index=False)

    for _, row in profiles.sort_values("cluster_id").iterrows():
        cid = int(row["cluster_id"])
        create_cluster_profile_card(
            dataset=dataset,
            row=row,
            defining=defining,
            rec_counts=rec_counts,
            out_path=viz_dir / f"cluster_{cid}_profile_card.png",
        )

    create_cluster_compare_dashboard(
        dataset=dataset,
        profiles=profiles,
        out_path=viz_dir / "00_cluster_overview_dashboard.png",
    )

    write_markdown_section(
        dataset=dataset,
        summary=summary,
        out_path=report_dir / "better_cluster_definition_section.md",
    )

    print(f"Created better cluster visuals for {dataset}: {viz_dir}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    for dataset in args.datasets:
        process_dataset(output_root, dataset)


if __name__ == "__main__":
    main()
