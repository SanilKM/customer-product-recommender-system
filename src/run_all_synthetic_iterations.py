"""
Run cluster-first kNN across Syn_01 to Syn_04.

Run from Coding folder:
    python src/run_all_synthetic_iterations.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATASET_METADATA = {
    "Syn_01": {
        "dataset_role": "Large synthetic iteration 1",
        "source_label": "LLM synthetic dataset 1",
        "note": "Generated larger synthetic dataset.",
    },
    "Syn_02": {
        "dataset_role": "Large synthetic iteration 2",
        "source_label": "LLM synthetic dataset 2",
        "note": "Generated larger synthetic dataset.",
    },
    "Syn_03": {
        "dataset_role": "Original 15k baseline",
        "source_label": "Original 15k dataset copied into Syn_03",
        "note": "Grok was not able to generate a good synthetic dataset for this iteration, so I copied the original 15k dataset into Syn_03.",
    },
    "Syn_04": {
        "dataset_role": "Large synthetic iteration 4",
        "source_label": "LLM synthetic dataset 4",
        "note": "Generated larger synthetic dataset.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=str, default="data/01_raw")
    parser.add_argument("--output-root", type=str, default="data/06_cluster_first_knn_outputs")
    parser.add_argument("--datasets", nargs="+", default=["Syn_01", "Syn_02", "Syn_03", "Syn_04"])
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--outlier-method", type=str, default="iqr", choices=["none", "iqr", "quantile"])
    return parser.parse_args()


def has_required_data(raw_dir: Path) -> bool:
    required = [
        "customer_demographics.csv",
        "product_holdings.csv",
        "digital_login.csv",
        "transaction_aggregates.csv",
        "conversion_data_july_2026.csv",
    ]
    return all((raw_dir / f).exists() for f in required)


def raw_customer_count(raw_dir: Path) -> int | None:
    path = raw_dir / "customer_demographics.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["Customer ID"])
        return int(df["Customer ID"].nunique())
    except Exception:
        return None


def build_dataset_metadata(datasets: list[str], raw_root: Path) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        meta = DATASET_METADATA.get(dataset, {
            "dataset_role": "Synthetic iteration",
            "source_label": "Synthetic dataset",
            "note": "",
        })
        raw_dir = raw_root / dataset
        rows.append({
            "dataset": dataset,
            "dataset_role": meta["dataset_role"],
            "source_label": meta["source_label"],
            "note": meta["note"],
            "raw_customer_count": raw_customer_count(raw_dir),
            "raw_folder": str(raw_dir),
        })
    return pd.DataFrame(rows)


def add_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset" not in df.columns:
        return df
    merge_cols = [
        "dataset",
        "dataset_role",
        "source_label",
        "note",
        "raw_customer_count",
    ]
    merge_cols = [c for c in merge_cols if c in metadata.columns]
    return df.merge(metadata[merge_cols], on="dataset", how="left")


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    summary_dir = output_root / "_summary"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_dataset_metadata(args.datasets, raw_root)
    metadata.to_csv(summary_dir / "dataset_metadata.csv", index=False)

    script_path = Path("src") / "cluster_first_knn_recommender.py"
    run_rows = []

    for dataset in args.datasets:
        raw_dir = raw_root / dataset
        output_dir = output_root / dataset
        role = DATASET_METADATA.get(dataset, {}).get("dataset_role", "Synthetic iteration")

        if not has_required_data(raw_dir):
            msg = f"Skipping {dataset}: required CSVs not found in {raw_dir}"
            print(msg)
            run_rows.append({
                "dataset": dataset,
                "dataset_role": role,
                "status": "skipped_missing_data",
                "message": msg,
            })
            continue

        print(f"\nDataset: {dataset} | {role}")

        cmd = [
            sys.executable,
            str(script_path),
            "--raw-dir", str(raw_dir),
            "--conversion-data", str(raw_dir / "conversion_data_july_2026.csv"),
            "--output-dir", str(output_dir),
            "--dataset-name", dataset,
            "--n-clusters", str(args.n_clusters),
            "--n-neighbors", str(args.n_neighbors),
            "--outlier-method", args.outlier_method,
        ]

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        run_rows.append({
            "dataset": dataset,
            "dataset_role": role,
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        })

    pd.DataFrame(run_rows).to_csv(summary_dir / "run_all_iterations_log.csv", index=False)

    aggregate_specs = [
        ("dataset_run_summary.csv", "all_dataset_run_summary.csv"),
        ("tech_specs_runtime_by_step.csv", "all_tech_specs_runtime_by_step.csv"),
        ("cumulative_hit_rate_metrics.csv", "all_cumulative_hit_rate_metrics.csv"),
        ("cluster_first_knn_accuracy_against_july_conversions.csv", "all_accuracy_wide.csv"),
        ("cluster_first_cluster_profiles.csv", "all_cluster_profiles.csv"),
        ("cluster_exact_feature_values.csv", "all_cluster_exact_feature_values.csv"),
        ("cluster_defining_features_top10_exact_values.csv", "all_cluster_defining_features_top10_exact_values.csv"),
        ("outlier_treatment_summary.csv", "all_outlier_treatment_summary.csv"),
        ("top3_hit_rate_by_actual_product.csv", "all_top3_hit_rate_by_actual_product.csv"),
    ]

    for source_name, out_name in aggregate_specs:
        frames = []
        for dataset in args.datasets:
            path = output_root / dataset / source_name
            if path.exists():
                df = pd.read_csv(path)
                if "dataset" not in df.columns:
                    df.insert(0, "dataset", dataset)
                frames.append(df)
        if frames:
            combined = add_metadata(pd.concat(frames, ignore_index=True), metadata)
            combined.to_csv(summary_dir / out_name, index=False)

    # Confusion matrices need a dataset tag before aggregation.
    frames = []
    for dataset in args.datasets:
        path = output_root / dataset / "confusion_matrix_rank1_actual_vs_predicted.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "dataset", dataset)
            frames.append(df)
    if frames:
        combined = add_metadata(pd.concat(frames, ignore_index=True), metadata)
        combined.to_csv(summary_dir / "all_confusion_matrices_rank1.csv", index=False)

    print("\nAggregated summary outputs written to:", summary_dir)
    print("\nDataset metadata:")
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
