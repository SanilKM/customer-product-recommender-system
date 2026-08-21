# Customer Product Recommender System

A machine learning recommender system that generates personalized product recommendations for customers using customer demographics, product holdings, transaction behavior, digital activity, propensity scores, and historical conversion outcomes.

This project focuses on building a practical **Next-Best-Product recommendation pipeline** that can rank products for each customer, evaluate recommendation quality, and explain the customer segments driving the recommendations.

## Features

- Customer-level product recommendation pipeline
- Ranked top-N recommendations for each customer
- Baseline propensity-score recommender
- Hybrid recommender using multiple customer signals
- Customer-to-customer similarity using k-nearest neighbors
- Cluster-first kNN recommendation approach
- K-means customer segmentation
- Conversion-based evaluation using Hit Rate@K
- Confusion matrix generation for recommendation performance
- Cluster-level feature summaries
- Exact-value cluster interpretation for business validation
- Outlier detection and treatment using IQR-based capping
- Synthetic dataset support for scalable testing
- Automated output generation across multiple dataset sizes

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- CSV-based data pipeline

## Project Structure

```text
.
├── data/
│   ├── 01_raw/                         # Raw input datasets
│   ├── 02_features/                    # Processed customer-product features
│   ├── 03_model_ready/                 # Model-ready matrices and tables
│   ├── 04_conversion_ml_outputs/       # Conversion-model and evaluation outputs
│   └── 06_cluster_first_knn_outputs/   # Cluster-first kNN outputs
│
├── notebooks/
│   ├── EDA_Sanity_Check.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Recommender_Modeling_and_Validation.ipynb
│   └── Conversion_ML_and_CF.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── recommender_modeling.py
│   ├── conversion_ml_recommender.py
│   ├── cluster_first_knn_recommender.py
│   └── run_all_synthetic_iterations.py
│
└── outputs/
    └── charts, summaries, recommendation results, and evaluation files