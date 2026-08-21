Axis Bank Modeling Outputs

Generated from:
C:\Users\maheshsk\Downloads\Axis Int - Rec Sys\Coding\data\02_features

Main outputs:
1. model_ready_customer_product_features_scored.csv
   Full customer-product table with propensity, digital intent, transaction interest,
   co-occurrence, segment affinity, and final hybrid score.

2. final_top_3_recommendations_hybrid.csv
   Final top-3 product recommendations per customer-month, excluding products already held.

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
