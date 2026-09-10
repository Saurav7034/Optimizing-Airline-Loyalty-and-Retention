"""
Pipeline Runner — Airline Loyalty Platform

Runs the full analytics pipeline:
  1. Load & clean data
  2. Engineer features (with temporal cutoff)
  3. Train churn model & score all customers
  4. Train segmentation & assign segments
  5. Apply retention rules
  6. Save final scored dataset

Run once before launching the Streamlit app:
    python run_pipeline.py
"""

import os
import pandas as pd

print("=" * 60)
print("  Airline Loyalty Analytics — Pipeline Runner")
print("=" * 60)

# Step 1: Load data
print("\n[1/5] Loading and cleaning data …")
from analytics.data_loader import build_master_dataset
loyalty_df, activity_df = build_master_dataset()

# Step 2: Feature engineering
print("\n[2/5] Engineering features …")
from analytics.feature_engineering import build_features
features = build_features(loyalty_df, activity_df)

# Step 3: Churn model
print("\n[3/5] Training churn model …")
from analytics.churn_model import train_model, predict_churn
model_result = train_model(features)
features = predict_churn(features, model_result["model"], model_result["scaler"])

# Step 4: Segmentation
print("\n[4/5] Training segmentation …")
from analytics.segmentation import train_segmentation, segment_profile_table
seg_result = train_segmentation(features)
features = seg_result["features"]

# Step 5: Retention rules
print("\n[5/5] Applying retention rules …")
from analytics.retention import apply_retention_rules
features = apply_retention_rules(features)

# Save
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(BASE_DIR, "analytics", "scored_customers.parquet")
features.to_parquet(out_path, index=False)
print(f"\n✅ Scored dataset saved → {out_path}")

# Summary
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Total customers        : {len(features):,}")
print(f"  Churn rate (label)     : {features['churn_label'].mean():.1%}")
print(f"  AUC-ROC                : {model_result['auc']:.4f}")
print(f"  5-fold CV AUC          : {model_result['cv_auc_mean']:.4f} ± {model_result['cv_auc_std']:.4f}")
print(f"  Silhouette score       : {seg_result['silhouette_score']:.4f}")
print()

profile = segment_profile_table(features)
print("  Segment Summary:")
print(profile.to_string(index=False))

print("\n  Risk Tier Distribution:")
print(features["churn_risk_tier"].value_counts().to_string())

print("\n  Top 3 Feature Importances:")
print(model_result["feature_importance"].head(3).to_string(index=False))

print("\n" + "=" * 60)
print("  Pipeline complete. Run: streamlit run app.py")
print("=" * 60)
