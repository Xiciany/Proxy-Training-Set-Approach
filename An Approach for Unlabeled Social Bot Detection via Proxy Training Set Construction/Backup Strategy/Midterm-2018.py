# -*- coding: utf-8 -*-
"""
Unsupervised clustering method (feature combination search + PCA + Jenks breaks)
Search exactly 2 features.
Balance check and silhouette score removed for performance optimization.
Bootstrap 95% confidence intervals included.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import itertools
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ====================== Configuration ======================
# The script is assumed to be in a subfolder (e.g., "Main Experiment"),
# so data is in "../data" relative to this script.
DATA_DIR = "../data"
TARGET_FILE = os.path.join(DATA_DIR, "midterm-2018.csv")

RULE_FEATURE = 'followers_count'   # Rule feature: lower mean indicates bot cluster
RANDOM_STATE = 42

# ====================== Data Loading ======================
def load_midterm2018_data():
    """Load midterm-2018 from unified CSV."""
    df = pd.read_csv(TARGET_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    # Automatically select numeric feature columns (exclude 'user_id' and 'label')
    exclude_cols = ['user_id', 'label']
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Ensure all selected columns are numeric (additional safety)
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the dataset!")
    # Fill missing values with 0
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Dataset size: {len(df)}")
    print(f"True class distribution: Human {sum(df['label']==0)}, Bot {sum(df['label']==1)}")
    print("="*80)
    print(f"Automatically selected numeric features ({len(feature_cols)}): {feature_cols}")
    return df, feature_cols

# ====================== Helper Functions ======================
def is_binary_feature(col_series):
    unique_vals = col_series.unique()
    return set(unique_vals).issubset({0, 1})

def jenks_break_fast(data):
    """
    Fast Jenks break using prefix sums (maximizes between-group variance).
    Time complexity O(n), n = number of samples.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    prefix_sum = np.zeros(n+1, dtype=np.float64)
    np.cumsum(sorted_vals, out=prefix_sum[1:])  # prefix_sum[1:] = cumulative sum
    total_sum = prefix_sum[n]
    max_var = -1.0
    best_th = sorted_vals[n//2]  # default median

    for i in range(1, n):
        left_count = i
        right_count = n - i
        left_sum = prefix_sum[i]
        right_sum = total_sum - left_sum
        if left_count > 0 and right_count > 0:
            var_between = (left_sum*left_sum / left_count + right_sum*right_sum / right_count) - total_sum*total_sum / n
            if var_between > max_var:
                max_var = var_between
                best_th = (sorted_vals[i-1] + sorted_vals[i]) / 2.0
    return best_th, max_var

def evaluate_feature_combo(X_sub):
    """
    PCA to 1D, Jenks break, return labels and between-group variance.
    """
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sub).flatten()
    th, var_between = jenks_break_fast(X_pca)
    labels = (X_pca >= th).astype(int)
    return labels, var_between

# ==================== Bootstrap Confidence Intervals ====================
def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, random_state=42):
    """
    Compute bootstrap confidence intervals for Accuracy, Precision, Recall, F1.
    Returns a dict with metric names as keys and (lower, upper) tuples.
    """
    n = len(y_true)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    np.random.seed(random_state)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred[idx]
        acc = accuracy_score(y_true_bs, y_pred_bs)
        prec = precision_score(y_true_bs, y_pred_bs, zero_division=0)
        rec = recall_score(y_true_bs, y_pred_bs, zero_division=0)
        f1 = f1_score(y_true_bs, y_pred_bs, zero_division=0)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
    ci = {}
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    for key in metrics:
        arr = np.array(metrics[key])
        ci[key] = (np.percentile(arr, lower_percentile), np.percentile(arr, upper_percentile))
    return ci

# ==================== Main Pipeline ====================
def main():
    print("=" * 80)
    print("Unsupervised Clustering Method: Feature Search (2 features) + PCA + Jenks Breaks")
    print("Performance optimized (no balance check, no silhouette)")
    print("Adapted for midterm-2018 (labels used only for final evaluation)")
    print("=" * 80)

    # Load data and automatically select numeric features
    df, feature_cols = load_midterm2018_data()
    n_features = len(feature_cols)
    if n_features < 2:
        print("Need at least 2 features to perform combination search.")
        return

    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    binary_flags = [is_binary_feature(df[col]) for col in feature_cols]

    # Search all 2-feature combinations
    total_combos = len(list(itertools.combinations(range(n_features), 2)))
    print(f"\nSearching all 2-feature combinations (total {total_combos})...")

    candidates = []  # stores (feature indices, labels, between-group variance)

    pbar = tqdm(total=total_combos, desc="Search progress")
    for combo in itertools.combinations(range(n_features), 2):
        idx_list = list(combo)
        X_sub = X_scaled[:, idx_list]
        labels, var = evaluate_feature_combo(X_sub)
        candidates.append((idx_list, labels, var))
        pbar.update(1)
    pbar.close()

    if not candidates:
        print("No valid combinations found!")
        return

    # Select the combination with the maximum between-group variance
    var_vals = np.array([c[2] for c in candidates])
    best_idx = np.argmax(var_vals)
    best_indices, best_labels, best_var = candidates[best_idx]
    best_combo = [feature_cols[i] for i in best_indices]

    print(f"\n🏆 Best feature combination: {best_combo}")
    print(f"   Between-group variance = {best_var:.4f}")
    counts = np.bincount(best_labels)
    print(f"   Cluster 0: {counts[0]}, Cluster 1: {counts[1]}")

    # ==================== Rule-based label mapping ====================
    # Determine which feature to use as rule: prefer 'followers_count' if exists, else first feature
    if RULE_FEATURE in feature_cols:
        rule_col = RULE_FEATURE
    else:
        rule_col = feature_cols[0]
        print(f"Note: Specified rule feature '{RULE_FEATURE}' not found. Using '{rule_col}' instead.")

    rule_col_index = feature_cols.index(rule_col)
    cluster0_mean = X_raw[best_labels == 0, rule_col_index].mean()
    cluster1_mean = X_raw[best_labels == 1, rule_col_index].mean()

    print(f"\nRule-based bot detection using '{rule_col}':")
    print(f"   Cluster 0 mean {rule_col} = {cluster0_mean:.4f}")
    print(f"   Cluster 1 mean {rule_col} = {cluster1_mean:.4f}")

    if cluster0_mean < cluster1_mean:
        # Cluster 0 -> Bot, Cluster 1 -> Human
        y_pred = np.where(best_labels == 0, 1, 0)
        mapping_desc = "Cluster 0 = Bot, Cluster 1 = Human"
    else:
        y_pred = np.where(best_labels == 1, 1, 0)
        mapping_desc = "Cluster 0 = Human, Cluster 1 = Bot"

    print(f"Mapping rule: {mapping_desc}")

    # ==================== Evaluation (with Bootstrap CI) ====================
    y_true = df['label'].values
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ci = bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, random_state=42)

    print("\n" + "=" * 80)
    print("Classification Performance via Rule Mapping (95% CI)")
    print("=" * 80)
    print(f"Accuracy : {acc:.4f}  [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
    print(f"Precision: {pre:.4f}  [{ci['precision'][0]:.4f}, {ci['precision'][1]:.4f}]")
    print(f"Recall   : {rec:.4f}  [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
    print(f"F1 Score : {f1:.4f}  [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"             Pred Human   Pred Bot")
    print(f"True Human     {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"True Bot       {cm[1,0]:>6}   {cm[1,1]:>6}")

if __name__ == "__main__":
    main()