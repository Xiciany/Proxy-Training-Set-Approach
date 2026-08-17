import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
MIN_FEATURES = 2          # Minimum number of features in combination
MAX_FEATURES = 3          # Maximum number of features in combination
BALANCE_THRESHOLD = 0.25  # Minimum proportion for each cluster

# ==================== Data Loading ====================
# Script is assumed to be in a subfolder (e.g., "Main Experiment"),
# so data is in "../data" relative to this script.
DATA_DIR = "../data"
FILE_PATH = os.path.join(DATA_DIR, "Twibot-20.csv")

def load_data():
    """Load Twibot-20 dataset from pre-extracted CSV and auto-select numeric features."""
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset size: {len(df)}")
    print(f"True class distribution: Human {sum(df['label']==0)}, Bot {sum(df['label']==1)}")
    print("="*80)

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

    print(f"Automatically selected numeric features ({len(feature_cols)}): {feature_cols}")

    # Fill missing values with 0 to avoid scaling errors
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y_true = df['label'].values

    # Save raw followers_count for rule-based labeling (if exists)
    if 'followers_count' in feature_cols:
        followers_raw = df['followers_count'].values
    else:
        # Fallback: use the first feature as rule feature
        rule_col = feature_cols[0]
        print(f"Note: 'followers_count' not found. Using '{rule_col}' as rule feature.")
        followers_raw = df[rule_col].values

    # Determine binary feature flags
    binary_flags = [is_binary_feature(df[col]) for col in feature_cols]

    return df, feature_cols, X, y_true, followers_raw, binary_flags

def is_binary_feature(series):
    unique_vals = series.dropna().unique()
    return set(unique_vals).issubset({0, 1})

# ==================== Jenks natural breaks ====================
def jenks_break(data):
    sorted_vals = np.sort(data)
    max_var = -1
    best_th = np.median(sorted_vals)
    for i in range(1, len(sorted_vals) - 1):
        left = sorted_vals[:i]
        right = sorted_vals[i:]
        var_between = (len(left) * (np.mean(left) - np.mean(data))**2 +
                       len(right) * (np.mean(right) - np.mean(data))**2) / len(data)
        if var_between > max_var:
            max_var = var_between
            best_th = (sorted_vals[i-1] + sorted_vals[i]) / 2
    return best_th, max_var

def evaluate_split(X_sub):
    """
    PCA to 1D, Jenks break, return labels, silhouette, and between-group variance.
    Returns (None, None, None) if either cluster proportion < BALANCE_THRESHOLD.
    """
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sub).flatten()
    th, var_between = jenks_break(X_pca)
    labels = (X_pca >= th).astype(int)
    counts = np.bincount(labels)
    if len(counts) < 2 or min(counts) / len(labels) < BALANCE_THRESHOLD:
        return None, None, None
    try:
        sil = silhouette_score(X_sub, labels)
    except:
        sil = -1
    return labels, sil, var_between

# ==================== Search best feature combination (fusion strategy) ====================
def search_best_combo(X_scaled, feature_names, binary_flags):
    n_features = X_scaled.shape[1]
    valid_candidates = []  # stores (feature indices, labels, silhouette, variance)

    for r in range(MIN_FEATURES, MAX_FEATURES + 1):
        combos = list(combinations(range(n_features), r))
        print(f"Searching {r}-feature combinations (total {len(combos)})...")
        for idx in combos:
            idx_list = list(idx)
            X_sub = X_scaled[:, idx_list]
            labels, sil, var = evaluate_split(X_sub)
            if labels is not None:
                valid_candidates.append((idx_list, labels, sil, var))

    if not valid_candidates:
        print("No combination found that satisfies the balance constraint!")
        return None, None, None, None

    # Extract all silhouette and variance values
    sil_vals = np.array([c[2] for c in valid_candidates])
    var_vals = np.array([c[3] for c in valid_candidates])

    # Min-Max normalization
    sil_min, sil_max = sil_vals.min(), sil_vals.max()
    var_min, var_max = var_vals.min(), var_vals.max()

    sil_norm = (sil_vals - sil_min) / (sil_max - sil_min) if sil_max > sil_min else np.zeros_like(sil_vals)
    var_norm = (var_vals - var_min) / (var_max - var_min) if var_max > var_min else np.zeros_like(var_vals)

    # Weight w = proportion of binary features in the combination (bias toward silhouette)
    weights = np.array([sum(binary_flags[i] for i in c[0]) / len(c[0]) for c in valid_candidates])

    # Combined score = w * sil_norm + (1-w) * var_norm
    scores = weights * sil_norm + (1 - weights) * var_norm

    # Select the best combination
    best_idx = np.argmax(scores)
    best_indices, best_labels, best_sil, best_var = valid_candidates[best_idx]
    best_features = [feature_names[i] for i in best_indices]
    best_score = scores[best_idx]
    best_weight = weights[best_idx]

    return best_features, best_labels, best_score, (best_sil, best_var, best_weight, sil_norm[best_idx], var_norm[best_idx])

# ==================== Bootstrap confidence intervals ====================
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
    print("Fusion Strategy: Weighted Silhouette + Between-Group Variance")
    print(f"Enforce each cluster at least {BALANCE_THRESHOLD*100:.0f}% of data")
    print("Label mapping rule: cluster with lower mean of 'followers_count' → Bot")
    print("=" * 80)

    # Load data
    df, feature_cols, X, y_true, followers_raw, binary_flags = load_data()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Search for best feature combination (2~3 features)
    best_features, best_labels, best_score, extra = search_best_combo(
        X_scaled, feature_cols, binary_flags
    )

    if best_features is None:
        return

    best_sil, best_var, weight, sil_norm, var_norm = extra

    print(f"\n🏆 Best feature combination: {best_features}")
    print(f"   Combined score = {best_score:.4f}  (silhouette norm={sil_norm:.4f}, between-group var norm={var_norm:.4f})")
    print(f"   Weight w (binary feature proportion) = {weight:.2f}")
    print(f"   Silhouette (raw) = {best_sil:.4f}")
    print(f"   Between-group variance (raw) = {best_var:.4f}")
    counts = np.bincount(best_labels)
    print(f"   Cluster 0: {counts[0]}, Cluster 1: {counts[1]}")

    # ==================== Rule-based label mapping using followers_count ====================
    cluster0_mean_followers = followers_raw[best_labels == 0].mean()
    cluster1_mean_followers = followers_raw[best_labels == 1].mean()

    print(f"\nRule-based detection using followers_count:")
    print(f"   Cluster 0 mean followers_count = {cluster0_mean_followers:.2f}")
    print(f"   Cluster 1 mean followers_count = {cluster1_mean_followers:.2f}")

    # The cluster with lower mean is considered bot (bots usually have fewer followers)
    if cluster0_mean_followers < cluster1_mean_followers:
        # Cluster 0 -> Bot, Cluster 1 -> Human
        y_pred = np.where(best_labels == 0, 1, 0)
        mapping_desc = "Cluster 0 = Bot, Cluster 1 = Human"
    else:
        # Cluster 1 -> Bot, Cluster 0 -> Human
        y_pred = np.where(best_labels == 1, 1, 0)
        mapping_desc = "Cluster 0 = Human, Cluster 1 = Bot"

    print(f"Mapping rule: {mapping_desc}")

    # ==================== Evaluation with Bootstrap CI ====================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ci = bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, random_state=42)

    print("\n" + "=" * 80)
    print("Classification Performance via Rule Mapping (95% CI)")
    print("=" * 80)
    print(f"Accuracy : {acc:.4f}  [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
    print(f"Precision: {prec:.4f}  [{ci['precision'][0]:.4f}, {ci['precision'][1]:.4f}]")
    print(f"Recall   : {rec:.4f}  [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
    print(f"F1 Score : {f1:.4f}  [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"             Pred Human   Pred Bot")
    print(f"True Human     {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"True Bot       {cm[1,0]:>6}   {cm[1,1]:>6}")

if __name__ == "__main__":
    main()