import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import itertools
import os

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BALANCE_THRESHOLD = 0.25   # Minimum proportion for each cluster
# Rule feature used to identify bot clusters (lower mean indicates bot)
# If the specified feature is not available, the first numeric feature will be used.
RULE_FEATURE = 'followers_count'  # Typical feature: lower follower count often indicates bot

# ==================== Data Loading ====================
# The script is assumed to be in a subfolder (e.g., "Backup Strategy" or "Main Experiment"),
# so data is in "../data" relative to this script.
DATA_DIR = "../data"
FILE_PATH = os.path.join(DATA_DIR, "Cresci-2017.csv")

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
# Additional safety: ensure columns are truly numeric
feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

if not feature_cols:
    raise ValueError("No numeric feature columns found in the dataset!")

print(f"Automatically selected numeric features ({len(feature_cols)}): {feature_cols}")

# Fill missing values with 0 to avoid scaling errors
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y_true = df['label'].values

# Save raw data for rule-based labeling
X_raw = X.copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Check if a feature is binary (values only 0/1) ----------
def is_binary_feature(col_name):
    unique_vals = df[col_name].unique()
    return set(unique_vals).issubset({0, 1})

binary_flags = [is_binary_feature(col) for col in feature_cols]

# ---------- Jenks natural breaks ----------
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

def evaluate_feature_combo(X_sub, col_indices):
    """
    Reduce feature subset to 1D via PCA, then apply Jenks break.
    If either cluster proportion < BALANCE_THRESHOLD, returns (None, None, None).
    Otherwise returns (labels, silhouette_score, group_variance).
    """
    pca = PCA(n_components=1, random_state=42)
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

# ==================== Search all feature combinations (fusion strategy) ====================
print("Searching all feature combinations (1-2 features) using fusion metric (weighted silhouette + between-group variance)...")

valid_candidates = []  # stores (feature indices, labels, silhouette, variance)

n_features = len(feature_cols)
min_r = 1
max_r = min(2, n_features)
for r in range(min_r, max_r + 1):
    for combo in itertools.combinations(range(n_features), r):
        idx_list = list(combo)
        X_sub = X_scaled[:, idx_list]
        labels, sil, var = evaluate_feature_combo(X_sub, idx_list)
        if labels is not None:
            valid_candidates.append((idx_list, labels, sil, var))

if not valid_candidates:
    print("No combination found that satisfies the balance constraint!")
    exit()

# Extract all silhouette and variance values
sil_vals = np.array([c[2] for c in valid_candidates])
var_vals = np.array([c[3] for c in valid_candidates])

# Min-Max normalization
sil_min, sil_max = sil_vals.min(), sil_vals.max()
var_min, var_max = var_vals.min(), var_vals.max()

sil_norm = (sil_vals - sil_min) / (sil_max - sil_min) if sil_max > sil_min else np.zeros_like(sil_vals)
var_norm = (var_vals - var_min) / (var_max - var_min) if var_max > var_min else np.zeros_like(var_vals)

# Compute weight w = proportion of binary features in the combination (bias toward silhouette)
weights = np.array([sum(binary_flags[i] for i in c[0]) / len(c[0]) for c in valid_candidates])

# Combined score = w * sil_norm + (1-w) * var_norm
scores = weights * sil_norm + (1 - weights) * var_norm

# Select the best combination
best_idx = np.argmax(scores)
best_indices, best_labels, best_sil, best_var = valid_candidates[best_idx]
best_combo = [feature_cols[i] for i in best_indices]
best_score = scores[best_idx]
best_weight = weights[best_idx]

print(f"\n🏆 Best feature combination (fusion strategy): {best_combo}")
print(f"   Combined score = {best_score:.4f}  (silhouette norm={sil_norm[best_idx]:.4f}, between-group var norm={var_norm[best_idx]:.4f})")
print(f"   Weight w (binary feature proportion) = {best_weight:.2f}")
print(f"   Silhouette (raw) = {best_sil:.4f}")
print(f"   Between-group variance (raw) = {best_var:.4f}")
counts = np.bincount(best_labels)
print(f"   Cluster 0: {counts[0]}, Cluster 1: {counts[1]}")

# ==================== Rule-based label mapping ====================
# Determine which rule feature to use: if RULE_FEATURE exists, use it; otherwise use the first numeric feature
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

# The cluster with lower mean is considered bot (bots usually have lower influence/follower count)
if cluster0_mean < cluster1_mean:
    # Cluster 0 -> bot, Cluster 1 -> human
    y_pred = np.where(best_labels == 0, 1, 0)
    mapping_desc = f"Cluster 0 = Bot (lower {rule_col}), Cluster 1 = Human"
else:
    # Cluster 1 -> bot
    y_pred = np.where(best_labels == 1, 1, 0)
    mapping_desc = f"Cluster 0 = Human, Cluster 1 = Bot (lower {rule_col})"

print(f"Mapping rule: {mapping_desc}")

# ==================== Evaluation ====================
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n" + "="*80)
print("Classification Performance via Rule Mapping")
print("="*80)

# ---------- Bootstrap 95% confidence intervals ----------
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

ci = bootstrap_ci(y_true, y_pred)

print(f"Accuracy : {acc:.4f}  [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
print(f"Precision: {pre:.4f}  [{ci['precision'][0]:.4f}, {ci['precision'][1]:.4f}]")
print(f"Recall   : {rec:.4f}  [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
print(f"F1 Score : {f1:.4f}  [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(f"             Pred Human   Pred Bot")
print(f"True Human     {cm[0,0]:>6}   {cm[0,1]:>6}")
print(f"True Bot       {cm[1,0]:>6}   {cm[1,1]:>6}")