import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import itertools
import os

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BALANCE_THRESHOLD = 0.25   # Minimum proportion for each cluster
MAX_CLUSTERS = 8           # Maximum number of clusters to try
# Rule features used to identify bot clusters (clusters with lower mean are considered bots)
RULE_FEATURES = ['avg_post_score', 'avg_upvote_ratio', 'avg_comment_score']

# ==================== Data Loading ====================
DATA_DIR = "../data"
FILE_PATH = os.path.join(DATA_DIR, "BotSim-24.csv")

df = pd.read_csv(FILE_PATH)
print(f"Dataset size: {len(df)}")
print(f"True class distribution: Human {sum(df['label']==0)}, Bot {sum(df['label']==1)}")
print("="*80)

# Exclude non-feature columns and keep only numeric columns
exclude_cols = ['user_id', 'label']
feature_cols = [
    c for c in df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
]
# Ensure all selected columns are numeric (additional safety)
feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

print(f"Selected numeric feature columns ({len(feature_cols)}): {feature_cols}")

# If any missing values, fill with 0 to avoid scaling errors
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y_true = df['label'].values

X_raw = X.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def is_binary_feature(col_name):
    unique_vals = df[col_name].unique()
    return set(unique_vals).issubset({0, 1})

binary_flags = [is_binary_feature(col) for col in feature_cols]

# ---------- Evaluate a single feature combination for a given k ----------
def evaluate_k(X_sub, k):
    pca = PCA(n_components=1, random_state=42)
    X_pca = pca.fit_transform(X_sub).flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    counts = np.bincount(labels)
    if len(counts) < k or min(counts) / len(labels) < BALANCE_THRESHOLD:
        return None, None, None
    try:
        sil = silhouette_score(X_sub, labels)
    except:
        sil = -1
    # Weighted between-group variance (based on original feature space)
    global_mean = np.mean(X_sub, axis=0)
    total_var = 0
    for c in range(k):
        cluster_points = X_sub[labels == c]
        if len(cluster_points) == 0:
            continue
        center = np.mean(cluster_points, axis=0)
        n_c = len(cluster_points)
        total_var += n_c * np.sum((center - global_mean)**2)
    weighted_var = total_var / len(labels)
    return labels, sil, weighted_var

# ==================== Search all feature combinations + auto-determine cluster number ====================
MAX_FEATURES = 3
print(f"Searching feature combinations (1~{MAX_FEATURES} features), auto-selecting best cluster number (2~{MAX_CLUSTERS})...")

valid_candidates = []  # each element: (feature index list, labels, sil, var, k)
n_features = len(feature_cols)
for r in range(1, MAX_FEATURES + 1):
    for combo in itertools.combinations(range(n_features), r):
        idx_list = list(combo)
        X_sub = X_scaled[:, idx_list]
        # Try different k
        for k in range(2, MAX_CLUSTERS + 1):
            labels, sil, var = evaluate_k(X_sub, k)
            if labels is not None:
                valid_candidates.append((idx_list, labels, sil, var, k))

if not valid_candidates:
    print("No combination found that satisfies the balance constraint!")
    exit()

# Extract all sil and var
sil_vals = np.array([c[2] for c in valid_candidates])
var_vals = np.array([c[3] for c in valid_candidates])

# Min-Max normalization
sil_min, sil_max = sil_vals.min(), sil_vals.max()
var_min, var_max = var_vals.min(), var_vals.max()

sil_norm = (sil_vals - sil_min) / (sil_max - sil_min) if sil_max > sil_min else np.zeros_like(sil_vals)
var_norm = (var_vals - var_min) / (var_max - var_min) if var_max > var_min else np.zeros_like(var_vals)

# Calculate proportion of binary features in each combination as weight w (bias toward silhouette)
weights = np.array([sum(binary_flags[i] for i in c[0]) / len(c[0]) for c in valid_candidates])

# Combined score = w * sil_norm + (1-w) * var_norm
scores = weights * sil_norm + (1 - weights) * var_norm

# Select the candidate with the highest score
best_idx = np.argmax(scores)
best_indices, best_labels, best_sil, best_var, best_k = valid_candidates[best_idx]
best_combo = [feature_cols[i] for i in best_indices]
best_score = scores[best_idx]
best_weight = weights[best_idx]

print(f"\n🏆 Best feature combination (fusion strategy): {best_combo}")
print(f"   Auto-selected number of clusters: {best_k}")
print(f"   Combined score = {best_score:.4f}  (silhouette norm={sil_norm[best_idx]:.4f}, between-group var norm={var_norm[best_idx]:.4f})")
print(f"   Weight w (binary feature proportion) = {best_weight:.2f}")
print(f"   Silhouette (raw) = {best_sil:.4f}")
print(f"   Between-group variance (raw) = {best_var:.4f}")
counts = np.bincount(best_labels)
print(f"   Cluster sizes: {dict(zip(range(best_k), counts))}")

# ==================== Determine if each cluster is bot based on rule features ====================
# Find rule features that exist in the dataset
rule_indices = [feature_cols.index(f) for f in RULE_FEATURES if f in feature_cols]

# If none of the defined rule features exist, use a fallback chain
if not rule_indices:
    fallback_features = ['followers_count', 'avg_user_mention_count', 'avg_tweet_length']
    for fallback in fallback_features:
        if fallback in feature_cols:
            rule_indices = [feature_cols.index(fallback)]
            RULE_FEATURES = [fallback]
            print(f"Warning: None of the predefined rule features found. Using '{fallback}' as rule feature.")
            break
    # If even fallback features are not found, use the first numeric feature
    if not rule_indices:
        if feature_cols:
            rule_indices = [0]
            RULE_FEATURES = [feature_cols[0]]
            print(f"Warning: Using the first feature '{feature_cols[0]}' as rule feature.")
        else:
            raise ValueError("No feature columns available to use as rule feature!")

cluster_rule_means = []
for k in range(best_k):
    mask = (best_labels == k)
    if np.sum(mask) == 0:
        cluster_rule_means.append(0)
    else:
        cluster_vals = X_raw[mask][:, rule_indices]
        cluster_rule_means.append(np.mean(cluster_vals))

total_weighted_mean = np.average(cluster_rule_means, weights=counts)

print(f"\nDetermining bot clusters based on rule features {RULE_FEATURES}:")
for k in range(best_k):
    is_bot = cluster_rule_means[k] < total_weighted_mean
    label_str = "Bot" if is_bot else "Human"
    print(f"   Cluster {k}: mean = {cluster_rule_means[k]:.4f}, overall mean = {total_weighted_mean:.4f} → {label_str}")

# Generate predicted labels
y_pred = np.zeros_like(best_labels, dtype=int)
for k in range(best_k):
    if cluster_rule_means[k] < total_weighted_mean:
        y_pred[best_labels == k] = 1
    else:
        y_pred[best_labels == k] = 0

# ==================== Evaluation ====================
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n" + "="*80)
print("Classification Performance via Rule Mapping")
print("="*80)

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, random_state=42):
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