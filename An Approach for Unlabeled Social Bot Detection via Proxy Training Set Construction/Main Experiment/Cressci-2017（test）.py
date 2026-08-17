import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import itertools
import warnings
warnings.filterwarnings('ignore')

# ====================== Path Configuration ======================
DATA_DIR = r"../data"
TWIBOT_FILE = os.path.join(DATA_DIR, "Twibot-22（part）.csv")
CRESCI_FILE = os.path.join(DATA_DIR, "Cresci-2017.csv")

# ====================== Parameters ======================
RANDOM_STATE = 42
BALANCE_THRESHOLD = 0.25          # Minimum proportion for each cluster during clustering
TARGET_RATIO = (3, 4)             # Bot : Human ratio for sampling (≈1:1.33)
SAMPLE_SIZE_RATIO = 1.0           # Sampling size relative to target dataset size (1.0 = equal)
MIN_SAMPLE_RATIO = 0.75           # Minimum target coverage ratio for dynamic tolerance sampling
MAX_TOL = 0.8                     # Maximum tolerance
TOL_STEP = 0.02                   # Tolerance increment step
INIT_TOL = 0.01                   # Initial tolerance
SCALE_FEATURES = ['followers_count', 'following_count', 'user_age_days', 'tweet_count',
                  'avg_tweets_per_day', 'listed_count']

# ====================== Data Loading Functions ======================
def load_twibot22_from_csv():
    """Load Twibot-22 dataset from pre‑extracted CSV file."""
    df = pd.read_csv(TWIBOT_FILE, encoding='utf-8-sig')
    # Ensure label column exists and is integer
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    # All columns except user_id and label are features
    feature_cols = [col for col in df.columns if col not in ['user_id', 'label']]
    return df, feature_cols

def load_cresci_from_csv():
    """Load Cresci-2017 dataset from pre‑extracted CSV file."""
    df = pd.read_csv(CRESCI_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    feature_cols = [col for col in df.columns if col not in ['user_id', 'label']]
    return df, feature_cols

# ====================== Helper Functions (unchanged) ======================
def is_binary_feature(series):
    uniq = series.dropna().unique()
    return set(uniq).issubset({0, 1})

def jenks_break(data):
    sorted_vals = np.sort(data)
    max_var = -1
    best_th = np.median(sorted_vals)
    for i in range(1, len(sorted_vals) - 1):
        left = sorted_vals[:i]
        right = sorted_vals[i:]
        var_between = (len(left) * (np.mean(left) - np.mean(data)) ** 2 +
                       len(right) * (np.mean(right) - np.mean(data)) ** 2) / len(data)
        if var_between > max_var:
            max_var = var_between
            best_th = (sorted_vals[i - 1] + sorted_vals[i]) / 2
    return best_th, max_var

def evaluate_feature_combo(X_sub):
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

def search_best_combo(X_scaled, feature_names, binary_flags):
    """
    Search best feature combination using silhouette + between-group variance.
    """
    n_features = X_scaled.shape[1]
    valid_candidates = []
    for r in range(1, 3):  # search 1~2 features
        combos = list(itertools.combinations(range(n_features), r))
        print(f"  Searching {r}-feature combinations (total {len(combos)})...")
        for idx in combos:
            idx_list = list(idx)
            X_sub = X_scaled[:, idx_list]
            labels, sil, var = evaluate_feature_combo(X_sub)
            if labels is not None:
                valid_candidates.append((idx_list, labels, sil, var))
    if not valid_candidates:
        return None, None, None, None

    sil_vals = np.array([c[2] for c in valid_candidates])
    var_vals = np.array([c[3] for c in valid_candidates])
    sil_min, sil_max = sil_vals.min(), sil_vals.max()
    var_min, var_max = var_vals.min(), var_vals.max()
    sil_norm = (sil_vals - sil_min) / (sil_max - sil_min) if sil_max > sil_min else np.zeros_like(sil_vals)
    var_norm = (var_vals - var_min) / (var_max - var_min) if var_max > var_min else np.zeros_like(var_vals)

    # Weight based on proportion of binary features in the combination
    weights = np.array([sum(binary_flags[i] for i in c[0]) / len(c[0]) for c in valid_candidates])
    scores = weights * sil_norm + (1 - weights) * var_norm
    best_idx = np.argmax(scores)
    best_combo, best_labels, best_sil, best_var = valid_candidates[best_idx]
    best_features = [feature_names[i] for i in best_combo]
    best_score = scores[best_idx]
    best_weight = weights[best_idx]
    return best_features, best_labels, best_score, (best_sil, best_var, best_weight, sil_norm[best_idx], var_norm[best_idx])

# ====================== Multi‑cluster Dynamic Sampling ======================
def select_training_set_multi_cluster(twibot_df, cresci_df, common_features, feature_importance,
                                      cresci_cluster_labels, target_ratio=(3, 4), sample_size=None,
                                      min_sample_ratio=0.75, max_tol=0.8, tol_step=0.02, init_tol=0.01):
    """
    For each cluster, perform dynamic tolerance sampling and enforce overall bot‑human balance.
    """
    if sample_size is None:
        sample_size = len(cresci_df)

    bot_ratio, human_ratio = target_ratio
    total_parts = bot_ratio + human_ratio
    n_bot_total = int(sample_size * bot_ratio / total_parts)
    n_human_total = sample_size - n_bot_total

    unique_clusters = np.unique(cresci_cluster_labels)
    total_target = len(cresci_df)

    selected_indices = []

    # Sort features by importance
    sorted_feat_names = [f for f, _ in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                         if f in common_features]

    def _dynamic_sampling_subset(target_sub, twibot_subset, common_feats, importance,
                                 n_bot, n_human, min_sample_ratio, max_tol, tol_step, init_tol):
        sorted_feats = [f for f, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)
                        if f in common_feats]
        weights = np.array([importance[f] for f in sorted_feats])
        weights = weights / (weights.sum() + 1e-6)

        target_total = n_bot + n_human
        min_needed = int(target_total * min_sample_ratio)

        current_tols = np.ones(len(sorted_feats)) * init_tol
        candidate_indices = twibot_subset.index.tolist()

        for it in range(200):
            filtered = twibot_subset.loc[candidate_indices].copy()
            mask = np.ones(len(filtered), dtype=bool)
            for i, feat in enumerate(sorted_feats):
                cresci_mean = target_sub[feat].mean()
                lower = cresci_mean * (1 - current_tols[i])
                upper = cresci_mean * (1 + current_tols[i])
                if upper - lower < 1e-3:
                    lower = cresci_mean - 1e-3
                    upper = cresci_mean + 1e-3
                mask &= (filtered[feat] >= lower) & (filtered[feat] <= upper)
            new_indices = filtered.index[mask].tolist()
            if len(new_indices) >= min_needed:
                candidate_indices = new_indices
                break
            else:
                increment = tol_step * (1 - weights)
                current_tols = np.minimum(current_tols + increment, max_tol)
                if it == 199:
                    candidate_indices = new_indices
                    break

        train_candidates = twibot_subset.loc[candidate_indices]
        n_bot_avail = sum(train_candidates['label'] == 1)
        n_human_avail = sum(train_candidates['label'] == 0)
        n_bot_sample = min(n_bot, n_bot_avail)
        n_human_sample = min(n_human, n_human_avail)
        bot_sub = train_candidates[train_candidates['label'] == 1].sample(n=n_bot_sample,
                                                                          random_state=RANDOM_STATE) if n_bot_sample > 0 else pd.DataFrame()
        human_sub = train_candidates[train_candidates['label'] == 0].sample(n=n_human_sample,
                                                                            random_state=RANDOM_STATE) if n_human_sample > 0 else pd.DataFrame()
        sampled = pd.concat([bot_sub, human_sub])
        if len(sampled) < target_total:
            remaining = target_total - len(sampled)
            avail = twibot_subset.index.difference(sampled.index)
            if len(avail) > 0:
                extra = np.random.choice(avail, min(remaining, len(avail)), replace=False)
                extra_df = twibot_subset.loc[extra]
                sampled = pd.concat([sampled, extra_df])
        return sampled

    for cid in unique_clusters:
        cluster_mask = (cresci_cluster_labels == cid)
        cluster_size = np.sum(cluster_mask)
        if cluster_size == 0:
            continue
        cluster_quota = int(sample_size * cluster_size / total_target)
        n_bot_cluster = int(cluster_quota * bot_ratio / total_parts)
        n_human_cluster = cluster_quota - n_bot_cluster

        cresci_sub = cresci_df.loc[cluster_mask]
        sampled_sub = _dynamic_sampling_subset(
            cresci_sub, twibot_df, common_features, feature_importance,
            n_bot_cluster, n_human_cluster, min_sample_ratio, max_tol, tol_step, init_tol
        )
        selected_indices.extend(sampled_sub.index.tolist())

    # Aggregate all selected samples
    train_df = twibot_df.loc[selected_indices]
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    elif len(train_df) < sample_size:
        avail = twibot_df.index.difference(train_df.index)
        if len(avail) > 0:
            extra = np.random.choice(avail, min(sample_size - len(train_df), len(avail)), replace=False)
            extra_df = twibot_df.loc[extra]
            train_df = pd.concat([train_df, extra_df])

    print(f"  Sampling completed, total samples: {len(train_df)}")
    print(f"  Bots: {sum(train_df['label'] == 1)}, Humans: {sum(train_df['label'] == 0)}")
    return train_df

# ====================== Pseudo‑alignment ======================
def align_by_scaling(train_df, target_df, features, mode='mean'):
    scaled_df = train_df.copy()
    for feat in features:
        if feat not in target_df.columns:
            continue
        if train_df[feat].dtype in ['int64', 'int32'] and train_df[feat].nunique() <= 2:
            continue
        train_vals = train_df[feat]
        if train_vals.std() == 0:
            continue
        if (train_vals < 0).any():
            continue
        if mode == 'mean':
            target_val = target_df[feat].mean()
            current_val = train_vals.mean()
        else:
            target_val = target_df[feat].median()
            current_val = train_vals.median()
        if current_val == 0:
            continue
        scale = target_val / current_val
        scaled_df[feat] = train_vals * scale
        print(f"    Scaling feature {feat:25s} : factor = {scale:.4f}")
    return scaled_df

# ====================== Bootstrap Confidence Intervals ======================
def bootstrap_ci(y_true, y_pred, y_proba=None, n_iter=1000, alpha=0.05):
    rng = np.random.RandomState(42)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    scores = {m: [] for m in metrics}
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores['accuracy'].append(accuracy_score(yt, yp))
        scores['precision'].append(precision_score(yt, yp, zero_division=0))
        scores['recall'].append(recall_score(yt, yp, zero_division=0))
        scores['f1'].append(f1_score(yt, yp, zero_division=0))
        if y_proba is not None:
            yprob = y_proba[idx]
            scores['auc'].append(roc_auc_score(yt, yprob))
    ci = {}
    for m in metrics:
        if scores[m]:
            mean = np.mean(scores[m])
            lower = np.percentile(scores[m], 100 * alpha / 2)
            upper = np.percentile(scores[m], 100 * (1 - alpha / 2))
            ci[m] = (mean, lower, upper)
        else:
            ci[m] = (np.nan, np.nan, np.nan)
    return ci

def train_and_evaluate(train_df, test_df, features, label_col='label', model_params=None):
    X_train = train_df[features].values
    y_train = train_df[label_col].values
    X_test = test_df[features].values
    y_test = test_df[label_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    model = XGBClassifier(**model_params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    ci = bootstrap_ci(y_test, y_pred, y_proba)
    return {
        'accuracy': (acc, ci['accuracy'][1], ci['accuracy'][2]),
        'precision': (prec, ci['precision'][1], ci['precision'][2]),
        'recall': (rec, ci['recall'][1], ci['recall'][2]),
        'f1': (f1, ci['f1'][1], ci['f1'][2]),
        'auc': (auc, ci['auc'][1], ci['auc'][2])
    }

# ====================== Main Pipeline ======================
def main():
    print("=" * 80)
    print("Improved Combined Version: Optimal Feature Clustering + Multi‑cluster Dynamic Sampling + Pseudo‑alignment")
    print("=" * 80)

    # 1. Load data
    print("\n[Loading data]")
    twibot_data, twibot_features = load_twibot22_from_csv()
    print(f"Twibot-22 samples: {len(twibot_data)}")
    cresci_data, cresci_features = load_cresci_from_csv()
    print(f"Cresci-2017 samples: {len(cresci_data)}")

    # Common features
    cresci_cols = set(cresci_data.columns)
    twibot_cols = set(twibot_data.columns)
    exclude = {'user_id', 'label'}
    common = sorted(list(cresci_cols.intersection(twibot_cols) - exclude))
    print(f"Common features count: {len(common)}")
    if len(common) == 0:
        raise ValueError("No common features found.")

    # 2. Search best feature combination on target data
    print("\n[Searching best feature combination on target data]")
    X_target = cresci_data[common].values
    scaler_target = StandardScaler()
    X_target_scaled = scaler_target.fit_transform(X_target)
    binary_flags = [is_binary_feature(cresci_data[feat]) for feat in common]
    best_features, best_labels, best_score, extra = search_best_combo(X_target_scaled, common, binary_flags)
    if best_features is None:
        print("No valid feature combination found, exiting.")
        return
    print(f"  Best feature combination: {best_features}, combined score: {best_score:.4f}")
    # Add cluster labels to cresci_data
    cresci_data['cluster'] = best_labels
    cluster_counts = np.bincount(best_labels)
    print(f"  Cluster 0: {cluster_counts[0]}, Cluster 1: {cluster_counts[1]}")

    # 3. Compute feature importance on source data
    print("\n[Computing feature importance]")
    sample_df = twibot_data.sample(n=min(200000, len(twibot_data)), random_state=RANDOM_STATE)
    X_temp = sample_df[common].values
    y_temp = sample_df['label'].values
    scaler_temp = StandardScaler()
    X_temp_scaled = scaler_temp.fit_transform(X_temp)
    model_temp = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                               random_state=RANDOM_STATE, objective='binary:logistic',
                               eval_metric='logloss', verbosity=0)
    model_temp.fit(X_temp_scaled, y_temp)
    importance = dict(zip(common, model_temp.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Feature importance ranking (high to low):")
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        print(f"  {i:2d}. {feat:25s} : {imp:.4f}")

    # 4. Multi‑cluster dynamic sampling
    print("\n[Multi‑cluster dynamic sampling]")
    sample_size = int(len(cresci_data) * SAMPLE_SIZE_RATIO)
    train_data = select_training_set_multi_cluster(
        twibot_data, cresci_data, common, importance,
        cresci_cluster_labels=best_labels,
        target_ratio=TARGET_RATIO,
        sample_size=sample_size,
        min_sample_ratio=MIN_SAMPLE_RATIO,
        max_tol=MAX_TOL,
        tol_step=TOL_STEP,
        init_tol=INIT_TOL
    )
    print(f"Sampled training set size: {len(train_data)}")
    print(f"Bots: {sum(train_data['label'] == 1)}, Humans: {sum(train_data['label'] == 0)}")

    # 5. Pseudo‑alignment (scaling)
    print("\n[Pseudo‑alignment] Scaling numerical features of training set (mean alignment)")
    scale_features = []
    for f in SCALE_FEATURES:
        if f in common and f in train_data.columns and f in cresci_data.columns:
            if train_data[f].dtype in ['int64', 'float64'] and train_data[f].nunique() > 2:
                scale_features.append(f)
    if scale_features:
        train_data_scaled = align_by_scaling(train_data, cresci_data, scale_features, mode='mean')
    else:
        print("  No features to scale, skipping pseudo‑alignment")
        train_data_scaled = train_data

    # Verify scaling effect
    print("\n  Verification of scaled key feature means:")
    for f in ['followers_count', 'user_age_days', 'tweet_count']:
        if f in scale_features:
            target_mean = cresci_data[f].mean()
            scaled_mean = train_data_scaled[f].mean()
            print(f"    {f:20s} target mean = {target_mean:10.2f}  scaled mean = {scaled_mean:10.2f}")

    # 6. Train and evaluate on Cresci-2017
    print("\n[Training and evaluating on Cresci-2017]")
    result = train_and_evaluate(train_data_scaled, cresci_data, common)
    print("\n=== Experimental Results (95% CI) ===")
    for metric, (mean, low, high) in result.items():
        print(f"{metric:10s} : {mean:.4f}  (95% CI: {low:.4f} - {high:.4f})")

if __name__ == "__main__":
    main()