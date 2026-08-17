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
DATA_DIR = "../data"
SOURCE_FILE = os.path.join(DATA_DIR, "Twibot-22（part）.csv")    # Pre-extracted source dataset
TARGET_FILE = os.path.join(DATA_DIR, "Twibot-20.csv")           # Pre-extracted target dataset

# ====================== Parameters ======================
RANDOM_STATE = 42
BALANCE_THRESHOLD = 0.20          # Minimum proportion for each cluster during clustering
SAMPLE_SIZE_RATIO = 5.0           # Sampling size = 5× target size (consistent with paper)
MIN_SAMPLE_RATIO = 0.75           # Minimum target coverage ratio for dynamic tolerance sampling
MAX_TOL = 0.8                     # Maximum tolerance
TOL_STEP = 0.02                   # Tolerance increment step
INIT_TOL = 0.01                   # Initial tolerance
TARGET_RATIO = (1, 1)             # Training set bot:human = 1:1

# Continuous non-binary features to be scaled
SCALE_FEATURES = ['followers_count', 'following_count', 'listed_count', 'tweet_count',
                  'follower_following_ratio', 'avg_tweet_length', 'avg_url_count',
                  'avg_user_mention_count', 'avg_hashtag_count', 'avg_tweets_per_day',
                  'user_age_days']

# ====================== Data Loading (from CSV) ======================
def load_source_data():
    """Load source dataset (Twibot-22 part) from CSV."""
    df = pd.read_csv(SOURCE_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    # Ensure user_age_days is non-negative
    if 'user_age_days' in df.columns:
        df['user_age_days'] = df['user_age_days'].clip(lower=0)
    # Set index to user_id if exists
    if 'user_id' in df.columns:
        df.set_index('user_id', inplace=True)
    return df

def load_target_data():
    """Load target dataset (Twibot-20) from CSV."""
    df = pd.read_csv(TARGET_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    if 'user_age_days' in df.columns:
        df['user_age_days'] = df['user_age_days'].clip(lower=0)
    if 'user_id' in df.columns:
        df.set_index('user_id', inplace=True)
    return df

# ====================== Feature Search (Section 3.2.1) ======================
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
    n_features = X_scaled.shape[1]
    valid_candidates = []
    # Force search for 2~3 feature combinations to avoid poor clustering from single binary feature
    for r in range(2, 3):
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
    weights = np.array([sum(binary_flags[i] for i in c[0]) / len(c[0]) for c in valid_candidates])
    scores = weights * sil_norm + (1 - weights) * var_norm
    best_idx = np.argmax(scores)
    best_combo, best_labels, best_sil, best_var = valid_candidates[best_idx]
    best_features = [feature_names[i] for i in best_combo]
    best_score = scores[best_idx]
    return best_features, best_labels, best_score, (best_sil, best_var)

# ====================== Multi‑cluster Dynamic Tolerance Sampling (Section 3.2.3, step 2) ======================
def select_training_set_multi_cluster(source_df, target_df, common_features, feature_importance,
                                      target_cluster_labels, target_ratio=(1,1), sample_size=None,
                                      min_sample_ratio=0.75, max_tol=0.8, tol_step=0.02, init_tol=0.01):
    """
    For each cluster, independently perform dynamic tolerance sampling,
    allocate quotas by cluster size, and enforce overall bot-human balance.
    Returns the naturally sampled training set (before global calibration).
    """
    if sample_size is None:
        sample_size = len(target_df)

    bot_ratio, human_ratio = target_ratio
    total_parts = bot_ratio + human_ratio
    n_bot_total = int(sample_size * bot_ratio / total_parts)
    n_human_total = sample_size - n_bot_total

    unique_clusters = np.unique(target_cluster_labels)
    total_target = len(target_df)

    selected_indices = []

    def _dynamic_sampling_subset(target_sub, source_subset, common_feats, importance,
                                 n_bot, n_human, min_sample_ratio, max_tol, tol_step, init_tol):
        sorted_feats = [f for f, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)
                        if f in common_feats]
        weights = np.array([importance[f] for f in sorted_feats])
        weights = weights / (weights.sum() + 1e-6)

        target_total = n_bot + n_human
        min_needed = int(target_total * min_sample_ratio)

        current_tols = np.ones(len(sorted_feats)) * init_tol
        candidate_indices = source_subset.index.tolist()

        for it in range(200):
            filtered = source_subset.loc[candidate_indices].copy()
            mask = np.ones(len(filtered), dtype=bool)
            for i, feat in enumerate(sorted_feats):
                target_mean = target_sub[feat].mean()
                lower = target_mean * (1 - current_tols[i])
                upper = target_mean * (1 + current_tols[i])
                if upper - lower < 1e-3:
                    lower = target_mean - 1e-3
                    upper = target_mean + 1e-3
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

        train_candidates = source_subset.loc[candidate_indices]
        target_center = target_sub[sorted_feats].mean()
        target_std = target_sub[sorted_feats].std().replace(0, 1)
        source_norm = (train_candidates[sorted_feats] - target_center) / target_std
        dist = np.sqrt(((source_norm ** 2) * weights).sum(axis=1))
        train_candidates = train_candidates.copy()
        train_candidates['dist'] = dist

        n_avail_bot = sum(train_candidates['label'] == 1)
        n_avail_human = sum(train_candidates['label'] == 0)
        n_bot_sample = min(n_bot, n_avail_bot)
        n_human_sample = min(n_human, n_avail_human)

        bot_candidates = train_candidates[train_candidates['label'] == 1].sort_values('dist')
        human_candidates = train_candidates[train_candidates['label'] == 0].sort_values('dist')
        selected_bot = bot_candidates.head(n_bot_sample)
        selected_human = human_candidates.head(n_human_sample)
        sampled = pd.concat([selected_bot, selected_human])

        if len(sampled) < (n_bot + n_human):
            remaining = (n_bot + n_human) - len(sampled)
            avail = source_subset.index.difference(sampled.index)
            if len(avail) > 0:
                extra = np.random.choice(avail, min(remaining, len(avail)), replace=False)
                extra_df = source_subset.loc[extra]
                sampled = pd.concat([sampled, extra_df])
        return sampled

    for cid in unique_clusters:
        cluster_mask = (target_cluster_labels == cid)
        cluster_size = np.sum(cluster_mask)
        if cluster_size == 0:
            continue
        cluster_quota = int(sample_size * cluster_size / total_target)
        n_bot_cluster = int(cluster_quota * bot_ratio / total_parts)
        n_human_cluster = cluster_quota - n_bot_cluster

        target_sub = target_df.loc[cluster_mask]
        sampled_sub = _dynamic_sampling_subset(
            target_sub, source_df, common_features, feature_importance,
            n_bot_cluster, n_human_cluster, min_sample_ratio, max_tol, tol_step, init_tol
        )
        selected_indices.extend(sampled_sub.index.tolist())

    train_df = source_df.loc[selected_indices]
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    elif len(train_df) < sample_size:
        avail = source_df.index.difference(train_df.index)
        if len(avail) > 0:
            extra = np.random.choice(avail, min(sample_size - len(train_df), len(avail)), replace=False)
            extra_df = source_df.loc[extra]
            train_df = pd.concat([train_df, extra_df])

    print(f"  Natural sampling completed, total samples: {len(train_df)}")
    print(f"  Bots: {sum(train_df['label'] == 1)}, Humans: {sum(train_df['label'] == 0)}")
    return train_df

# ====================== Global Class Ratio Calibration (Section 3.2.3, step 3) ======================
def calibrate_class_ratio(train_df, source_pool, target_df, top_k_features, target_ratio=(1,1), tolerance=0.03):
    """
    Global class ratio calibration via batch exchange to quickly reach target bot-human ratio.
    Paper logic: if bots are insufficient, add bots and remove humans, vice versa.
    """
    p_target = target_ratio[0] / (target_ratio[0] + target_ratio[1])
    cur_df = train_df.copy()
    cur_bot_ratio = sum(cur_df['label'] == 1) / len(cur_df)

    if abs(cur_bot_ratio - p_target) <= tolerance:
        print("  Initial ratio already meets target, no calibration needed")
        return cur_df

    # Compute global center of target domain
    target_center = target_df[top_k_features].mean()
    target_std = target_df[top_k_features].std().replace(0, 1)

    # Prepare source pool (excluding already selected samples)
    remaining_pool = source_pool.loc[source_pool.index.difference(cur_df.index)]

    # Compute adjustment quantity
    target_bot_count = int(len(cur_df) * p_target)
    current_bot_count = sum(cur_df['label'] == 1)
    delta = target_bot_count - current_bot_count

    if delta == 0:
        return cur_df

    if delta > 0:
        need_class = 1      # need to add bots
        remove_class = 0    # need to remove humans
    else:
        need_class = 0      # need to add humans
        remove_class = 1    # need to remove bots
        delta = -delta      # absolute value

    # Take needed class samples from remaining pool (sorted by distance)
    candidates = remaining_pool[remaining_pool['label'] == need_class]
    if len(candidates) == 0:
        print("  Warning: no available samples in pool, calibration stopped")
        return cur_df

    norm_candidates = (candidates[top_k_features] - target_center) / target_std
    dist_candidates = np.sqrt(((norm_candidates ** 2).mean(axis=1)))
    candidates = candidates.copy()
    candidates['dist'] = dist_candidates
    candidates_sorted = candidates.sort_values('dist')

    add_count = min(delta, len(candidates_sorted))
    if add_count == 0:
        return cur_df
    add_samples = candidates_sorted.head(add_count)

    # Remove opposite class samples from current training set (farthest)
    to_remove = cur_df[cur_df['label'] == remove_class]
    if len(to_remove) < add_count:
        # If not enough opposite class, remove farthest samples regardless of class
        cur_norm = (cur_df[top_k_features] - target_center) / target_std
        cur_dist = np.sqrt(((cur_norm ** 2).mean(axis=1)))
        cur_df_temp = cur_df.copy()
        cur_df_temp['dist'] = cur_dist
        remove_indices = cur_df_temp.sort_values('dist', ascending=False).head(add_count).index
    else:
        remove_norm = (to_remove[top_k_features] - target_center) / target_std
        remove_dist = np.sqrt(((remove_norm ** 2).mean(axis=1)))
        to_remove_temp = to_remove.copy()
        to_remove_temp['dist'] = remove_dist
        remove_indices = to_remove_temp.sort_values('dist', ascending=False).head(add_count).index

    # Execute exchange
    cur_df = cur_df.drop(remove_indices)
    cur_df = pd.concat([cur_df, add_samples])

    final_bot_ratio = sum(cur_df['label'] == 1) / len(cur_df)
    print(f"  After calibration, total samples: {len(cur_df)}, Bots: {sum(cur_df['label'] == 1)}, Humans: {sum(cur_df['label'] == 0)}")
    print(f"  Bot ratio: {final_bot_ratio:.3f} (target: {p_target:.3f})")
    return cur_df

# ====================== Pseudo‑alignment (Section 3.2.3, step 4) ======================
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
        # Special handling for user_age_days: clip negative values
        if feat == 'user_age_days':
            target_vals = target_df[feat].clip(lower=0)
            train_vals = train_vals.clip(lower=0)
            target_val = target_vals.mean()
            current_val = train_vals.mean()
        else:
            target_val = target_df[feat].mean()
            current_val = train_vals.mean()

        if current_val == 0:
            continue
        scale = target_val / current_val
        scaled_df[feat] = train_vals * scale
        print(f"    Scaling feature {feat:25s} : factor = {scale:.4f}")
    return scaled_df

# ====================== Evaluation ======================
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
    print("Main Pipeline (Section 3.2) - Twibot-20 Detection (Revised)")
    print("Clustering(2~3 features) + Multi-cluster Tolerance Sampling + Batch Global Calibration + Pseudo-alignment")
    print("=" * 80)

    # 1. Load data
    print("\n[Loading source data (Twibot-22 part)]")
    source_df = load_source_data()
    print(f"Source samples: {len(source_df)}")

    print("\n[Loading target data (Twibot-20)]")
    target_df = load_target_data()
    print(f"Target samples: {len(target_df)}")

    # 2. Define common features
    common_features = [
        'followers_count', 'following_count', 'listed_count', 'tweet_count',
        'follower_following_ratio', 'is_verified', 'has_url', 'has_location',
        'user_age_days', 'total_sampled_tweets', 'avg_tweet_length',
        'avg_url_count', 'avg_user_mention_count', 'avg_hashtag_count',
        'avg_tweets_per_day'
    ]
    # Keep only features present in both datasets
    common_features = [f for f in common_features if f in source_df.columns and f in target_df.columns]
    print(f"Common features count: {len(common_features)}")
    print("Common features:", common_features)

    # 3. Compute feature importance (on source data)
    print("\n[Computing feature importance]")
    sample_df = source_df.sample(n=min(200000, len(source_df)), random_state=RANDOM_STATE)
    X_temp = sample_df[common_features].values
    y_temp = sample_df['label'].values
    scaler_temp = StandardScaler()
    X_temp_scaled = scaler_temp.fit_transform(X_temp)
    model_temp = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                               random_state=RANDOM_STATE, objective='binary:logistic',
                               eval_metric='logloss', verbosity=0)
    model_temp.fit(X_temp_scaled, y_temp)
    importance = dict(zip(common_features, model_temp.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Feature importance ranking (high to low):")
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        print(f"  {i:2d}. {feat:25s} : {imp:.4f}")

    # 4. Search best feature combination on target domain (Section 3.2.1) – force 2~3 features
    print("\n[Searching best feature combination on Twibot-20 (2~3 features)]")
    X_target = target_df[common_features].values
    scaler_target = StandardScaler()
    X_target_scaled = scaler_target.fit_transform(X_target)
    binary_flags = [is_binary_feature(target_df[feat]) for feat in common_features]
    best_features, best_labels, best_score, extra = search_best_combo(X_target_scaled, common_features, binary_flags)
    if best_features is None:
        print("No valid feature combination found, exiting.")
        return
    print(f"  Best feature combination: {best_features}, combined score: {best_score:.4f}")
    target_df['cluster'] = best_labels
    cluster_counts = np.bincount(best_labels)
    print(f"  Cluster 0: {cluster_counts[0]}, Cluster 1: {cluster_counts[1]}")

    # 5. Multi-cluster dynamic tolerance sampling (Section 3.2.3, step 2)
    print("\n[Multi-cluster dynamic tolerance sampling]")
    sample_size = int(len(target_df) * SAMPLE_SIZE_RATIO)  # 5× sampling
    train_data_natural = select_training_set_multi_cluster(
        source_df, target_df, common_features, importance,
        target_cluster_labels=best_labels,
        target_ratio=TARGET_RATIO,
        sample_size=sample_size,
        min_sample_ratio=MIN_SAMPLE_RATIO,
        max_tol=MAX_TOL,
        tol_step=TOL_STEP,
        init_tol=INIT_TOL
    )
    print(f"After natural sampling, training set size: {len(train_data_natural)}")

    # 6. Global class ratio calibration (Section 3.2.3, step 3) – batch exchange
    print("\n[Global ratio calibration] Adjust to 1:1 bot-human (batch exchange)")
    top_k_for_calib = min(10, len(common_features))
    top_feats = [f for f, _ in sorted_imp[:top_k_for_calib] if f in common_features]
    train_data_calibrated = calibrate_class_ratio(
        train_data_natural, source_df, target_df,
        top_k_features=top_feats,
        target_ratio=(1,1),
        tolerance=0.03
    )

    # 7. Pseudo-alignment (Section 3.2.3, step 4)
    print("\n[Pseudo-alignment] Scaling numerical features of training set (mean alignment)")
    scale_features = []
    for f in SCALE_FEATURES:
        if f in common_features and f in train_data_calibrated.columns and f in target_df.columns:
            if train_data_calibrated[f].dtype in ['int64', 'float64'] and train_data_calibrated[f].nunique() > 2:
                scale_features.append(f)
    if scale_features:
        train_data_scaled = align_by_scaling(train_data_calibrated, target_df, scale_features, mode='mean')
    else:
        print("  No features to scale, skipping pseudo-alignment")
        train_data_scaled = train_data_calibrated

    # Verify scaling effect
    print("\n  Verification of scaled key feature means:")
    for f in ['followers_count', 'user_age_days', 'avg_tweet_length']:
        if f in scale_features:
            target_mean = target_df[f].mean()
            scaled_mean = train_data_scaled[f].mean()
            print(f"    {f:25s} target mean = {target_mean:10.2f}  scaled mean = {scaled_mean:10.2f}")

    # 8. Train and evaluate on Twibot-20
    print("\n[Training and evaluating on Twibot-20]")
    result = train_and_evaluate(train_data_scaled, target_df, common_features)
    print("\n=== Experimental Results (95% CI) ===")
    for metric, (mean, low, high) in result.items():
        print(f"{metric:10s} : {mean:.4f}  (95% CI: {low:.4f} - {high:.4f})")

if __name__ == "__main__":
    main()