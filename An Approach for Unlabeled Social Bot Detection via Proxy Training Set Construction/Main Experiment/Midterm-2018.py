import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ====================== Path Configuration ======================
# The script is assumed to be in a subfolder (e.g., "Main Experiment"),
# so data is in "../data" relative to this script.
DATA_DIR = "../data"
SOURCE_FILE = os.path.join(DATA_DIR, "Twibot-22（part）.csv")   # Pre-extracted source
TARGET_FILE = os.path.join(DATA_DIR, "midterm-2018.csv")        # Extracted target

# ====================== Parameters ======================
TOP_K_FEATURES = 15          # If fewer common features, use all
TRAIN_SIZE_RATIO = 5         # Training set size = target samples × 5
BALANCE_SOURCE = True        # Balance source classes during sampling
RANDOM_STATE = 42
USE_ALL_FEATURES_FOR_TRAIN = True   # Use all common features for training
# Features to scale (continuous non-binary)
SCALE_FEATURES = ['followers_count', 'following_count', 'listed_count',
                  'follower_following_ratio', 'user_age_days']

# ====================== Data Loading ======================
def load_source_data():
    """Load Twibot-22 part from unified CSV."""
    df = pd.read_csv(SOURCE_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    if 'user_age_days' in df.columns:
        df['user_age_days'] = df['user_age_days'].clip(lower=0)
    if 'user_id' in df.columns:
        df.set_index('user_id', inplace=True)
    return df

def load_target_data():
    """Load midterm-2018 from unified CSV."""
    df = pd.read_csv(TARGET_FILE, encoding='utf-8-sig')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    # Ensure all numeric features are float
    for c in df.columns:
        if c != 'label' and np.issubdtype(df[c].dtype, np.number):
            df[c] = df[c].fillna(0).astype(float)
    return df

# ====================== Global Distribution Sampling (Label-Free) ======================
def select_training_set_global(source_df, target_df, feature_importance,
                               common_features, top_k=None, sample_size_ratio=5, balance_source=True):
    """
    Sample training set based on overall target distribution (no target labels used).
    """
    if top_k is None:
        top_k = len(common_features)

    # 1. Select top features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in sorted_features[:top_k] if f in common_features]
    if len(top_features) < top_k:
        top_features = common_features[:top_k]
    print(f"  Using top-{len(top_features)} features for global distribution matching: {top_features}")

    # 2. Compute global mean and std of target (no labels!)
    target_mean = target_df[top_features].mean()
    target_std = target_df[top_features].std().replace(0, 1)

    # 3. Compute weights based on feature importance
    weights = np.array([feature_importance.get(f, 1.0) for f in top_features])
    weights = weights / (weights.sum() + 1e-6)

    # 4. Compute weighted Euclidean distance from each source sample to target global center
    source_norm = (source_df[top_features] - target_mean) / target_std
    weighted_dist = np.sqrt(((source_norm ** 2) * weights).sum(axis=1))
    source_with_dist = source_df[top_features].copy()
    source_with_dist['distance'] = weighted_dist
    source_with_dist['label'] = source_df['label'].values

    # 5. Determine sample size
    n_target = len(target_df)
    n_train = int(n_target * sample_size_ratio)
    print(f"  Target size: {n_target}, Planned training samples: {n_train}")

    # 6. Sampling with optional class balance
    if balance_source:
        n_per_class = n_train // 2
        print(f"  Class-balanced: sample {n_per_class} bots and {n_per_class} humans (sorted by distance)")
        sampled_indices = []
        for label_val, class_name in [(1, 'bot'), (0, 'human')]:
            class_data = source_with_dist[source_with_dist['label'] == label_val].sort_values('distance')
            n_sample = min(n_per_class, len(class_data))
            sampled = class_data.head(n_sample)
            sampled_indices.extend(sampled.index.tolist())
            print(f"    {class_name}: sampled {len(sampled)} (available {len(class_data)})")
    else:
        print("  Sampling without balancing (source original ratio)")
        class_counts = source_with_dist['label'].value_counts()
        ratio_bot = class_counts.get(1, 0) / len(source_with_dist)
        n_bot = int(n_train * ratio_bot)
        n_human = n_train - n_bot
        sampled_indices = []
        for label_val, n_sample in [(1, n_bot), (0, n_human)]:
            class_data = source_with_dist[source_with_dist['label'] == label_val].sort_values('distance')
            n_sample = min(n_sample, len(class_data))
            sampled = class_data.head(n_sample)
            sampled_indices.extend(sampled.index.tolist())

    # 7. Build training set
    train_df = source_df.loc[sampled_indices][common_features + ['label']].copy()
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 8. Verify distribution alignment
    print("\n  Verification of global distribution alignment (key features):")
    for feat in ['followers_count', 'is_verified', 'listed_count']:
        if feat in common_features:
            target_mean_val = target_df[feat].mean()
            actual_mean = train_df[feat].mean()
            print(f"    {feat:25s} target mean={target_mean_val:10.2f} sampled mean={actual_mean:10.2f}")

    return train_df

# ====================== Scaling Alignment ======================
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
    print("Line 2 Reproduction (Improved v4) - Adapted for midterm-2018")
    print("【Label-Free Sampling】 Using only overall target distribution, no target labels touched")
    print("=" * 80)

    # 1. Load source data
    print("\n[Loading source data (Twibot-22 part)]")
    source_df = load_source_data()
    print(f"Source samples: {len(source_df)}")
    print(f"Class distribution:\n{source_df['label'].value_counts()}")

    # 2. Load target data
    print("\n[Loading target data (midterm-2018)]")
    target_df = load_target_data()
    print(f"Target samples: {len(target_df)}")
    print(f"Class distribution (for reference only, not used in sampling):\n{target_df['label'].value_counts()}")

    # 3. Define common features (based on the unified column names)
    common_features = [
        'followers_count', 'following_count', 'listed_count', 'tweet_count',
        'follower_following_ratio', 'is_verified', 'has_url',
        'user_age_days', 'avg_tweets_per_day'
    ]
    # Keep only features present in both datasets
    common_features = [f for f in common_features if f in source_df.columns and f in target_df.columns]
    print(f"\nCommon features: {len(common_features)}")
    print("Common feature list:", common_features)

    if len(common_features) == 0:
        raise ValueError("No common features! Aborting.")

    # 4. Compute feature importance on source data
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

    # 5. Sample training set (global distribution, no target labels)
    print("\n[Sampling training set (based on overall target distribution, label-free)]")
    top_k = min(TOP_K_FEATURES, len(common_features))
    train_data = select_training_set_global(
        source_df, target_df, importance, common_features,
        top_k=top_k,
        sample_size_ratio=TRAIN_SIZE_RATIO,
        balance_source=BALANCE_SOURCE
    )
    print(f"Sampled training set size: {len(train_data)}")
    print(f"Bots: {sum(train_data['label'] == 1)}, Humans: {sum(train_data['label'] == 0)}")

    # 6. Scaling alignment (mean)
    scale_features = [f for f in SCALE_FEATURES if f in common_features and f not in ['is_verified', 'has_url']]
    print("\n[Scaling alignment (mean)]")
    train_data_scaled = align_by_scaling(train_data, target_df, scale_features, mode='mean')

    # Verify scaling
    print("\n  Verification of scaled feature means:")
    for feat in ['followers_count', 'following_count', 'listed_count']:
        if feat in scale_features:
            target_mean = target_df[feat].mean()
            scaled_mean = train_data_scaled[feat].mean()
            print(f"    {feat:25s} target mean={target_mean:10.2f} scaled mean={scaled_mean:10.2f}")

    # 7. Select features for training
    if USE_ALL_FEATURES_FOR_TRAIN:
        train_features = common_features
    else:
        train_features = common_features  # default to all

    # 8. Train and evaluate on midterm-2018
    print("\n[Training and evaluating on midterm-2018]")
    result = train_and_evaluate(train_data_scaled, target_df, train_features)
    print("\n=== Experimental Results (95% CI) ===")
    for metric, (mean, low, high) in result.items():
        print(f"{metric:10s} : {mean:.4f}  (95% CI: {low:.4f} - {high:.4f})")

if __name__ == "__main__":
    main()