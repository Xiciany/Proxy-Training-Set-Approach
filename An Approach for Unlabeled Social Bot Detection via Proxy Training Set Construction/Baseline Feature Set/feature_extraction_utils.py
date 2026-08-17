# -*- coding: utf-8 -*-
"""
Feature extraction utilities for social bot detection.
Provides reusable functions to derive new features from raw fields.
"""

import pandas as pd
import numpy as np

def compute_follower_following_ratio(df, follower_col='followers_count',
                                     following_col='following_count',
                                     new_col='follower_following_ratio'):
    """
    Compute follower/following ratio.
    """
    if follower_col in df.columns and following_col in df.columns:
        df[new_col] = df[follower_col] / (df[following_col] + 1)
    return df

def compute_avg_tweets_per_day(df, tweet_col='tweet_count',
                               age_col='user_age_days',
                               new_col='avg_tweets_per_day'):
    """
    Compute average tweets per day.
    """
    if tweet_col in df.columns and age_col in df.columns:
        df[new_col] = df[tweet_col] / (df[age_col] + 1)
    return df

def compute_description_length(df, desc_col='description', new_col='desc_length'):
    """
    Compute length of description (if description exists).
    """
    if desc_col in df.columns:
        df[new_col] = df[desc_col].fillna('').astype(str).str.len()
    return df

def compute_has_any_url(df, url_cols=['has_url', 'desc_has_url'],
                        new_col='has_any_url'):
    """
    Compute whether user has any URL in profile or description.
    """
    existing = [c for c in url_cols if c in df.columns]
    if existing:
        df[new_col] = df[existing].any(axis=1).astype(int)
    return df

def compute_special_char_ratio(df, text_col, new_col='special_char_ratio'):
    """
    Compute ratio of special characters in a text field.
    """
    if text_col in df.columns:
        # Define special characters (non-alphanumeric)
        def ratio(s):
            if not isinstance(s, str) or len(s) == 0:
                return 0.0
            special = sum(1 for c in s if not c.isalnum() and not c.isspace())
            return special / len(s)
        df[new_col] = df[text_col].fillna('').astype(str).apply(ratio)
    return df

def compute_english_ratio(df, text_col, new_col='english_ratio'):
    """
    Compute ratio of English characters (A-Z, a-z) in a text field.
    """
    if text_col in df.columns:
        def ratio(s):
            if not isinstance(s, str) or len(s) == 0:
                return 0.0
            eng = sum(1 for c in s if c.isalpha() and c.isascii())
            return eng / len(s)
        df[new_col] = df[text_col].fillna('').astype(str).apply(ratio)
    return df

def derive_features(df, feature_rules):
    """
    Apply a list of feature derivation rules.
    Each rule is a dict: {
        'func': callable,
        'kwargs': dict of keyword arguments to pass to func
    }
    Returns updated df.
    """
    for rule in feature_rules:
        func = rule.get('func')
        kwargs = rule.get('kwargs', {})
        if func is not None:
            df = func(df, **kwargs)
    return df