import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

from config import (logger, DATA_PATH, MAX_BENIGN, MAX_MALICIOUS,
                    TEST_SIZE_RATIO, VAL_SIZE_RATIO,
                    TEST_PRIOR_BENIGN, TEST_PRIOR_MALICIOUS,
                    UNDERSAMPLING_STRATEGY, UNDERSAMPLING_RATIO)


def undersample_training_data(X_train, y_train, strategy='random', ratio=1.0):
    """
    Apply undersampling to balance training data
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features (sql_query, injection_type)
    y_train : Series
        Training labels
    strategy : str
        Undersampling strategy: 'random', 'tomek', 'nearmiss', or 'hybrid'
    ratio : float
        Ratio of minority to majority class (1.0 = balanced, 0.5 = 2:1 majority)
    
    Returns:
    --------
    X_resampled, y_resampled : Resampled training data
    """
    
    logger.info(f"Original class distribution: {Counter(y_train)}")
    
    try:
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        logger.warning("imbalanced-learn not installed. Using simple random undersampling.")
        strategy = 'simple_random'
    
    # Combine features for sampling
    X_indices = X_train.index.values.reshape(-1, 1)
    
    if strategy == 'simple_random' or strategy == 'random':
        # Simple random undersampling
        minority_class = y_train.value_counts().idxmin()
        majority_class = y_train.value_counts().idxmax()
        
        n_minority = (y_train == minority_class).sum()
        n_majority_target = int(n_minority / ratio)
        
        minority_indices = X_train[y_train == minority_class].index
        majority_indices = X_train[y_train == majority_class].index
        
        if n_majority_target < len(majority_indices):
            sampled_majority = np.random.choice(majority_indices, n_majority_target, replace=False)
            selected_indices = np.concatenate([minority_indices, sampled_majority])
        else:
            selected_indices = X_train.index
        
        X_resampled = X_train.loc[selected_indices].reset_index(drop=True)
        y_resampled = y_train.loc[selected_indices].reset_index(drop=True)
        
    elif strategy == 'tomek':
        # Remove Tomek links (borderline examples)
        sampler = TomekLinks()
        X_resampled_idx, y_resampled = sampler.fit_resample(X_indices, y_train)
        selected_indices = X_resampled_idx.flatten()
        X_resampled = X_train.iloc[selected_indices].reset_index(drop=True)
        y_resampled = pd.Series(y_resampled, name='target').reset_index(drop=True)
        
    elif strategy == 'nearmiss':
        # Select samples based on nearest neighbors
        sampler = NearMiss(version=1, n_neighbors=3)
        X_resampled_idx, y_resampled = sampler.fit_resample(X_indices, y_train)
        selected_indices = X_resampled_idx.flatten()
        X_resampled = X_train.iloc[selected_indices].reset_index(drop=True)
        y_resampled = pd.Series(y_resampled, name='target').reset_index(drop=True)
        
    elif strategy == 'hybrid':
        # Combine Tomek links removal with random undersampling
        sampler = ImbPipeline([
            ('tomek', TomekLinks()),
            ('random', RandomUnderSampler(sampling_strategy=ratio, random_state=42))
        ])
        X_resampled_idx, y_resampled = sampler.fit_resample(X_indices, y_train)
        selected_indices = X_resampled_idx.flatten()
        X_resampled = X_train.iloc[selected_indices].reset_index(drop=True)
        y_resampled = pd.Series(y_resampled, name='target').reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.info(f"Resampled class distribution: {Counter(y_resampled)}")
    logger.info(f"Reduction: {len(X_train)} -> {len(X_resampled)} samples "
               f"({100 * len(X_resampled) / len(X_train):.1f}%)")
    
    return X_resampled, y_resampled


def load_and_split_data(undersample_strategy=None, undersample_ratio=None):
    """
    Enhanced data loading with configurable undersampling
    
    This function extends the original load_and_split_data() with optional
    undersampling of the training set to address class imbalance.
    
    Parameters:
    -----------
    undersample_strategy : str or None
        Undersampling method (uses config default if None)
    undersample_ratio : float or None
        Ratio for undersampling (uses config default if None)
    """
    
    # Use config defaults if not specified
    if undersample_strategy is None:
        undersample_strategy = UNDERSAMPLING_STRATEGY
    if undersample_ratio is None:
        undersample_ratio = UNDERSAMPLING_RATIO
    
    logger.info("Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATA_PATH}")
        raise
    
    df['target'] = df['vulnerability_status'].map({'Yes': 1, 'No': 0})
    
    benign_df = df[df['target'] == 0]
    malicious_df = df[df['target'] == 1]
    
    # Initial sampling as before
    if len(benign_df) > MAX_BENIGN:
        benign_df = benign_df.sample(MAX_BENIGN, random_state=42)
    if len(malicious_df) > MAX_MALICIOUS:
        malicious_df = malicious_df.sample(MAX_MALICIOUS, random_state=42)
    
    sampled_df = (pd.concat([benign_df, malicious_df])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True))
    
    logger.info(f"Initial sampled dataset: {len(sampled_df)} rows "
               f"(benign={len(benign_df)}, malicious={len(malicious_df)})")
    
    X = sampled_df[['sql_query', 'injection_type']]
    y = sampled_df['target']
    
    # Split into train/val/test
    X_trainval, X_test_pool, y_trainval, y_test_pool = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, stratify=y, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE_RATIO, stratify=y_trainval, random_state=42
    )
    
    # Apply undersampling ONLY to training set
    if undersample_strategy is not None and undersample_strategy != 'none':
        logger.info(f"Applying {undersample_strategy} undersampling with ratio {undersample_ratio}...")
        X_train, y_train = undersample_training_data(
            X_train, y_train, 
            strategy=undersample_strategy,
            ratio=undersample_ratio
        )
    
    # Adjust test set ratio (keep this as before)
    ratio = TEST_PRIOR_BENIGN / TEST_PRIOR_MALICIOUS
    test_df = pd.concat([X_test_pool, y_test_pool], axis=1)
    test_benign = test_df[test_df['target'] == 0]
    test_malicious = test_df[test_df['target'] == 1]
    
    required_benign = int(len(test_malicious) * ratio)
    
    if required_benign <= len(test_benign):
        test_benign_sampled = test_benign.sample(required_benign, random_state=42)
        final_test_df = pd.concat([test_benign_sampled, test_malicious])
    else:
        required_malicious = int(len(test_benign) / ratio)
        test_malicious_sampled = test_malicious.sample(required_malicious, random_state=42)
        final_test_df = pd.concat([test_benign, test_malicious_sampled])
    
    final_test_df = final_test_df.sample(frac=1, random_state=42)
    X_test_final = final_test_df[['sql_query', 'injection_type']]
    y_test_final = final_test_df['target']
    
    def _fmt(y):
        return (f"{len(y)} rows [malicious={y.sum()} ({100*y.mean():.1f}%) "
               f"benign={len(y)-y.sum()}]")
    
    logger.info(f"Train (after undersampling): {_fmt(y_train)}")
    logger.info(f"Val : {_fmt(y_val)}")
    logger.info(f"Test: {_fmt(y_test_final)}")
    
    return X_train, X_val, X_test_final, y_train, y_val, y_test_final
