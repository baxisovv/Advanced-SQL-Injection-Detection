"""
TF-IDF Only Baseline Experiment

This script trains and tests models using ONLY TF-IDF features (no structural features).
Purpose: Establish true baseline to show the contribution of structural features.

Usage:
    python tfidf.py

Output:
    - Models saved to ./models_out/
    - Results saved to ./figures/tfidf_only/
    - Comparison with other experiments
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_fscore_support, roc_auc_score)
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = 'rbsqli_dataset.csv'
MODEL_DIR = './models_out/'
OUTPUT_DIR = './figures/tfidf_only/'
RANDOM_STATE = 42

# Sampling parameters
MAX_BENIGN = 500000
MAX_MALICIOUS = 100000
TEST_SIZE_RATIO = 0.20
VAL_SIZE_RATIO = 0.20
TEST_PRIOR_BENIGN = 0.95
TEST_PRIOR_MALICIOUS = 0.05

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def load_and_prepare_data():
    """Load dataset and prepare train/val/test splits"""
    print_header("LOADING DATASET (TF-IDF ONLY BASELINE)")
    
    print(f"Loading from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['target'] = df['vulnerability_status'].map({'Yes': 1, 'No': 0})
    
    print(f"✓ Loaded {len(df)} total samples")
    
    # Sample to manageable size
    benign_df = df[df['target'] == 0]
    malicious_df = df[df['target'] == 1]
    
    if len(benign_df) > MAX_BENIGN:
        benign_df = benign_df.sample(MAX_BENIGN, random_state=RANDOM_STATE)
    if len(malicious_df) > MAX_MALICIOUS:
        malicious_df = malicious_df.sample(MAX_MALICIOUS, random_state=RANDOM_STATE)
    
    sampled_df = pd.concat([benign_df, malicious_df]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"Sampled dataset: {len(sampled_df)} rows (benign={len(benign_df)}, malicious={len(malicious_df)})")
    
    # Prepare features and labels
    X = sampled_df[['sql_query']]
    y = sampled_df['target']
    
    # Split into train+val and test
    X_trainval, X_test_pool, y_trainval, y_test_pool = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    
    # Adjust test set ratio
    test_df = pd.concat([X_test_pool, y_test_pool], axis=1)
    test_benign = test_df[test_df['target'] == 0]
    test_malicious = test_df[test_df['target'] == 1]
    
    ratio = TEST_PRIOR_BENIGN / TEST_PRIOR_MALICIOUS
    required_benign = int(len(test_malicious) * ratio)
    
    if required_benign <= len(test_benign):
        test_benign_sampled = test_benign.sample(required_benign, random_state=RANDOM_STATE)
        final_test_df = pd.concat([test_benign_sampled, test_malicious])
    else:
        required_malicious = int(len(test_benign) / ratio)
        test_malicious_sampled = test_malicious.sample(required_malicious, random_state=RANDOM_STATE)
        final_test_df = pd.concat([test_benign, test_malicious_sampled])
    
    final_test_df = final_test_df.sample(frac=1, random_state=RANDOM_STATE)
    X_test = final_test_df[['sql_query']]
    y_test = final_test_df['target']
    
    # Split trainval into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_RATIO/(1-TEST_SIZE_RATIO), 
        stratify=y_trainval, random_state=RANDOM_STATE
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    print(f"  Test:  {len(y_test)} samples (benign={len(y_test)-y_test.sum()}, malicious={y_test.sum()})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_undersampling(X_train, y_train):
    """Apply random undersampling for balanced training"""
    print_header("APPLYING UNDERSAMPLING")
    
    print("Original class distribution:", Counter(y_train))
    
    undersampler = RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)
    
    # Create temporary DataFrame for undersampling
    X_temp = pd.DataFrame({'sql_query': X_train['sql_query'].values, 'target': y_train})
    
    X_resampled, y_resampled = undersampler.fit_resample(X_temp[['sql_query']], X_temp['target'])
    
    print("Resampled class distribution:", Counter(y_resampled))
    print(f"Reduction: {len(y_train)} → {len(y_resampled)} samples ({100*len(y_resampled)/len(y_train):.1f}%)")
    
    return X_resampled, y_resampled


def extract_tfidf_features(X_train, X_val, X_test):
    """Extract ONLY TF-IDF features (no structural features)"""
    print_header("EXTRACTING TF-IDF FEATURES ONLY")
    
    print("Fitting TF-IDF vectorizer on training data...")
    print("  max_features: 150")
    print("  analyzer: char")
    print("  ngram_range: (3, 5)")
    
    vectorizer = TfidfVectorizer(
        max_features=150,
        analyzer='char',
        ngram_range=(3, 5),
        lowercase=True
    )
    
    # Fit on training data
    X_train_tfidf = vectorizer.fit_transform(X_train['sql_query'])
    print(f"✓ Training TF-IDF shape: {X_train_tfidf.shape}")
    
    # Transform val and test
    X_val_tfidf = vectorizer.transform(X_val['sql_query'])
    X_test_tfidf = vectorizer.transform(X_test['sql_query'])
    
    print(f"✓ Validation TF-IDF shape: {X_val_tfidf.shape}")
    print(f"✓ Test TF-IDF shape: {X_test_tfidf.shape}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_only_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✓ Vectorizer saved: {vectorizer_path}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def get_models():
    """Get model instances"""
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
    }
    return models


def cross_validate_models(X_train, y_train, models):
    """Perform 5-fold cross-validation"""
    print_header("5-FOLD CROSS-VALIDATION")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    
    for name, model in models.items():
        print(f"  [{name:20s}] ", end='', flush=True)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    return cv_results


def train_models(X_train, y_train, models):
    """Train all models"""
    print_header("TRAINING MODELS")
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...", end='', flush=True)
        model.fit(X_train, y_train)
        
        trained_models[name] = model
        print(" ✓")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{name}_tfidf_only_model.pkl')
        joblib.dump(model, model_path)
    
    print(f"\n✓ All models saved to {MODEL_DIR}")
    return trained_models


def find_optimal_threshold(model, X_val, y_val, model_name):
    """Find optimal threshold on validation set"""
        # Standard classifiers
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 100)
        
    best_f1 = 0
    best_threshold = 0.5
        
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
    return best_threshold, best_f1


def evaluate_models(models, X_val, X_test, y_val, y_test):
    """Evaluate models on validation and test sets"""
    print_header("EVALUATING MODELS")
    
    results = {}
    
    print("\nPhase 1: Finding optimal thresholds on validation set...")
    thresholds = {}
    
    for name, model in models.items():
        threshold, val_f1 = find_optimal_threshold(model, X_val, y_val, name)
        thresholds[name] = threshold
        print(f"  [{name:20s}] Val F1={val_f1:.4f}, threshold={threshold:.4f}")
    
    print("\nPhase 2: Evaluating on test set with locked thresholds...")
    
    for name, model in models.items():
        threshold = thresholds[name]
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results[name] = {
            'threshold': threshold,
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'confusion_matrix': cm
        }
        
        print(f"  [{name:20s}] Acc={accuracy:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f} FPR={fpr:.4f}")
    
    return results


def generate_visualizations(results, cv_results):
    """Generate comparison visualizations"""
    print_header("GENERATING VISUALIZATIONS")
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.keys())
    metrics = {
        'F1 Score': [results[m]['f1'] for m in models],
        'Precision': [results[m]['precision'] for m in models],
        'Recall': [results[m]['recall'] for m in models],
        'Accuracy': [results[m]['accuracy'] for m in models]
    }
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        colors = ['green' if v >= 0.95 else 'orange' if v >= 0.90 else 'red' for v in values]
        bars = ax.barh(models, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} (TF-IDF Only)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=10)
        
        ax.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle('TF-IDF Only Baseline - Test Set Performance', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tfidf_only_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tfidf_only_performance.png")
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'])
        axes[idx].set_title(f'{name}\nF1={res["f1"]:.4f}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - TF-IDF Only', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tfidf_only_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tfidf_only_confusion_matrices.png")
    
    # 3. CV vs Test comparison
    if cv_results:
        cv_models = list(cv_results.keys())
        cv_means = [cv_results[m]['mean'] for m in cv_models]
        test_f1s = [results[m]['f1'] for m in cv_models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(cv_models))
        width = 0.35
        
        ax.bar(x - width/2, cv_means, width, label='CV Mean F1', alpha=0.8, color='blue')
        ax.bar(x + width/2, test_f1s, width, label='Test F1', alpha=0.8, color='green')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Cross-Validation vs Test Performance (TF-IDF Only)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cv_models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'tfidf_only_cv_vs_test.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: tfidf_only_cv_vs_test.png")


def save_results_summary(results, cv_results):
    """Save results to CSV"""
    print_header("SAVING RESULTS SUMMARY")
    
    # Test results
    test_data = []
    for name, res in results.items():
        test_data.append({
            'Model': name,
            'Threshold': res['threshold'],
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1'],
            'FPR': res['fpr']
        })
    
    test_df = pd.DataFrame(test_data)
    test_df = test_df.sort_values('F1', ascending=False)
    
    csv_path = os.path.join(OUTPUT_DIR, 'tfidf_only_results.csv')
    test_df.to_csv(csv_path, index=False)
    print(f"✓ Test results saved: {csv_path}")
    
    # CV results
    if cv_results:
        cv_data = []
        for name, res in cv_results.items():
            cv_data.append({
                'Model': name,
                'CV_Mean_F1': res['mean'],
                'CV_Std_F1': res['std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        cv_csv_path = os.path.join(OUTPUT_DIR, 'tfidf_only_cv_results.csv')
        cv_df.to_csv(cv_csv_path, index=False)
        print(f"✓ CV results saved: {cv_csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (TF-IDF ONLY)")
    print("="*80)
    print("\nTest Set Performance:")
    print(test_df.to_string(index=False))
    
    best_model = test_df.iloc[0]
    print("\n" + "="*80)
    print("BEST MODEL (TF-IDF Only):")
    print("="*80)
    print(f"  Model: {best_model['Model']}")
    print(f"  F1 Score: {best_model['F1']:.4f}")
    print(f"  Precision: {best_model['Precision']:.4f}")
    print(f"  Recall: {best_model['Recall']:.4f}")
    print(f"  FPR: {best_model['FPR']:.4f}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("TF-IDF ONLY BASELINE EXPERIMENT")
    print("="*80)
    print("\nThis establishes a true baseline using ONLY TF-IDF features.")
    print("No structural features are used in this experiment.")
    
    try:
        # 1. Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
        
        # 2. Apply undersampling
        X_train_resampled, y_train_resampled = apply_undersampling(X_train, y_train)
        
        # 3. Extract TF-IDF features ONLY
        X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(
            X_train_resampled, X_val, X_test
        )
        
        # 4. Get models
        models = get_models()
        
        # 5. Cross-validation
        cv_results = cross_validate_models(X_train_tfidf, y_train_resampled, models)
        
        # 6. Train models
        trained_models = train_models(X_train_tfidf, y_train_resampled, models)
        
        # 7. Evaluate
        results = evaluate_models(trained_models, X_val_tfidf, X_test_tfidf, y_val, y_test)
        
        # 8. Generate visualizations
        generate_visualizations(results, cv_results)
        
        # 9. Save results
        save_results_summary(results, cv_results)
        
        print("\n" + "="*80)
        print("TF-IDF ONLY BASELINE COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print("  - tfidf_only_performance.png")
        print("  - tfidf_only_confusion_matrices.png")
        print("  - tfidf_only_cv_vs_test.png")
        print("  - tfidf_only_results.csv")
        print("  - tfidf_only_cv_results.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
