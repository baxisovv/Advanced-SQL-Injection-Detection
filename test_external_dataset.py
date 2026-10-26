"""
Multi-Dataset External Testing for SQL Injection Detection

Tests trained models on MULTIPLE external datasets to validate generalization.
Handles different encodings (UTF-8, UTF-16, etc.) automatically.

Datasets to test:
- test.csv
- test1.csv
- test2.csv
- test3.csv
- test4.csv

All datasets should have: Query (or similar) and Label (0=benign, 1=malicious)
Script filters to Label==1 (malicious only) to test detection capability.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import chardet

# Configuration
MODEL_DIR = './models_out/'
OUTPUT_DIR = './figures/multi_dataset_test/'
EXPERIMENT_TAG = 'best_with_ensembles'  # Which experiment to test

# List of test datasets
TEST_DATASETS = [
    'test.csv',
    'test1.csv',
    'test2.csv',
    'test3.csv',
    'test4.csv'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


def detect_encoding(filepath):
    """Detect file encoding automatically with smart fallback"""
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"  Detected encoding: {encoding} (confidence: {confidence:.2%})")
            
            # If confidence is low or encoding is cp1250 (common false positive), try UTF-8 first
            if confidence < 0.7 or encoding in ['cp1250', 'windows-1250']:
                print(f"  Low confidence or cp1250 detected - will try UTF-8 first")
                return ['utf-8', encoding]  # Return list to try multiple
            
            return encoding
    except Exception as e:
        print(f"  Warning: Could not detect encoding ({e}). Trying UTF-8...")
        return 'utf-8'


def load_dataset_safe(filepath, malicious_only=True):
    """
    Safely load dataset with automatic encoding detection
    
    Handles:
    - Different encodings (UTF-8, UTF-16, etc.)
    - Different column names (Query/query, Label/label)
    - Different separators (comma, tab, semicolon)
    """
    print(f"\nLoading: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return None
    
    # Try to detect encoding
    encoding_to_try = detect_encoding(filepath)
    
    # Make it a list if it's not already
    if not isinstance(encoding_to_try, list):
        encoding_to_try = [encoding_to_try]
    
    # Try different separators and encodings
    separators = [',', ';', '\t']
    df = None
    
    for encoding in encoding_to_try:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep, on_bad_lines='skip')
                if df.shape[1] >= 2:  # Should have at least 2 columns
                    print(f"  ✓ Loaded with encoding '{encoding}' and separator '{sep}': {len(df)} rows, {df.shape[1]} columns")
                    break
            except Exception as e:
                continue
        
        if df is not None and not df.empty:
            break  # Successfully loaded
    
    if df is None or df.empty:
        print(f"  ✗ Failed to load dataset with any encoding/separator combination")
        return None
    
    # Normalize column names (handle different naming conventions)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Map common column name variations
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        # Query-like columns
        if 'query' in col_lower or 'sql' in col_lower or 'payload' in col_lower or 'sentence' in col_lower or 'text' in col_lower:
            column_mapping[col] = 'Query'
        # Label-like columns
        elif 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
            column_mapping[col] = 'Label'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"  ✓ Mapped columns: {column_mapping}")
    
    # Validate required columns
    if 'Query' not in df.columns or 'Label' not in df.columns:
        print(f"  ✗ Missing required columns. Found: {df.columns.tolist()}")
        print(f"     Expected: 'Query' and 'Label' (or similar)")
        return None
    
    # Convert Label to numeric if needed
    if df['Label'].dtype == 'object':
        # Try to map common string labels
        label_map = {
            'malicious': 1, 'benign': 0,
            'attack': 1, 'normal': 0,
            'yes': 1, 'no': 0,
            '1': 1, '0': 0,
            'true': 1, 'false': 0
        }
        df['Label'] = df['Label'].astype(str).str.lower().str.strip().map(label_map)
    
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df = df.dropna(subset=['Label'])  # Remove rows with invalid labels
    df['Label'] = df['Label'].astype(int)
    
    # Show class distribution
    print(f"  Class distribution:")
    print(f"    Benign (0):    {(df['Label'] == 0).sum()} ({100*(df['Label'] == 0).sum()/len(df):.1f}%)")
    print(f"    Malicious (1): {(df['Label'] == 1).sum()} ({100*(df['Label'] == 1).sum()/len(df):.1f}%)")
    
    # Filter to malicious only
    if malicious_only:
        df_filtered = df[df['Label'] == 1].copy()
        print(f"  ✓ Filtered to MALICIOUS ONLY: {len(df_filtered)} samples")
        
        if len(df_filtered) == 0:
            print(f"  ✗ No malicious samples found!")
            return None
        
        # Add fake injection_type for compatibility
        df_filtered['injection_type'] = 'external_attack'
        return df_filtered
    
    return df


def load_all_datasets():
    """Load all test datasets"""
    print("="*80)
    print("LOADING ALL TEST DATASETS")
    print("="*80)
    
    datasets = {}
    for dataset_file in TEST_DATASETS:
        df = load_dataset_safe(dataset_file, malicious_only=True)
        if df is not None:
            dataset_name = dataset_file.replace('.csv', '')
            datasets[dataset_name] = df
        else:
            print(f"⚠️  Skipping {dataset_file} (could not load)")
    
    print(f"\n✓ Successfully loaded {len(datasets)} / {len(TEST_DATASETS)} datasets")
    return datasets


def load_models_and_preprocessor(experiment_tag):
    """Load trained models and preprocessor"""
    print("\n" + "="*80)
    print(f"LOADING TRAINED MODELS (Experiment: {experiment_tag})")
    print("="*80)
    
    # Load preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, f'preprocessor_{experiment_tag}.pkl')
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    
    preprocessor = joblib.load(preprocessor_path)
    print(f"✓ Loaded preprocessor: {preprocessor_path}")
    
    # Load models
    models = {}
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(f'_{experiment_tag}_model.pkl')]
    
    for model_file in model_files:
        model_name = model_file.replace(f'_{experiment_tag}_model.pkl', '')
        model_path = os.path.join(MODEL_DIR, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"✓ Loaded model: {model_name}")
    
    print(f"\nTotal models loaded: {len(models)}")
    return preprocessor, models


def test_single_dataset(preprocessor, models, df_test, dataset_name):
    """Test all models on a single dataset"""
    print(f"\n{'='*80}")
    print(f"TESTING: {dataset_name}")
    print(f"{'='*80}")
    
    # Prepare data
    X_test = df_test[['Query', 'injection_type']].copy()
    X_test = X_test.rename(columns={'Query': 'sql_query'})
    y_test = df_test['Label'].values
    
    # Preprocess
    print(f"Preprocessing {len(df_test)} queries...")
    try:
        X_test_processed = preprocessor.transform(X_test)
        print(f"✓ Feature matrix shape: {X_test_processed.shape}")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return None
    
    # Test each model
    results = {}
    
    for model_name, model in models.items():
        try:
            # Get predictions
            if model_name == 'IsolationForest':
                predictions_raw = model.predict(X_test_processed)
                predictions = np.where(predictions_raw == -1, 1, 0)
                scores = model.decision_function(X_test_processed)
                probs = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            else:
                predictions = model.predict(X_test_processed)
                probs = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics (all queries are malicious, so detection = correct predictions)
            detection_rate = (predictions == 1).sum() / len(predictions)
            avg_confidence = probs.mean()
            detected = (predictions == 1).sum()
            missed = (predictions == 0).sum()
            
            results[model_name] = {
                'predictions': predictions,
                'probabilities': probs,
                'detection_rate': detection_rate,
                'detected': detected,
                'missed': missed,
                'avg_confidence': avg_confidence
            }
            
            print(f"  [{model_name:20s}] Detection: {detection_rate:>6.1%}  Avg Conf: {avg_confidence:.3f}  Missed: {missed}")
            
        except Exception as e:
            print(f"  [{model_name:20s}] ✗ Error: {e}")
            continue
    
    return results


def test_all_datasets(preprocessor, models, datasets):
    """Test all models on all datasets"""
    print("\n" + "="*80)
    print("TESTING ALL MODELS ON ALL DATASETS")
    print("="*80)
    
    all_results = {}
    
    for dataset_name, df_test in datasets.items():
        results = test_single_dataset(preprocessor, models, df_test, dataset_name)
        if results:
            all_results[dataset_name] = results
    
    return all_results


def generate_summary_table(all_results):
    """Generate comprehensive summary table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DETECTION SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for dataset_name, results in all_results.items():
        for model_name, res in results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Detection_Rate': res['detection_rate'],
                'Detected': res['detected'],
                'Missed': res['missed'],
                'Total': res['detected'] + res['missed'],
                'Avg_Confidence': res['avg_confidence']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary
    print("\n" + summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'multi_dataset_summary.csv'), index=False)
    print(f"\n✓ Summary saved to: {OUTPUT_DIR}multi_dataset_summary.csv")
    
    return summary_df


def plot_detection_heatmap(summary_df):
    """Plot heatmap of detection rates across datasets and models"""
    pivot_data = summary_df.pivot(index='Model', columns='Dataset', values='Detection_Rate')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=70, vmax=100, cbar_kws={'label': 'Detection Rate (%)'})
    plt.title('Detection Rates Across All Datasets\n(All queries are malicious - higher is better)', 
             fontsize=16, fontweight='bold')
    plt.ylabel('Model', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'detection_heatmap.png'), dpi=300)
    plt.close()
    print("✓ Saved: detection_heatmap.png")


def plot_dataset_comparison(summary_df):
    """Compare detection rates across datasets for each model"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Grouped bar chart
    datasets = summary_df['Dataset'].unique()
    models = summary_df['Model'].unique()
    
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = summary_df[summary_df['Model'] == model]
        values = [model_data[model_data['Dataset'] == ds]['Detection_Rate'].values[0] * 100 
                 if ds in model_data['Dataset'].values else 0 
                 for ds in datasets]
        axes[0].bar(x + i * width, values, width, label=model, alpha=0.8)
    
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Detection Rate (%)', fontsize=12)
    axes[0].set_title('Detection Rate by Dataset and Model', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(datasets)
    axes[0].legend(title='Model', fontsize=10)
    axes[0].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
    axes[0].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Threshold')
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].set_ylim(0, 105)
    
    # Plot 2: Average detection per model across all datasets
    avg_detection = summary_df.groupby('Model')['Detection_Rate'].mean().sort_values(ascending=True)
    colors = ['green' if v >= 0.95 else 'orange' if v >= 0.90 else 'red' for v in avg_detection]
    
    axes[1].barh(avg_detection.index, avg_detection.values * 100, color=colors, alpha=0.7)
    axes[1].axvline(x=95, color='green', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].axvline(x=90, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_xlabel('Average Detection Rate (%)', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    axes[1].set_title('Average Detection Across All 5 Datasets', fontsize=14, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(avg_detection.values):
        axes[1].text(v * 100 + 1, i, f'{v*100:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_comparison.png'), dpi=300)
    plt.close()
    print("✓ Saved: dataset_comparison.png")


def plot_confidence_comparison(all_results):
    """Plot confidence distributions across datasets"""
    n_datasets = len(all_results)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, (dataset_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        for model_name, res in results.items():
            if model_name == 'IsolationForest':
                continue
            ax.hist(res['probabilities'], bins=30, alpha=0.5, label=model_name, edgecolor='black')
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{dataset_name}\n(All malicious - scores should be high)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confidence Score Distributions Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_comparison.png'), dpi=300)
    plt.close()
    print("✓ Saved: confidence_comparison.png")


def plot_model_consistency(summary_df):
    """Analyze model consistency across datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Standard deviation of detection rates per model
    model_std = summary_df.groupby('Model')['Detection_Rate'].std().sort_values(ascending=True)
    colors = ['green' if v < 0.05 else 'orange' if v < 0.10 else 'red' for v in model_std]
    
    axes[0].barh(model_std.index, model_std.values * 100, color=colors, alpha=0.7)
    axes[0].axvline(x=5, color='green', linestyle='--', alpha=0.5, label='<5% (Excellent)')
    axes[0].axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='<10% (Good)')
    axes[0].set_xlabel('Standard Deviation (%)', fontsize=12)
    axes[0].set_title('Model Consistency Across Datasets\n(Lower = More Consistent)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, axis='x', alpha=0.3)
    
    # Plot 2: Min vs Max detection rate per model
    model_stats = summary_df.groupby('Model')['Detection_Rate'].agg(['min', 'max'])
    x = np.arange(len(model_stats))
    
    axes[1].barh(x, model_stats['max'] * 100, alpha=0.7, label='Best Dataset', color='green')
    axes[1].barh(x, model_stats['min'] * 100, alpha=0.7, label='Worst Dataset', color='red')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(model_stats.index)
    axes[1].set_xlabel('Detection Rate (%)', fontsize=12)
    axes[1].set_title('Best vs Worst Dataset Performance\n(Smaller gap = Better Generalization)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_consistency.png'), dpi=300)
    plt.close()
    print("✓ Saved: model_consistency.png")


def analyze_overfitting(summary_df):
    """Comprehensive overfitting analysis"""
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS - MULTI-DATASET VALIDATION")
    print("="*80)
    
    # Calculate average detection per model
    avg_by_model = summary_df.groupby('Model')['Detection_Rate'].mean().sort_values(ascending=False)
    std_by_model = summary_df.groupby('Model')['Detection_Rate'].std()
    
    print("\nAverage Detection Rate Across All 5 Datasets:")
    print("-" * 60)
    for model in avg_by_model.index:
        avg = avg_by_model[model]
        std = std_by_model[model]
        min_det = summary_df[summary_df['Model'] == model]['Detection_Rate'].min()
        max_det = summary_df[summary_df['Model'] == model]['Detection_Rate'].max()
        
        print(f"{model:20s}: {avg*100:>6.2f}% ± {std*100:>4.2f}%  "
              f"(range: {min_det*100:.1f}% - {max_det*100:.1f}%)")
    
    # Overall assessment
    overall_avg = avg_by_model.mean()
    overall_std = std_by_model.mean()
    
    print("\n" + "="*80)
    print("FINAL VERDICT:")
    print("="*80)
    
    if overall_avg >= 0.95 and overall_std < 0.05:
        print("✅ EXCELLENT GENERALIZATION!")
        print(f"   Average: {overall_avg*100:.1f}% across 5 diverse datasets")
        print(f"   Consistency: ±{overall_std*100:.1f}% (very stable)")
        print("   → Models are NOT overfitting to RBSQLi dataset")
        print("   → Results are LEGITIMATE and production-ready")
    elif overall_avg >= 0.90 and overall_std < 0.10:
        print("✓ GOOD GENERALIZATION")
        print(f"   Average: {overall_avg*100:.1f}% across 5 datasets")
        print(f"   Consistency: ±{overall_std*100:.1f}% (reasonably stable)")
        print("   → Minor dataset-specific effects")
        print("   → Acceptable for production with monitoring")
    elif overall_avg >= 0.85:
        print("⚠️  MODERATE PERFORMANCE")
        print(f"   Average: {overall_avg*100:.1f}% across 5 datasets")
        print(f"   Consistency: ±{overall_std*100:.1f}%")
        print("   → Some generalization issues")
        print("   → Consider ensemble methods or more training data")
    else:
        print("❌ SIGNIFICANT OVERFITTING DETECTED")
        print(f"   Average: {overall_avg*100:.1f}% across 5 datasets")
        print(f"   Drops significantly from training (99.9%+)")
        print("   → Models memorized RBSQLi patterns")
        print("   → Require retraining with diverse data")
    
    # Save analysis
    analysis_df = pd.DataFrame({
        'Model': avg_by_model.index,
        'Avg_Detection': avg_by_model.values,
        'Std_Detection': [std_by_model[m] for m in avg_by_model.index],
        'Min_Detection': [summary_df[summary_df['Model'] == m]['Detection_Rate'].min() for m in avg_by_model.index],
        'Max_Detection': [summary_df[summary_df['Model'] == m]['Detection_Rate'].max() for m in avg_by_model.index]
    })
    
    analysis_df.to_csv(os.path.join(OUTPUT_DIR, 'overfitting_analysis.csv'), index=False)
    print(f"\n✓ Analysis saved to: {OUTPUT_DIR}overfitting_analysis.csv")


def main():
    """Main execution"""
    print("="*80)
    print("MULTI-DATASET EXTERNAL TESTING FOR SQL INJECTION DETECTION")
    print("="*80)
    print(f"\nExperiment: {EXPERIMENT_TAG}")
    print(f"Testing on {len(TEST_DATASETS)} datasets")
    print(f"Output: {OUTPUT_DIR}")
    
    try:
        # Step 1: Load all datasets
        datasets = load_all_datasets()
        
        if not datasets:
            print("\n❌ No datasets loaded successfully!")
            return
        
        # Step 2: Load models
        preprocessor, models = load_models_and_preprocessor(EXPERIMENT_TAG)
        
        # Step 3: Test all datasets
        all_results = test_all_datasets(preprocessor, models, datasets)
        
        if not all_results:
            print("\n❌ No results generated!")
            return
        
        # Step 4: Generate summary
        summary_df = generate_summary_table(all_results)
        
        # Step 5: Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        plot_detection_heatmap(summary_df)
        plot_dataset_comparison(summary_df)
        plot_confidence_comparison(all_results)
        plot_model_consistency(summary_df)
        
        # Step 6: Overfitting analysis
        analyze_overfitting(summary_df)
        
        # Final summary
        print("\n" + "="*80)
        print("MULTI-DATASET TESTING COMPLETE!")
        print("="*80)
        print(f"\nTested on {len(datasets)} datasets:")
        for ds_name, ds_df in datasets.items():
            print(f"  - {ds_name}: {len(ds_df)} malicious queries")
        
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            print(f"  - {f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()