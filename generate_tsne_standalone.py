"""
Standalone t-SNE Visualization Generator

This script loads already-trained models and generates t-SNE visualizations
without retraining. Use this when t-SNE failed during main pipeline.

Usage:
    python generate_tsne_standalone.py

Requirements:
    - Trained models in ./models_out/
    - Test data (will reload from original dataset)
    - Saved preprocessor
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from scipy.sparse import issparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
EXPERIMENT_TAG = 'best_with_ensembles'  # Which experiment to visualize
DATA_PATH = 'rbsqli_dataset.csv'  # Your original dataset
MODEL_DIR = './models_out/'
OUTPUT_DIR = './figures/tsne_standalone/'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STANDALONE t-SNE VISUALIZATION GENERATOR")
print("="*80)
print(f"\nExperiment: {EXPERIMENT_TAG}")
print(f"Models directory: {MODEL_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


class TSNEVisualizer:
    """t-SNE visualization with attack type tracking"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.embeddings_ = None
        self.subsample_indices_ = None
    
    def fit_transform(self, X, perplexity=30, max_iter=1000, learning_rate=200):
        """
        Compute t-SNE embeddings (FIXED: uses max_iter instead of n_iter)
        """
        print(f"\nComputing t-SNE embeddings (perplexity={perplexity}, max_iter={max_iter})...")
        
        # Convert sparse to dense
        if issparse(X):
            print("Converting sparse matrix to dense for t-SNE...")
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Subsample if too large
        if X_dense.shape[0] > 10000:
            print(f"Dataset has {X_dense.shape[0]} samples. Subsampling to 10000 for t-SNE performance.")
            indices = np.random.RandomState(self.random_state).choice(
                X_dense.shape[0], 10000, replace=False
            )
            X_dense = X_dense[indices]
            self.subsample_indices_ = indices
        else:
            self.subsample_indices_ = None
        
        # Fit t-SNE (FIXED: max_iter instead of n_iter)
        try:
            self.tsne_model = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,  # FIXED: was n_iter
                learning_rate=learning_rate,
                random_state=self.random_state,
                verbose=1
            )
            
            self.embeddings_ = self.tsne_model.fit_transform(X_dense)
            print(f"✓ t-SNE embeddings computed: {self.embeddings_.shape}")
            return self.embeddings_
            
        except Exception as e:
            print(f"✗ Error computing t-SNE: {e}")
            # Try with older parameter name
            print("Attempting with n_iter parameter...")
            self.tsne_model = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=max_iter,
                learning_rate=learning_rate,
                random_state=self.random_state,
                verbose=1
            )
            self.embeddings_ = self.tsne_model.fit_transform(X_dense)
            print(f"✓ t-SNE embeddings computed: {self.embeddings_.shape}")
            return self.embeddings_
    
    def plot_binary_classification(self, y_true, y_pred=None, 
                                   save_path='tsne_binary.png'):
        """Visualize binary classification"""
        if self.embeddings_ is None:
            raise RuntimeError("Must call fit_transform() first")
        
        # Handle subsampling
        if self.subsample_indices_ is not None:
            y_true = np.array(y_true)[self.subsample_indices_]
            if y_pred is not None:
                y_pred = np.array(y_pred)[self.subsample_indices_]
        
        fig, axes = plt.subplots(1, 2 if y_pred is not None else 1, 
                                figsize=(16 if y_pred is not None else 10, 7))
        
        if y_pred is None:
            axes = [axes]
        
        # Plot 1: True labels
        scatter = axes[0].scatter(
            self.embeddings_[:, 0],
            self.embeddings_[:, 1],
            c=y_true,
            cmap='RdYlGn_r',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        axes[0].set_title('t-SNE Visualization: True Labels', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Class (0=Benign, 1=Malicious)')
        
        # Plot 2: Predictions (if provided)
        if y_pred is not None:
            correct = (y_true == y_pred)
            incorrect = ~correct
            
            axes[1].scatter(
                self.embeddings_[correct, 0],
                self.embeddings_[correct, 1],
                c=y_true[correct],
                cmap='RdYlGn_r',
                alpha=0.6,
                s=20,
                label='Correct',
                edgecolors='none'
            )
            
            axes[1].scatter(
                self.embeddings_[incorrect, 0],
                self.embeddings_[incorrect, 1],
                c='black',
                marker='x',
                s=50,
                label=f'Misclassified ({incorrect.sum()})',
                alpha=0.8
            )
            
            axes[1].set_title('t-SNE Visualization: Predictions vs Ground Truth', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel('t-SNE Dimension 1')
            axes[1].set_ylabel('t-SNE Dimension 2')
            axes[1].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
    
    def plot_attack_types(self, injection_types, y_true=None,
                         save_path='tsne_attack_types.png'):
        """Visualize attack type clustering"""
        if self.embeddings_ is None:
            raise RuntimeError("Must call fit_transform() first")
        
        if self.subsample_indices_ is not None:
            injection_types = np.array(injection_types)[self.subsample_indices_]
            if y_true is not None:
                y_true = np.array(y_true)[self.subsample_indices_]
        
        unique_types = pd.Series(injection_types).unique()
        n_types = len(unique_types)
        
        if n_types <= 10:
            palette = sns.color_palette("tab10", n_types)
        else:
            palette = sns.color_palette("husl", n_types)
        
        type_to_color = {attack_type: palette[i] 
                        for i, attack_type in enumerate(unique_types)}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for attack_type in unique_types:
            mask = (injection_types == attack_type)
            
            if y_true is not None:
                malicious_mask = mask & (y_true == 1)
                benign_mask = mask & (y_true == 0)
                
                if malicious_mask.sum() > 0:
                    ax.scatter(
                        self.embeddings_[malicious_mask, 0],
                        self.embeddings_[malicious_mask, 1],
                        c=[type_to_color[attack_type]],
                        label=f'{attack_type} (Malicious)',
                        alpha=0.7,
                        s=30,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.5
                    )
                
                if benign_mask.sum() > 0:
                    ax.scatter(
                        self.embeddings_[benign_mask, 0],
                        self.embeddings_[benign_mask, 1],
                        c=[type_to_color[attack_type]],
                        label=f'{attack_type} (Benign)',
                        alpha=0.4,
                        s=20,
                        marker='^',
                        edgecolors='gray',
                        linewidths=0.5
                    )
            else:
                ax.scatter(
                    self.embeddings_[mask, 0],
                    self.embeddings_[mask, 1],
                    c=[type_to_color[attack_type]],
                    label=attack_type,
                    alpha=0.7,
                    s=30,
                    edgecolors='black',
                    linewidths=0.5
                )
        
        ax.set_title('t-SNE Visualization: SQL Injection Attack Types\n'
                    '(Circle=Malicious, Triangle=Benign)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
    
    def plot_runtime_detections(self, injection_types, y_true, y_pred, 
                               model_name='Model',
                               save_path='tsne_runtime_detections.png'):
        """Visualize which attack types were caught during runtime"""
        if self.embeddings_ is None:
            raise RuntimeError("Must call fit_transform() first")
        
        if self.subsample_indices_ is not None:
            injection_types = np.array(injection_types)[self.subsample_indices_]
            y_true = np.array(y_true)[self.subsample_indices_]
            y_pred = np.array(y_pred)[self.subsample_indices_]
        
        # Focus on malicious samples
        malicious_mask = (y_true == 1)
        injection_types_mal = injection_types[malicious_mask]
        y_pred_mal = y_pred[malicious_mask]
        embeddings_mal = self.embeddings_[malicious_mask]
        
        # Calculate detection rate per attack type
        unique_types = pd.Series(injection_types_mal).unique()
        detection_rates = {}
        
        for attack_type in unique_types:
            type_mask = (injection_types_mal == attack_type)
            detected = y_pred_mal[type_mask] == 1
            detection_rates[attack_type] = {
                'rate': detected.mean(),
                'detected': detected.sum(),
                'total': type_mask.sum()
            }
        
        # Sort by detection rate
        sorted_types = sorted(detection_rates.items(), 
                            key=lambda x: x[1]['rate'], 
                            reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        n_types = len(unique_types)
        if n_types <= 10:
            palette = sns.color_palette("tab10", n_types)
        else:
            palette = sns.color_palette("husl", n_types)
        
        type_to_color = {attack_type: palette[i] 
                        for i, attack_type in enumerate(unique_types)}
        
        # Plot each attack type
        for attack_type, stats in sorted_types:
            type_mask = (injection_types_mal == attack_type)
            detected_mask = type_mask & (y_pred_mal == 1)
            missed_mask = type_mask & (y_pred_mal == 0)
            
            # Detected attacks (circles)
            if detected_mask.sum() > 0:
                ax.scatter(
                    embeddings_mal[detected_mask, 0],
                    embeddings_mal[detected_mask, 1],
                    c=[type_to_color[attack_type]],
                    label=f'{attack_type} ✓ ({stats["rate"]:.1%} detected: '
                         f'{stats["detected"]}/{stats["total"]})',
                    alpha=0.8,
                    s=50,
                    marker='o',
                    edgecolors='black',
                    linewidths=1
                )
            
            # Missed attacks (X marks)
            if missed_mask.sum() > 0:
                ax.scatter(
                    embeddings_mal[missed_mask, 0],
                    embeddings_mal[missed_mask, 1],
                    c=[type_to_color[attack_type]],
                    alpha=0.5,
                    s=80,
                    marker='x',
                    linewidths=2
                )
        
        ax.set_title(f't-SNE: Runtime Attack Detection by Type - {model_name}\n'
                    f'(Circle=Detected ✓, X=Missed ✗)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, framealpha=0.9, 
                 title='Attack Types (Detection Rate)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        print("="*60)
        print(f"ATTACK TYPE DETECTION SUMMARY ({model_name}):")
        print("="*60)
        for attack_type, stats in sorted_types:
            print(f"  {attack_type:30s}: {stats['rate']:>6.1%} "
                 f"({stats['detected']:>4}/{stats['total']:<4} detected)")
        print("="*60)
        
        return detection_rates


def load_preprocessor(experiment_tag):
    """Load saved preprocessor"""
    preprocessor_path = os.path.join(MODEL_DIR, f'preprocessor_{experiment_tag}.pkl')
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    
    print(f"\nLoading preprocessor: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    print("✓ Preprocessor loaded")
    return preprocessor


def load_models(experiment_tag):
    """Load all saved models"""
    models = {}
    model_files = [f for f in os.listdir(MODEL_DIR) 
                   if f.endswith(f'_{experiment_tag}_model.pkl')]
    
    print(f"\nLoading models from {MODEL_DIR}:")
    for model_file in model_files:
        model_name = model_file.replace(f'_{experiment_tag}_model.pkl', '')
        model_path = os.path.join(MODEL_DIR, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"  ✓ {model_name}")
    
    print(f"\nTotal models loaded: {len(models)}")
    return models


def load_test_data(data_path):
    """
    Reload test data from original dataset
    Uses same split logic as original pipeline
    """
    print(f"\nLoading dataset: {data_path}")
    
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    df = pd.read_csv(data_path)
    df['target'] = df['vulnerability_status'].map({'Yes': 1, 'No': 0})
    
    print(f"✓ Loaded {len(df)} total samples")
    
    # Sample (same as original pipeline)
    MAX_BENIGN = 500000
    MAX_MALICIOUS = 100000
    TEST_SIZE_RATIO = 0.20
    VAL_SIZE_RATIO = 0.20
    TEST_PRIOR_BENIGN = 0.95
    TEST_PRIOR_MALICIOUS = 0.05
    
    benign_df = df[df['target'] == 0]
    malicious_df = df[df['target'] == 1]
    
    if len(benign_df) > MAX_BENIGN:
        benign_df = benign_df.sample(MAX_BENIGN, random_state=42)
    if len(malicious_df) > MAX_MALICIOUS:
        malicious_df = malicious_df.sample(MAX_MALICIOUS, random_state=42)
    
    sampled_df = (pd.concat([benign_df, malicious_df])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True))
    
    X = sampled_df[['sql_query', 'injection_type']]
    y = sampled_df['target']
    
    # Split (same as original)
    X_trainval, X_test_pool, y_trainval, y_test_pool = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, stratify=y, random_state=42
    )
    
    # Adjust test set ratio
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
    X_test = final_test_df[['sql_query', 'injection_type']]
    y_test = final_test_df['target']
    
    print(f"✓ Test set: {len(y_test)} samples (malicious={y_test.sum()}, benign={len(y_test)-y_test.sum()})")
    
    return X_test, y_test


def main():
    """Main execution"""
    try:
        # Step 1: Load preprocessor
        preprocessor = load_preprocessor(EXPERIMENT_TAG)
        
        # Step 2: Load models
        models = load_models(EXPERIMENT_TAG)
        
        if not models:
            print("\n❌ No models found!")
            return
        
        # Step 3: Load test data
        X_test, y_test = load_test_data(DATA_PATH)
        
        # Step 4: Preprocess test data
        print("\nPreprocessing test data...")
        X_test_processed = preprocessor.transform(X_test)
        print(f"✓ Processed features shape: {X_test_processed.shape}")
        
        # Step 5: Initialize t-SNE visualizer
        print("\n" + "="*80)
        print("GENERATING t-SNE VISUALIZATIONS")
        print("="*80)
        
        visualizer = TSNEVisualizer(random_state=RANDOM_STATE)
        
        # Step 6: Compute t-SNE embeddings
        embeddings = visualizer.fit_transform(X_test_processed, perplexity=30, max_iter=1000)
        
        # Step 7: Generate visualizations
        print("\nGenerating plots...")
        
        # Plot 1: Binary classification
        visualizer.plot_binary_classification(
            y_test.values,
            save_path=os.path.join(OUTPUT_DIR, f'tsne_binary_{EXPERIMENT_TAG}.png')
        )
        
        # Plot 2: Attack types
        visualizer.plot_attack_types(
            X_test['injection_type'].values,
            y_true=y_test.values,
            save_path=os.path.join(OUTPUT_DIR, f'tsne_attack_types_{EXPERIMENT_TAG}.png')
        )
        
        # Plot 3-N: Runtime detections for each model
        detection_summaries = {}
        
        for model_name, model in models.items():
            if model_name == 'IsolationForest':
                continue  # Skip for clarity
            
            print(f"\nGenerating t-SNE for {model_name}...")
            
            # Get predictions
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test_processed)[:, 1]
                    y_pred = (probs >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test_processed)
                
                # Generate runtime detection plot
                detection_rates = visualizer.plot_runtime_detections(
                    X_test['injection_type'].values,
                    y_test.values,
                    y_pred,
                    model_name=model_name,
                    save_path=os.path.join(OUTPUT_DIR, f'tsne_runtime_{model_name}_{EXPERIMENT_TAG}.png')
                )
                
                detection_summaries[model_name] = detection_rates
                
            except Exception as e:
                print(f"  ✗ Failed to generate plot for {model_name}: {e}")
        
        # Step 8: Save detection summary CSV
        if detection_summaries:
            summary_rows = []
            for model_name, attack_stats in detection_summaries.items():
                for attack_type, stats in attack_stats.items():
                    summary_rows.append({
                        'Model': model_name,
                        'Attack_Type': attack_type,
                        'Detection_Rate': stats['rate'],
                        'Detected': stats['detected'],
                        'Total': stats['total'],
                        'Missed': stats['total'] - stats['detected']
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(OUTPUT_DIR, f'attack_type_detection_summary_{EXPERIMENT_TAG}.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Detection summary saved: {summary_path}")
        
        # Final summary
        print("\n" + "="*80)
        print("t-SNE GENERATION COMPLETE!")
        print("="*80)
        print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"  - {f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Models exist in: {MODEL_DIR}")
        print(f"  2. Dataset exists at: {DATA_PATH}")
        print(f"  3. Experiment tag '{EXPERIMENT_TAG}' is correct")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
