"""
Enhanced Visualization Module for SQL Injection Detection

This module provides comprehensive visualizations including:
- Standard evaluation plots (PR curves, confusion matrices)
- Comparison plots (before/after tuning, feature selection, etc.)
- Training time and efficiency comparisons
- Attack type analysis

FIXED: Uses 'Agg' backend to avoid tkinter threading issues
"""

import os

# FIX: Set matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # This fixes the tkinter threading errors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

OUTPUT_DIR = './figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


def plot_pr_curve(y_true, model_probs_dict: dict, tag: str = '') -> None:
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 7))
    for name, probs in model_probs_dict.items():
        if probs is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{name} (AUC={pr_auc:.3f})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (Test Set)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'pr_curve_{tag}.png' if tag else 'pr_curve.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_granular_recall_heatmap(df_test_eval: pd.DataFrame, tag: str = '') -> None:
    """Plot heatmap showing recall by model and attack type"""
    malicious_only = df_test_eval[df_test_eval['true_label'] == 1]
    pivot_data = (
        malicious_only
        .groupby(['model_name', 'injection_type'])
        .apply(lambda x: (x['pred_label'] == 1).sum() / len(x) if len(x) > 0 else 0)
        .reset_index(name='recall')
    )
    heatmap_data = pivot_data.pivot(index='model_name', columns='injection_type', values='recall')

    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', 
                cbar_kws={'label': 'Recall Rate'}, linewidths=0.5)
    plt.title('Model Recall by SQL Injection Type', fontsize=16, fontweight='bold')
    plt.ylabel('Model', fontsize=14)
    plt.xlabel('Injection Classification', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f'recall_heatmap_{tag}.png' if tag else 'recall_heatmap.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_feature_importance(importance_array, feature_names: list[str], tag: str = '') -> None:
    """Plot feature importance for tree-based models"""
    sorted_idx        = np.argsort(importance_array)[::-1]
    sorted_names      = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importance_array[i] for i in sorted_idx]

    max_display = min(25, len(sorted_names))
    sorted_names       = sorted_names[:max_display]
    sorted_importances = sorted_importances[:max_display]

    plt.figure(figsize=(10, max(6, max_display * 0.35)))
    colors = ['green' if 'tfidf' not in name.lower() else 'blue' for name in sorted_names]
    ax = sns.barplot(x=sorted_importances, y=sorted_names, hue=sorted_names, 
                     palette=colors, legend=False)
    ax.set_xlabel('Relative Importance', fontsize=14)
    ax.set_title('Top Features by Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = f'feature_importance_{tag}.png' if tag else 'feature_importance.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_ablation_comparison(full_results: pd.DataFrame, ablation_results: pd.DataFrame) -> None:
    """Compare full features vs structural-only features"""
    merged = (
        full_results[['Model', 'Test_F1']].rename(columns={'Test_F1': 'Full Features'})
        .merge(ablation_results[['Model', 'Test_F1']].rename(columns={'Test_F1': 'Structural Only'}), on='Model')
    )
    melted = merged.melt(id_vars='Model', var_name='Feature Set', value_name='Test F1')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Model', y='Test F1', hue='Feature Set', palette='Set2')
    plt.title('Feature Ablation: Full vs Structural-Only', fontsize=16, fontweight='bold')
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Feature Set', fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_comparison.png'), dpi=300)
    plt.close()


def plot_confusion_matrices(y_true, test_preds_dict: dict, tag: str = '') -> None:
    """Generate confusion matrix for each model"""
    for name, preds in test_preds_dict.items():
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
        
        plt.figure(figsize=(7, 6))
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold')
        plt.grid(False) 
        plt.tight_layout()
        safe_name = name.replace(" ", "_")
        filename = f'conf_matrix_{safe_name}_{tag}.png' if tag else f'conf_matrix_{safe_name}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()


def plot_detailed_metrics(df_results: pd.DataFrame, tag: str) -> None:
    """Plot detailed diagnostic metrics for all models"""
    plot_df = df_results[['Model', 'Recall', 'FPR', 'TNR', 'FNR']].copy()
    plot_df = plot_df.rename(columns={'Recall': 'TPR (Recall)'})
    melted = plot_df.melt(id_vars='Model', var_name='Metric', value_name='Rate')
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=melted, x='Model', y='Rate', hue='Metric', palette='muted')
    plt.title(f'Detailed Diagnostic Metrics by Model [{tag}]', fontsize=16, fontweight='bold')
    plt.ylabel('Rate (0.0 to 1.0)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Metric', fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'detailed_metrics_{tag}.png'), dpi=300)
    plt.close()


def plot_chi_square_analysis(results: pd.DataFrame) -> None:
    """Visualize chi-square feature analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 20 overall
    top_20 = results.head(20)
    colors = ['green' if ft == 'Structural' else 'blue' for ft in top_20['Feature_Type']]
    axes[0].barh(range(20), top_20['Chi2_Score'], color=colors)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(top_20['Feature'], fontsize=10)
    axes[0].set_xlabel('Chi-Square Score', fontsize=12)
    axes[0].set_title('Top 20 Most Discriminative Features', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()

    # Top 20 TF-IDF only
    tfidf_only = results[results['Feature_Type'] == 'TF-IDF'].head(20)
    if len(tfidf_only) > 0:
        axes[1].barh(range(len(tfidf_only)), tfidf_only['Chi2_Score'], color='blue')
        axes[1].set_yticks(range(len(tfidf_only)))
        axes[1].set_yticklabels(tfidf_only['Feature'], fontsize=10)
        axes[1].set_xlabel('Chi-Square Score', fontsize=12)
        axes[1].set_title('Top 20 TF-IDF N-grams', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chi_square_analysis.png'), dpi=300)
    plt.close()


# NEW COMPARISON VISUALIZATIONS

def plot_experiment_comparison(experiment_results: dict, metric='Test_F1') -> None:
    """
    Compare multiple experiments (e.g., different feature selection strategies)
    
    Parameters:
    -----------
    experiment_results : dict
        Dictionary mapping experiment names to results DataFrames
    metric : str
        Metric to compare (default: Test_F1)
    """
    comparison_data = []
    for exp_name, results_df in experiment_results.items():
        for _, row in results_df.iterrows():
            comparison_data.append({
                'Experiment': exp_name,
                'Model': row['Model'],
                metric: row[metric]
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar plot
    plt.figure(figsize=(14, 7))
    
    # Get unique models
    models = comp_df['Model'].unique()
    experiments = comp_df['Experiment'].unique()
    
    x = np.arange(len(models))
    width = 0.8 / len(experiments)
    
    for i, exp in enumerate(experiments):
        exp_data = comp_df[comp_df['Experiment'] == exp]
        values = [exp_data[exp_data['Model'] == m][metric].values[0] if m in exp_data['Model'].values else 0 
                 for m in models]
        plt.bar(x + i * width, values, width, label=exp, alpha=0.8)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f'Experiment Comparison: {metric}', fontsize=16, fontweight='bold')
    plt.xticks(x + width * (len(experiments) - 1) / 2, models, rotation=30, ha='right')
    plt.legend(title='Experiment', fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(comp_df[metric].max() * 1.1, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'experiment_comparison_{metric}.png'), dpi=300)
    plt.close()


def plot_tuning_comparison(before_tuning: pd.DataFrame, after_tuning: pd.DataFrame) -> None:
    """Compare model performance before and after hyperparameter tuning"""
    merged = before_tuning[['Model', 'Test_F1']].rename(columns={'Test_F1': 'Before Tuning'})
    merged = merged.merge(
        after_tuning[['Model', 'Test_F1']].rename(columns={'Test_F1': 'After Tuning'}),
        on='Model'
    )
    
    melted = merged.melt(id_vars='Model', var_name='Stage', value_name='Test F1')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Model', y='Test F1', hue='Stage', palette='viridis')
    plt.title('Impact of Hyperparameter Tuning', fontsize=16, fontweight='bold')
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Tuning Stage', fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add improvement annotations
    for i, row in merged.iterrows():
        improvement = row['After Tuning'] - row['Before Tuning']
        if abs(improvement) > 0.001:
            plt.text(i, max(row['After Tuning'], row['Before Tuning']) + 0.02,
                    f'+{improvement:.3f}' if improvement > 0 else f'{improvement:.3f}',
                    ha='center', fontsize=9, color='green' if improvement > 0 else 'red',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tuning_comparison.png'), dpi=300)
    plt.close()


def plot_feature_selection_impact(full_features: pd.DataFrame, selected_features: pd.DataFrame) -> None:
    """Compare performance with full vs selected features"""
    merged = full_features[['Model', 'Test_F1']].rename(columns={'Test_F1': 'All Features'})
    merged = merged.merge(
        selected_features[['Model', 'Test_F1']].rename(columns={'Test_F1': 'Selected Features'}),
        on='Model'
    )
    
    melted = merged.melt(id_vars='Model', var_name='Feature Set', value_name='Test F1')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Model', y='Test F1', hue='Feature Set', palette='Set1')
    plt.title('Impact of Chi-Square Feature Selection', fontsize=16, fontweight='bold')
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Feature Set', fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_selection_impact.png'), dpi=300)
    plt.close()


def plot_training_time_comparison(results_dict: dict) -> None:
    """
    Compare training times across different experiments
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping experiment names to results DataFrames
    """
    plt.figure(figsize=(12, 6))
    
    for exp_name, results_df in results_dict.items():
        if 'Train_Time_s' in results_df.columns:
            plt.barh(results_df['Model'], results_df['Train_Time_s'], alpha=0.7, label=exp_name)
    
    plt.xlabel('Training Time (seconds)', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.title('Training Time Comparison Across Experiments', fontsize=16, fontweight='bold')
    plt.legend(title='Experiment', fontsize=10)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_time_comparison.png'), dpi=300)
    plt.close()


def plot_undersampling_impact(results_dict: dict) -> None:
    """
    Visualize impact of different undersampling strategies
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping undersampling strategy names to results DataFrames
    """
    # Prepare data
    comparison_data = []
    for strategy, results_df in results_dict.items():
        for _, row in results_df.iterrows():
            comparison_data.append({
                'Strategy': strategy,
                'Model': row['Model'],
                'Test_F1': row['Test_F1'],
                'Recall': row['Recall'],
                'Precision': row['Precision']
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['Test_F1', 'Recall', 'Precision']
    titles = ['F1 Score', 'Recall', 'Precision']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        pivot = comp_df.pivot(index='Model', columns='Strategy', values=metric)
        pivot.plot(kind='bar', ax=axes[idx], alpha=0.8)
        axes[idx].set_title(f'{title} by Undersampling Strategy', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(title, fontsize=11)
        axes[idx].set_xlabel('Model', fontsize=11)
        axes[idx].legend(title='Strategy', fontsize=9)
        axes[idx].grid(True, axis='y', alpha=0.3)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'undersampling_impact.png'), dpi=300)
    plt.close()


def plot_comprehensive_summary(all_results: dict) -> None:
    """
    Create a comprehensive summary visualization showing all experiments
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with experiment names as keys and results DataFrames as values
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: F1 Score Comparison
    f1_data = []
    for exp_name, results_df in all_results.items():
        for _, row in results_df.iterrows():
            f1_data.append({
                'Experiment': exp_name,
                'Model': row['Model'],
                'F1': row['Test_F1']
            })
    
    f1_df = pd.DataFrame(f1_data)
    pivot_f1 = f1_df.pivot(index='Model', columns='Experiment', values='F1')
    pivot_f1.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
    axes[0, 0].set_title('F1 Score Across All Experiments', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Test F1', fontsize=11)
    axes[0, 0].legend(title='Experiment', fontsize=8, loc='lower right')
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Precision vs Recall tradeoff
    for exp_name, results_df in all_results.items():
        if 'Ensemble' not in exp_name:  # Skip to reduce clutter
            axes[0, 1].scatter(results_df['Recall'], results_df['Precision'], 
                             label=exp_name, alpha=0.7, s=100)
    
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1.05])
    axes[0, 1].set_ylim([0, 1.05])
    
    # Plot 3: FPR Comparison
    fpr_data = []
    for exp_name, results_df in all_results.items():
        for _, row in results_df.iterrows():
            fpr_data.append({
                'Experiment': exp_name,
                'Model': row['Model'],
                'FPR': row['FPR']
            })
    
    fpr_df = pd.DataFrame(fpr_data)
    pivot_fpr = fpr_df.pivot(index='Model', columns='Experiment', values='FPR')
    pivot_fpr.plot(kind='bar', ax=axes[1, 0], alpha=0.8, color=['red', 'orange', 'yellow'])
    axes[1, 0].set_title('False Positive Rate (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('FPR', fontsize=11)
    axes[1, 0].legend(title='Experiment', fontsize=8)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Best Model Identification
    best_models = []
    for exp_name, results_df in all_results.items():
        best_idx = results_df['Test_F1'].idxmax()
        best_models.append({
            'Experiment': exp_name,
            'Best_Model': results_df.loc[best_idx, 'Model'],
            'Best_F1': results_df.loc[best_idx, 'Test_F1']
        })
    
    best_df = pd.DataFrame(best_models)
    axes[1, 1].barh(best_df['Experiment'], best_df['Best_F1'], alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Best F1 Score', fontsize=11)
    axes[1, 1].set_title('Best Performing Model per Experiment', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, axis='x', alpha=0.3)
    
    # Add model names as annotations
    for i, row in best_df.iterrows():
        axes[1, 1].text(row['Best_F1'] + 0.01, i, row['Best_Model'], 
                       va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_summary.png'), dpi=300)
    plt.close()


def save_experiment_report(all_results: dict, output_file: str = 'experiment_report.csv') -> None:
    """
    Save a comprehensive CSV report of all experiments
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with experiment names as keys and results DataFrames as values
    output_file : str
        Output CSV filename
    """
    report_data = []
    for exp_name, results_df in all_results.items():
        for _, row in results_df.iterrows():
            report_row = {'Experiment': exp_name}
            report_row.update(row.to_dict())
            report_data.append(report_row)
    
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(OUTPUT_DIR, output_file)
    report_df.to_csv(report_path, index=False)
    print(f"Comprehensive experiment report saved to {report_path}")