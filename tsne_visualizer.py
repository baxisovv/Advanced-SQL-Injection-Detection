"""
t-SNE Visualization for SQL Injection Detection

This module provides comprehensive t-SNE visualizations including:
- Binary classification (benign vs malicious)
- Multi-class attack type distribution  
- Runtime detection tracking (which specific attacks were caught)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.sparse import issparse
import os

from config import logger


class TSNEVisualizer:
    """
    t-SNE visualization for SQL injection detection
    
    Generates 2D embeddings of feature space with color-coding for:
    - Binary classification (benign vs malicious)
    - Multi-class attack types
    - Runtime detection results (for model evaluation)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tsne_model = None
        self.embeddings_ = None
        self.subsample_indices_ = None
    
    def fit_transform(self, X, perplexity=30, n_iter=1000, learning_rate=200):
        """
        Compute t-SNE embeddings
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Feature matrix to embed
        perplexity : float
            t-SNE perplexity parameter (typical range: 5-50)
        n_iter : int
            Number of optimization iterations
        learning_rate : float
            Learning rate for optimization
        
        Returns:
        --------
        embeddings : ndarray of shape (n_samples, 2)
            2D coordinates for each sample
        """
        logger.info(f"Computing t-SNE embeddings (perplexity={perplexity}, "
                   f"n_iter={n_iter})...")
        
        # Convert sparse to dense if needed (t-SNE requires dense)
        if issparse(X):
            logger.info("Converting sparse matrix to dense for t-SNE...")
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Subsample if dataset is very large (t-SNE is O(n²))
        if X_dense.shape[0] > 10000:
            logger.warning(f"Dataset has {X_dense.shape[0]} samples. "
                          f"Subsampling to 10000 for t-SNE performance.")
            indices = np.random.choice(X_dense.shape[0], 10000, replace=False)
            X_dense = X_dense[indices]
            self.subsample_indices_ = indices
        else:
            self.subsample_indices_ = None
        
        # Fit t-SNE
        self.tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=self.random_state,
            verbose=1
        )
        
        self.embeddings_ = self.tsne_model.fit_transform(X_dense)
        
        logger.info(f"t-SNE embeddings computed: {self.embeddings_.shape}")
        return self.embeddings_
    
    def plot_binary_classification(self, y_true, y_pred=None, 
                                   save_path='./figures/tsne_binary.png'):
        """
        Visualize t-SNE embeddings colored by true labels and predictions
        
        Parameters:
        -----------
        y_true : array-like
            True labels (0=benign, 1=malicious)
        y_pred : array-like, optional
            Predicted labels (if provided, will show misclassifications)
        save_path : str
            Where to save the plot
        """
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
            # Identify correct and incorrect predictions
            correct = (y_true == y_pred)
            incorrect = ~correct
            
            # Plot correct predictions
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
            
            # Highlight misclassifications
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
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Binary classification t-SNE plot saved to {save_path}")
    
    def plot_attack_types(self, injection_types, y_true=None,
                         save_path='./figures/tsne_attack_types.png'):
        """
        Visualize t-SNE embeddings colored by SQL injection attack types
        
        This visualization shows which attack types are detected during runtime
        and how they cluster in the feature space.
        
        Parameters:
        -----------
        injection_types : array-like
            Attack type labels for each sample
        y_true : array-like, optional
            True binary labels (0=benign, 1=malicious) to overlay
        save_path : str
            Where to save the plot
        """
        if self.embeddings_ is None:
            raise RuntimeError("Must call fit_transform() first")
        
        # Handle subsampling
        if self.subsample_indices_ is not None:
            injection_types = np.array(injection_types)[self.subsample_indices_]
            if y_true is not None:
                y_true = np.array(y_true)[self.subsample_indices_]
        
        # Get unique attack types
        unique_types = pd.Series(injection_types).unique()
        n_types = len(unique_types)
        
        # Create color palette
        if n_types <= 10:
            palette = sns.color_palette("tab10", n_types)
        else:
            palette = sns.color_palette("husl", n_types)
        
        type_to_color = {attack_type: palette[i] 
                        for i, attack_type in enumerate(unique_types)}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each attack type
        for attack_type in unique_types:
            mask = (injection_types == attack_type)
            
            # Different marker for benign vs malicious if y_true provided
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
        
        # Place legend outside plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attack types t-SNE plot saved to {save_path}")
        logger.info(f"  Visualized {n_types} distinct attack types")
    
    def plot_runtime_detections(self, injection_types, y_true, y_pred, 
                               model_name='Model',
                               save_path='./figures/tsne_runtime_detections.png'):
        """
        CRITICAL: Visualize which attack types were caught during runtime
        
        This plot shows:
        1. Which specific attack types the model successfully detected
        2. Which attack types were missed (false negatives)
        3. Spatial distribution of each attack type in feature space
        
        Parameters:
        -----------
        injection_types : array-like
            Attack type labels for each sample
        y_true : array-like
            True binary labels (0=benign, 1=malicious)
        y_pred : array-like
            Predicted binary labels
        model_name : str
            Name of the model being evaluated
        save_path : str
            Where to save the plot
        """
        if self.embeddings_ is None:
            raise RuntimeError("Must call fit_transform() first")
        
        # Handle subsampling
        if self.subsample_indices_ is not None:
            injection_types = np.array(injection_types)[self.subsample_indices_]
            y_true = np.array(y_true)[self.subsample_indices_]
            y_pred = np.array(y_pred)[self.subsample_indices_]
        
        # Categorize predictions for malicious samples only
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
        
        # Sort by detection rate for legend
        sorted_types = sorted(detection_rates.items(), 
                            key=lambda x: x[1]['rate'], 
                            reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Color palette
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
            
            # Plot detected attacks (filled circles)
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
            
            # Plot missed attacks (X markers)
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
        
        # Legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, framealpha=0.9, 
                 title='Attack Types (Detection Rate)')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Runtime detection t-SNE plot saved to {save_path}")
        logger.info("="*60)
        logger.info(f"ATTACK TYPE DETECTION SUMMARY ({model_name}):")
        logger.info("="*60)
        for attack_type, stats in sorted_types:
            logger.info(f"  {attack_type:30s}: {stats['rate']:>6.1%} "
                       f"({stats['detected']:>4}/{stats['total']:<4} detected)")
        logger.info("="*60)
        
        return detection_rates
