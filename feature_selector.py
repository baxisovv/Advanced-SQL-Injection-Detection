"""
Chi-Square Feature Selection for SQL Injection Detection

This module implements chi-square based feature selection with comprehensive
analysis and visualization capabilities.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import logger, CHI_SQUARE_K_FEATURES, CHI_SQUARE_PERCENTILE, CHI_SQUARE_P_VALUE_THRESHOLD


class ChiSquareFeatureSelector:
    """
    Chi-square based feature selection for SQL injection detection
    
    This class wraps sklearn's chi-square feature selection with additional
    analysis and visualization capabilities specific to the SQLi detection task.
    """
    
    def __init__(self, k=None, percentile=None, p_value_threshold=None):
        """
        Initialize feature selector
        
        Parameters:
        -----------
        k : int or None
            Number of top features to select (None = use percentile)
        percentile : int or None
            Percentile of features to keep (e.g., 75 = top 75%)
        p_value_threshold : float
            Maximum p-value for feature to be considered significant
        """
        self.k = k
        self.percentile = percentile
        self.p_value_threshold = p_value_threshold
        
        if k is not None:
            self.selector = SelectKBest(chi2, k=k)
        elif percentile is not None:
            self.selector = SelectPercentile(chi2, percentile=percentile)
        else:
            raise ValueError("Must specify either k or percentile")
        
        self.selected_features_ = None
        self.feature_scores_ = None
        self.feature_pvalues_ = None
        self.original_feature_names_ = None
        self.selected_feature_names_ = None
        self.n_structural_features_ = None
    
    def fit(self, X, y, feature_names, n_structural_features):
        """
        Fit the selector and compute chi-square statistics
        
        Parameters:
        -----------
        X : sparse matrix
            Feature matrix (structural + TF-IDF features)
        y : array-like
            Target labels
        feature_names : list
            Names of all features
        n_structural_features : int
            Number of structural features (non-TF-IDF)
        """
        self.original_feature_names_ = feature_names
        self.n_structural_features_ = n_structural_features
        
        # Fit selector
        self.selector.fit(X, y)
        
        # Get chi-square scores and p-values
        self.feature_scores_ = self.selector.scores_
        self.feature_pvalues_ = self.selector.pvalues_
        
        # Identify selected features
        self.selected_features_ = self.selector.get_support(indices=True)
        self.selected_feature_names_ = [
            feature_names[i] for i in self.selected_features_
        ]
        
        # Apply p-value threshold if specified
        if self.p_value_threshold is not None:
            significant_mask = self.feature_pvalues_ < self.p_value_threshold
            selected_mask = self.selector.get_support()
            
            # Combine: must be selected AND significant
            final_mask = selected_mask & significant_mask
            self.selected_features_ = np.where(final_mask)[0]
            self.selected_feature_names_ = [
                feature_names[i] for i in self.selected_features_
            ]
            
            logger.info(
                f"Chi-square selection: {len(self.selected_features_)} features "
                f"(after p-value threshold {self.p_value_threshold})"
            )
        
        logger.info(f"Selected {len(self.selected_features_)} / {len(feature_names)} features")
        
        # Analyze structural vs TF-IDF distribution
        self._analyze_feature_distribution()
        
        return self
    
    def transform(self, X):
        """Transform feature matrix to selected features only"""
        if self.selector is None:
            raise RuntimeError("Must call fit() before transform()")
        
        return self.selector.transform(X)
    
    def fit_transform(self, X, y, feature_names, n_structural_features):
        """Fit and transform in one step"""
        self.fit(X, y, feature_names, n_structural_features)
        return self.transform(X)
    
    def _analyze_feature_distribution(self):
        """Analyze distribution of selected features by type"""
        structural_selected = sum(
            1 for idx in self.selected_features_ 
            if idx < self.n_structural_features_
        )
        tfidf_selected = len(self.selected_features_) - structural_selected
        
        logger.info(f"  Structural features selected: {structural_selected} / {self.n_structural_features_}")
        logger.info(f"  TF-IDF features selected: {tfidf_selected} / "
                   f"{len(self.original_feature_names_) - self.n_structural_features_}")
        
        # Check if critical structural features are retained
        if structural_selected < self.n_structural_features_:
            dropped_structural = [
                self.original_feature_names_[i] 
                for i in range(self.n_structural_features_)
                if i not in self.selected_features_
            ]
            if dropped_structural:
                logger.warning(f"  Dropped structural features: {dropped_structural}")
    
    def get_feature_importance_df(self):
        """
        Create DataFrame with feature importance information
        
        Returns:
        --------
        DataFrame with columns: Feature, Chi2_Score, P_Value, Selected, Feature_Type
        """
        feature_types = [
            'Structural' if i < self.n_structural_features_ else 'TF-IDF'
            for i in range(len(self.original_feature_names_))
        ]
        
        df = pd.DataFrame({
            'Feature': self.original_feature_names_,
            'Chi2_Score': self.feature_scores_,
            'P_Value': self.feature_pvalues_,
            'Selected': [i in self.selected_features_ for i in range(len(self.original_feature_names_))],
            'Feature_Type': feature_types
        })
        
        return df.sort_values('Chi2_Score', ascending=False)
    
    def plot_feature_selection_analysis(self, save_dir='./figures/'):
        """Generate comprehensive visualization of feature selection"""
        os.makedirs(save_dir, exist_ok=True)
        
        importance_df = self.get_feature_importance_df()
        
        # Figure 1: Chi-square scores for all features (selected vs not selected)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top 30 features by chi-square score
        top_30 = importance_df.head(30)
        colors = ['green' if selected else 'gray' 
                 for selected in top_30['Selected']]
        axes[0, 0].barh(range(30), top_30['Chi2_Score'], color=colors)
        axes[0, 0].set_yticks(range(30))
        axes[0, 0].set_yticklabels(top_30['Feature'], fontsize=8)
        axes[0, 0].set_xlabel('Chi-Square Score')
        axes[0, 0].set_title('Top 30 Features by Chi-Square Score\n(Green = Selected, Gray = Not Selected)')
        axes[0, 0].invert_yaxis()
        
        # Plot 2: Distribution of chi-square scores
        axes[0, 1].hist(importance_df['Chi2_Score'], bins=50, edgecolor='black', alpha=0.7)
        min_selected_score = importance_df[importance_df['Selected']]['Chi2_Score'].min()
        axes[0, 1].axvline(min_selected_score,
                          color='red', linestyle='--', linewidth=2,
                          label=f'Selection Threshold')
        axes[0, 1].set_xlabel('Chi-Square Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Chi-Square Scores')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Plot 3: P-value distribution
        axes[1, 0].hist(importance_df['P_Value'], bins=50, edgecolor='black', alpha=0.7)
        if self.p_value_threshold:
            axes[1, 0].axvline(self.p_value_threshold, color='red', 
                             linestyle='--', linewidth=2,
                             label=f'P-value Threshold = {self.p_value_threshold}')
        axes[1, 0].set_xlabel('P-Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of P-Values')
        axes[1, 0].legend()
        
        # Plot 4: Feature type breakdown
        feature_type_counts = importance_df.groupby(['Feature_Type', 'Selected']).size().unstack(fill_value=0)
        feature_type_counts.plot(kind='bar', ax=axes[1, 1], color=['gray', 'green'])
        axes[1, 1].set_xlabel('Feature Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Feature Selection by Type')
        axes[1, 1].legend(['Not Selected', 'Selected'])
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}chi_square_feature_selection_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature selection analysis saved to {save_dir}")
        
        # Save detailed CSV
        importance_df.to_csv(f'{save_dir}feature_selection_details.csv', index=False)
        logger.info(f"Feature importance details saved to {save_dir}feature_selection_details.csv")


def find_optimal_k_features(X_train, y_train, X_val, y_val, 
                           feature_names, n_structural_features,
                           model_class, k_range=None):
    """
    Find optimal number of features by training models with different k values
    
    Returns:
    --------
    best_k : int
        Optimal number of features
    results_df : DataFrame
        Performance for each k value
    """
    from sklearn.metrics import f1_score
    
    if k_range is None:
        max_features = len(feature_names)
        k_range = [10, 20, 30, 50, 75, 100, 150, 200, max_features]
        k_range = [k for k in k_range if k <= max_features]
    
    results = []
    
    for k in k_range:
        logger.info(f"Testing k={k} features...")
        
        # Select features
        selector = ChiSquareFeatureSelector(k=k)
        X_train_selected = selector.fit_transform(
            X_train, y_train, feature_names, n_structural_features
        )
        X_val_selected = selector.transform(X_val)
        
        # Train model
        model = model_class()
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_val_pred = model.predict(X_val_selected)
        val_f1 = f1_score(y_val, y_val_pred)
        
        results.append({
            'k': k,
            'val_f1': val_f1,
            'n_structural': sum(1 for idx in selector.selected_features_ 
                              if idx < n_structural_features),
            'n_tfidf': k - sum(1 for idx in selector.selected_features_ 
                              if idx < n_structural_features)
        })
        
        logger.info(f"  k={k}: Val F1={val_f1:.4f}")
    
    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df['val_f1'].idxmax(), 'k']
    
    logger.info(f"\nOptimal k={best_k} with Val F1={results_df['val_f1'].max():.4f}")
    
    return int(best_k), results_df
