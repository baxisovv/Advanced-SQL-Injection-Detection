"""
Enhanced SQL Injection Detection Pipeline

This enhanced pipeline includes:
1. Undersampling strategies for class imbalance
2. Chi-square feature selection
3. Hyperparameter tuning comparisons
4. t-SNE visualizations with attack type tracking
5. Ensemble methods
6. Comprehensive experiment comparisons
"""

import os

import joblib
import pandas as pd
from sklearn.feature_selection import chi2

from config import (MODEL_SAVE_DIR, logger, UNDERSAMPLING_STRATEGY, 
                    UNDERSAMPLING_RATIO, ENABLE_CHI_SQUARE_SELECTION)
from data_loader import load_and_split_data
from evaluate import cross_validate_models, evaluate_models
from models import ModelHandler
from preprocessor import SQLPreprocessor
from visualize import (plot_ablation_comparison, plot_chi_square_analysis,
                      plot_experiment_comparison, plot_tuning_comparison,
                      plot_feature_selection_impact, plot_undersampling_impact,
                      plot_comprehensive_summary, save_experiment_report)


def run_pipeline(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 use_tfidf: bool,
                 use_chi_square: bool = False,
                 tune: bool = True,
                 use_ensembles: bool = False,
                 tag: str = '',
                 generate_tsne: bool = True) -> pd.DataFrame:
    """
    Run the complete pipeline with configurable options
    
    Parameters:
    -----------
    X_train, X_val, X_test : DataFrames
        Split datasets
    y_train, y_val, y_test : Series
        Split labels
    use_tfidf : bool
        Whether to use TF-IDF features
    use_chi_square : bool
        Whether to apply chi-square feature selection
    tune : bool
        Whether to tune hyperparameters
    use_ensembles : bool
        Whether to train ensemble models
    tag : str
        Tag for output files
    generate_tsne : bool
        Whether to generate t-SNE visualizations
    """
    label = f"[{tag}]" if tag else ""

    logger.info(f"{label} Fitting preprocessor (use_tfidf={use_tfidf}, "
               f"chi_square={use_chi_square})...")
    
    preprocessor = SQLPreprocessor(
        use_tfidf=use_tfidf,
        use_chi_square_selection=use_chi_square
    )
    
    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_val_p   = preprocessor.transform(X_val)
    X_test_p  = preprocessor.transform(X_test)
    preprocessor.save(suffix=tag or ('full' if use_tfidf else 'structural'))

    logger.info(f"{label} Feature matrix shape after processing: "
               f"train={X_train_p.shape} val={X_val_p.shape} test={X_test_p.shape}")

    # Chi-Square Feature Analysis (even if not used for selection)
    if use_tfidf and tag in ['full', 'full_tuned', 'baseline']:
        logger.info(f"{label} Performing Chi-Square Feature Analysis...")
        try:
            chi_scores, p_values = chi2(X_train_p, y_train)
            
            num_structural = len(preprocessor.structural_feature_names)
            all_feature_names = preprocessor.structural_feature_names.copy()
            if preprocessor.tfidf is not None:
                all_feature_names += [f'tfidf_{i}' for i in range(len(preprocessor.tfidf.vocabulary_))]
            
            feature_types = ['Structural' if i < num_structural else 'TF-IDF' 
                           for i in range(len(all_feature_names))]
            
            chi_results = pd.DataFrame({
                'Feature': all_feature_names,
                'Chi2_Score': chi_scores,
                'P_Value': p_values,
                'Feature_Type': feature_types
            }).sort_values('Chi2_Score', ascending=False)
            
            logger.info("\nTop 30 Features by Chi-Square Score:")
            logger.info("="*80)
            logger.info("\n" + chi_results.head(30).to_string(index=False))
            
            logger.info("\n" + "="*80)
            logger.info("Summary by Feature Type:")
            logger.info("="*80)
            structural_in_top20 = chi_results.head(20)['Feature_Type'].value_counts()
            logger.info(f"In top 20 features:\n{structural_in_top20.to_string()}")
            
            plot_chi_square_analysis(chi_results)
            chi_results.to_csv(f'./figures/chi_square_full_results_{tag}.csv', index=False)
            logger.info(f"{label} Chi-Square analysis saved to ./figures/")
        except Exception as e:
            logger.warning(f"Could not perform chi-square analysis: {e}")

    logger.info(f"{label} Starting cross-validation for confidence intervals...")
    cv_df = cross_validate_models(X_train_p, y_train)
    logger.info(f"\n{label} CV Summary:\n{cv_df.to_string(index=False)}\n")

    logger.info(f"{label} Training models (tune={tune}, ensembles={use_ensembles})...")
    model_handler = ModelHandler(use_ensembles=use_ensembles)
    model_handler.train_all(X_train_p, y_train, tune=tune)

    # Save models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    for name, model in model_handler.models.items():
        safe_name = name.replace(" ", "_")
        path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}_{tag}_model.pkl")
        joblib.dump(model, path)
    logger.info(f"{label} Models serialized to {MODEL_SAVE_DIR}")

    logger.info(f"{label} Evaluating on test set...")
    results_df = evaluate_models(
        model_handler, X_val_p, y_val, X_test_p, y_test, X_test,                      
        preprocessor.feature_names, tag=tag, generate_tsne=generate_tsne
    )

    results_df = results_df.merge(cv_df[['Model', 'CV_Mean_F1', 'CV_Std_F1', 'CV_95CI']],
                                  on='Model', how='left')
    
    # Add feature count information
    results_df['n_features'] = X_train_p.shape[1]
    
    return results_df


def run_comprehensive_experiments():
    """
    Run comprehensive experiment suite comparing all variations
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE SQL INJECTION DETECTION EXPERIMENT SUITE")
    logger.info("=" * 80)
    
    # Dictionary to store all results
    all_results = {}
    
    # ========================================================================
    # EXPERIMENT 1: Baseline (No tuning, no feature selection)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: Baseline (Full features, no tuning)")
    logger.info("="*80)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        undersample_strategy=None  # No undersampling for baseline
    )
    
    baseline_results = run_pipeline(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_tfidf=True, use_chi_square=False, tune=False, 
        use_ensembles=False, tag='baseline', generate_tsne=False
    )
    all_results['Baseline'] = baseline_results
    
    # ========================================================================
    # EXPERIMENT 2: With Hyperparameter Tuning
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: Full features WITH hyperparameter tuning")
    logger.info("="*80)
    
    tuned_results = run_pipeline(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_tfidf=True, use_chi_square=False, tune=True,
        use_ensembles=False, tag='full_tuned', generate_tsne=False
    )
    all_results['Tuned'] = tuned_results
    
    # Compare tuning impact
    plot_tuning_comparison(baseline_results, tuned_results)
    logger.info("Tuning comparison plot saved -> ./figures/tuning_comparison.png")
    
    # ========================================================================
    # EXPERIMENT 3: With Chi-Square Feature Selection
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 3: Chi-Square feature selection (75th percentile)")
    logger.info("="*80)
    
    chi_square_results = run_pipeline(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_tfidf=True, use_chi_square=True, tune=True,
        use_ensembles=False, tag='chi_square_75', generate_tsne=False
    )
    all_results['ChiSquare_75pct'] = chi_square_results
    
    # Compare feature selection impact
    plot_feature_selection_impact(tuned_results, chi_square_results)
    logger.info("Feature selection impact plot saved -> ./figures/feature_selection_impact.png")
    
    # ========================================================================
    # EXPERIMENT 4: Structural Features Only (Ablation)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 4: Ablation - Structural features ONLY")
    logger.info("="*80)
    
    ablation_results = run_pipeline(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_tfidf=False, use_chi_square=False, tune=False,
        use_ensembles=False, tag='ablation', generate_tsne=False
    )
    all_results['Structural_Only'] = ablation_results
    
    # Plot ablation comparison
    plot_ablation_comparison(tuned_results, ablation_results)
    logger.info("Ablation comparison plot saved -> ./figures/ablation_comparison.png")
    
    # ========================================================================
    # EXPERIMENT 5: With Undersampling
    # ========================================================================
    if UNDERSAMPLING_STRATEGY and UNDERSAMPLING_STRATEGY != 'none':
        logger.info("\n" + "="*80)
        logger.info(f"EXPERIMENT 5: With {UNDERSAMPLING_STRATEGY} undersampling")
        logger.info("="*80)
        
        X_train_us, X_val_us, X_test_us, y_train_us, y_val_us, y_test_us = load_and_split_data(
            undersample_strategy=UNDERSAMPLING_STRATEGY,
            undersample_ratio=UNDERSAMPLING_RATIO
        )
        
        undersampled_results = run_pipeline(
            X_train_us, X_val_us, X_test_us, y_train_us, y_val_us, y_test_us,
            use_tfidf=True, use_chi_square=False, tune=True,
            use_ensembles=False, tag=f'undersampled_{UNDERSAMPLING_STRATEGY}',
            generate_tsne=False
        )
        all_results[f'Undersampled_{UNDERSAMPLING_STRATEGY}'] = undersampled_results
    
    # ========================================================================
    # EXPERIMENT 6: Best Configuration with Ensembles
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 6: Best config WITH ensemble models + t-SNE")
    logger.info("="*80)
    
    # Use best configuration (tuned + chi-square + undersampling if enabled)
    if UNDERSAMPLING_STRATEGY and UNDERSAMPLING_STRATEGY != 'none':
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_and_split_data(
            undersample_strategy=UNDERSAMPLING_STRATEGY,
            undersample_ratio=UNDERSAMPLING_RATIO
        )
    else:
        X_tr, X_v, X_te, y_tr, y_v, y_te = X_train, X_val, X_test, y_train, y_val, y_test
    
    best_results = run_pipeline(
        X_tr, X_v, X_te, y_tr, y_v, y_te,
        use_tfidf=True, use_chi_square=True, tune=True,
        use_ensembles=True, tag='best_with_ensembles',
        generate_tsne=True  # Generate t-SNE for final model
    )
    all_results['Best_With_Ensembles'] = best_results
    
    # ========================================================================
    # COMPREHENSIVE COMPARISON AND REPORTING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPREHENSIVE COMPARISON VISUALIZATIONS")
    logger.info("="*80)
    
    # Generate comparison plots
    plot_experiment_comparison(all_results, metric='Test_F1')
    plot_experiment_comparison(all_results, metric='Recall')
    plot_experiment_comparison(all_results, metric='Precision')
    plot_experiment_comparison(all_results, metric='FPR')
    
    # Generate comprehensive summary
    plot_comprehensive_summary(all_results)
    
    # Save comprehensive report
    save_experiment_report(all_results)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("FINAL EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    for exp_name, results_df in all_results.items():
        best_model = results_df.loc[results_df['Test_F1'].idxmax()]
        logger.info(f"\n{exp_name}:")
        logger.info(f"  Best Model: {best_model['Model']}")
        logger.info(f"  Test F1: {best_model['Test_F1']:.4f}")
        logger.info(f"  Precision: {best_model['Precision']:.4f}")
        logger.info(f"  Recall: {best_model['Recall']:.4f}")
        logger.info(f"  FPR: {best_model['FPR']:.4f}")
        if 'n_features' in best_model:
            logger.info(f"  Features: {best_model['n_features']}")
    
    # Find overall best
    best_overall = None
    best_f1 = 0
    best_exp = None
    
    for exp_name, results_df in all_results.items():
        exp_best = results_df['Test_F1'].max()
        if exp_best > best_f1:
            best_f1 = exp_best
            best_exp = exp_name
            best_overall = results_df.loc[results_df['Test_F1'].idxmax()]
    
    logger.info("\n" + "="*80)
    logger.info("OVERALL BEST CONFIGURATION:")
    logger.info("="*80)
    logger.info(f"  Experiment: {best_exp}")
    logger.info(f"  Model: {best_overall['Model']}")
    logger.info(f"  Test F1: {best_overall['Test_F1']:.4f}")
    logger.info(f"  Precision: {best_overall['Precision']:.4f}")
    logger.info(f"  Recall: {best_overall['Recall']:.4f}")
    logger.info(f"  FPR: {best_overall['FPR']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE - All results saved to ./figures/")
    logger.info("="*80)
    
    return all_results


def main() -> None:
    """Run the comprehensive experiment suite"""
    try:
        all_results = run_comprehensive_experiments()
        
        # Save final summary
        logger.info("\nAll experiments completed successfully!")
        logger.info(f"Total experiments run: {len(all_results)}")
        logger.info("\nCheck ./figures/ for all visualizations and reports")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
