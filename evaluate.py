import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import CV_N_SPLITS, logger
from visualize import (plot_feature_importance, plot_granular_recall_heatmap, 
                       plot_pr_curve, plot_confusion_matrices, plot_detailed_metrics)


def _optimal_threshold(probs: np.ndarray, y_true) -> tuple[float, float]:
    """Find optimal threshold that maximizes F1 score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    denom = precisions + recalls
    
    f1s = np.zeros_like(denom)
    valid_mask = denom > 0
    f1s[valid_mask] = 2 * precisions[valid_mask] * recalls[valid_mask] / denom[valid_mask]
    
    idx  = int(np.argmax(f1s))
    thr  = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return thr, float(f1s[idx])


def visualize_with_tsne(X_test_processed, y_test, df_test_raw, 
                       model_handler, tag='full'):
    """
    Generate comprehensive t-SNE visualizations including attack type tracking
    
    **CRITICAL: This function generates attack-type-specific visualizations**
    **that show which attacks were caught during runtime**
    """
    try:
        from tsne_visualizer import TSNEVisualizer
    except ImportError:
        logger.warning("tsne_visualizer not found, skipping t-SNE visualizations")
        return None
    
    logger.info("\n" + "="*65)
    logger.info("Generating t-SNE Visualizations with Attack Type Tracking")
    logger.info("="*65)
    
    # Initialize visualizer
    visualizer = TSNEVisualizer(random_state=42)
    
    # Compute embeddings
    try:
        embeddings = visualizer.fit_transform(
            X_test_processed,
            perplexity=30,
            n_iter=1000
        )
    except Exception as e:
        logger.error(f"Failed to compute t-SNE embeddings: {e}")
        return None
    
    # Plot 1: Binary classification (true labels)
    visualizer.plot_binary_classification(
        y_test,
        save_path=f'./figures/tsne_binary_{tag}.png'
    )
    
    # Plot 2: Attack types distribution
    visualizer.plot_attack_types(
        df_test_raw['injection_type'].values,
        y_true=y_test.values,
        save_path=f'./figures/tsne_attack_types_{tag}.png'
    )
    
    # Plot 3-N: Runtime detections for each model
    detection_summaries = {}
    
    for model_name in model_handler.models:
        if model_name in ['IsolationForest']:  # Skip for clarity
            continue
            
        # Get predictions
        probs = model_handler.get_probs(model_name, X_test_processed)
        y_pred = (probs >= 0.5).astype(int)
        
        # Generate visualization showing which attacks were caught
        try:
            detection_rates = visualizer.plot_runtime_detections(
                df_test_raw['injection_type'].values,
                y_test.values,
                y_pred,
                model_name=model_name,
                save_path=f'./figures/tsne_runtime_{model_name}_{tag}.png'
            )
            detection_summaries[model_name] = detection_rates
        except Exception as e:
            logger.warning(f"Failed to generate t-SNE for {model_name}: {e}")
    
    # Save detection summary to CSV
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
        summary_df.to_csv(f'./figures/attack_type_detection_summary_{tag}.csv', 
                         index=False)
        
        logger.info(f"t-SNE visualizations complete. See ./figures/ for all plots.")
        logger.info(f"Attack detection summary saved to "
                   f"./figures/attack_type_detection_summary_{tag}.csv")
    
    return detection_summaries


def evaluate_models(model_handler, X_val_processed, y_val, X_test_processed, 
                   y_test, df_test_raw, feature_names: list[str], 
                   tag: str = 'eval', generate_tsne: bool = True) -> pd.DataFrame:
    """
    Comprehensive model evaluation with t-SNE visualizations
    
    Parameters:
    -----------
    model_handler : ModelHandler
        Trained models
    X_val_processed : sparse matrix
        Validation features
    y_val : array-like
        Validation labels
    X_test_processed : sparse matrix
        Test features
    y_test : array-like
        Test labels
    df_test_raw : DataFrame
        Raw test data with injection types
    feature_names : list
        Feature names
    tag : str
        Tag for output files
    generate_tsne : bool
        Whether to generate t-SNE visualizations
    """
    logger.info("-" * 60)
    logger.info("Phase 1 - Optimal threshold search on VALIDATION set")
    logger.info("-" * 60)

    optimal_thresholds: dict[str, float] = {}
    val_f1_scores:      dict[str, float] = {}
    val_probs_dict:     dict[str, np.ndarray] = {}

    for name in model_handler.models:
        probs = model_handler.get_probs(name, X_val_processed)
        thr, val_f1 = _optimal_threshold(probs, y_val)
        optimal_thresholds[name] = thr
        val_f1_scores[name]      = val_f1
        val_probs_dict[name]     = probs
        logger.info(f"  [{name:20s}]  Val F1={val_f1:.4f}  ->  locked threshold={thr:.4f}")

    logger.info("-" * 60)
    logger.info("Phase 2 - Final evaluation on TEST set (threshold locked from val)")
    logger.info("-" * 60)

    results              = []
    test_probs_dict:  dict[str, np.ndarray] = {}
    test_preds_dict:  dict[str, np.ndarray] = {}
    granular_frames      = []

    best_f1          = 0.0
    best_model_name  = ""

    for name in model_handler.models:
        probs     = model_handler.get_probs(name, X_test_processed)
        threshold = optimal_thresholds[name]
        preds     = (probs >= threshold).astype(int)

        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        test_f1   = f1_score(y_test, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        logger.info(
            f"  [{name:20s}]  thr={threshold:.4f}  Acc={accuracy:.4f}  "
            f"P={precision:.4f}  R={recall:.4f}  F1={test_f1:.4f}  FPR={fpr:.4f}"
        )

        test_probs_dict[name] = probs
        test_preds_dict[name] = preds

        # Get training time if available
        train_time = model_handler.training_times.get(name, 0.0)
        inference_time = model_handler.inference_times.get(name, 0.0)

        results.append({
            'Model':         name,
            'Val_F1':        val_f1_scores[name],
            'Opt_Threshold': threshold,
            'Accuracy':      accuracy,
            'Precision':     precision,
            'Recall':        recall,
            'Test_F1':       test_f1,
            'FPR':           fpr,
            'TNR':           tnr,
            'FNR':           fnr,
            'Train_Time_s':  train_time,
            'Inference_ms':  inference_time,
        })

        tmp = df_test_raw.copy()
        tmp['true_label'] = y_test.values
        tmp['pred_label'] = preds
        tmp['model_name'] = name
        granular_frames.append(tmp)

        if name not in ['IsolationForest'] and test_f1 > best_f1:
            best_f1         = test_f1
            best_model_name = name

    logger.info("Generating standard evaluation figures in ./figures/ ...")
    plot_pr_curve(y_test, test_probs_dict, tag=tag)
    plot_granular_recall_heatmap(pd.concat(granular_frames), tag=tag)
    plot_confusion_matrices(y_test, test_preds_dict, tag=tag)
    results_df = pd.DataFrame(results)
    plot_detailed_metrics(results_df, tag=tag)

    importances = model_handler.get_feature_importances(best_model_name)
    if importances is not None:
        plot_feature_importance(importances, feature_names, tag=tag)

    # Generate t-SNE visualizations with attack type tracking
    if generate_tsne:
        logger.info("\n" + "="*60)
        logger.info("⚠️  GENERATING T-SNE VISUALIZATIONS WITH ATTACK TYPE TRACKING")
        logger.info("="*60)
        
        detection_summaries = visualize_with_tsne(
            X_test_processed, y_test, df_test_raw,
            model_handler, tag=tag
        )
        
        # Attach detection summaries to results
        if detection_summaries:
            for idx, row in results_df.iterrows():
                model_name = row['Model']
                if model_name in detection_summaries:
                    # Calculate average detection rate across attack types
                    avg_detection = np.mean([
                        stats['rate'] for stats in detection_summaries[model_name].values()
                    ])
                    results_df.loc[idx, 'Avg_Attack_Detection_Rate'] = avg_detection

    return results_df


def _make_cv_models(pos_weight: float) -> dict:
    """Create models for cross-validation"""
    return {
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=pos_weight, n_estimators=100, eval_metric='logloss', random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(scale_pos_weight=pos_weight, n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        'IsolationForest': IsolationForest(contamination=0.20, random_state=42, n_jobs=-1),
    }


def cross_validate_models(X_train_processed, y_train, n_splits: int = CV_N_SPLITS) -> pd.DataFrame:
    """Perform stratified k-fold cross-validation"""
    logger.info("-" * 60)
    logger.info(f"Cross-validation: {n_splits}-fold stratified KFold on training data")
    logger.info("-" * 60)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_mal = int((y_train == 1).sum())
    n_ben = int((y_train == 0).sum())
    pos_w = n_ben / max(n_mal, 1)

    model_names = list(_make_cv_models(pos_w).keys())
    fold_f1s    = {name: [] for name in model_names}
    y_arr = np.asarray(y_train)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_arr), start=1):
        logger.info(f"  Fold {fold_idx}/{n_splits} ...")

        X_ft = X_train_processed[train_idx]
        y_ft = y_arr[train_idx]
        X_fv = X_train_processed[val_idx]
        y_fv = y_arr[val_idx]

        fold_pos_w = (y_ft == 0).sum() / max((y_ft == 1).sum(), 1)
        fold_models = _make_cv_models(float(fold_pos_w))

        for name, model in fold_models.items():
            if name == 'IsolationForest':
                benign_mask  = y_ft == 0
                X_ben        = X_ft[benign_mask]
                model.fit(X_ben)
                train_scores = model.decision_function(X_ben)
                s_min, s_max = train_scores.min(), train_scores.max()
                denom        = s_max - s_min if s_max != s_min else 1.0
                val_scores   = model.decision_function(X_fv)
                probs        = (s_max - val_scores) / denom
            else:
                model.fit(X_ft, y_ft)
                probs = model.predict_proba(X_fv)[:, 1]

            thr, _ = _optimal_threshold(probs, y_fv)
            preds  = (probs >= thr).astype(int)
            fold_f1s[name].append(f1_score(y_fv, preds, zero_division=0))

    logger.info("-" * 60)
    logger.info(f"{'Model':<20}  {'Mean F1':>8}  {'Std':>7}  {'95% CI':>10}")
    logger.info("-" * 60)

    summary = []
    for name in model_names:
        scores  = np.array(fold_f1s[name])
        mean    = scores.mean()
        std     = scores.std(ddof=1)
        ci95    = 1.96 * std / np.sqrt(n_splits)
        logger.info(f"  {name:<20}  {mean:.4f}    {std:.4f}    +/-{ci95:.4f}")
        summary.append({
            'Model':      name,
            'CV_Mean_F1': mean,
            'CV_Std_F1':  std,
            'CV_95CI':    ci95,
        })

    return pd.DataFrame(summary)
