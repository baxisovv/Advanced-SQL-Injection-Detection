import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, IsolationForest, 
                              RandomForestClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logger, TUNE_HYPERPARAMS

# Enhanced hyperparameter grids with more aggressive search
_PARAM_GRIDS: dict[str, dict] = {
    'DecisionTree': {
        'max_depth':         [5, 10, 15, 20], 
        'min_samples_split': [10, 20, 50, 100],
        'min_samples_leaf':  [5, 10, 20, 30],
    },
    'RandomForest': {
        'n_estimators':      [100, 200, 300, 400],
        'max_depth':         [10, 15, 20, 25], 
        'min_samples_split': [10, 20, 50],
        'max_features':      ['sqrt', 'log2'],
        'min_samples_leaf':  [1, 2, 4],
    },
    'ExtraTrees': {
        'n_estimators':      [100, 200, 300, 400],
        'max_depth':         [10, 15, 20, 25],
        'min_samples_split': [10, 20, 50],
        'max_features':      ['sqrt', 'log2'],
    },
    'XGBoost': {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [4, 6, 8, 10],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'subsample':         [0.7, 0.8, 0.9],
        'colsample_bytree':  [0.7, 0.8, 0.9],
        'min_child_weight':  [3, 5, 10],
        'gamma':             [0, 0.1, 0.2],
    },
    'LightGBM': {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [10, 15, 20],
        'learning_rate':     [0.01, 0.05, 0.1],
        'num_leaves':        [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'subsample':         [0.7, 0.8, 0.9],
    },
}


class ModelHandler:
    """
    Enhanced Model Handler with Ensemble Support
    
    Manages training, prediction, and evaluation of multiple models including:
    - Individual tree-based models
    - Isolation Forest for anomaly detection
    - Voting and stacking ensembles
    """
    
    def __init__(self, use_ensembles: bool = True) -> None:
        self.models: dict = {}
        self.training_times:  dict[str, float] = {}
        self.inference_times: dict[str, float] = {}
        self.if_score_min: float | None = None
        self.if_score_max: float | None = None
        self.use_ensembles = use_ensembles

    def _make_models(self, pos_weight: float) -> dict:
        """Create base models with balanced class weights"""
        return {
            'DecisionTree': DecisionTreeClassifier(
                class_weight='balanced', random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                scale_pos_weight=pos_weight,
                eval_metric='logloss', random_state=42, n_jobs=-1
            ),
            'LightGBM': LGBMClassifier(
                scale_pos_weight=pos_weight,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'IsolationForest': IsolationForest(
                contamination=0.20,
                random_state=42, n_jobs=-1
            ),
        }

    def create_voting_ensemble(self):
        """Create soft voting ensemble from best models"""
        if 'RandomForest' not in self.models or 'XGBoost' not in self.models:
            logger.warning("Cannot create ensemble - base models not trained")
            return None
        
        estimators = []
        weights = []
        
        # Add models with their relative weights based on typical performance
        if 'RandomForest' in self.models:
            estimators.append(('rf', self.models['RandomForest']))
            weights.append(0.25)
        
        if 'XGBoost' in self.models:
            estimators.append(('xgb', self.models['XGBoost']))
            weights.append(0.30)
        
        if 'LightGBM' in self.models:
            estimators.append(('lgbm', self.models['LightGBM']))
            weights.append(0.30)
        
        if 'ExtraTrees' in self.models:
            estimators.append(('et', self.models['ExtraTrees']))
            weights.append(0.15)
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        logger.info(f"Created voting ensemble with {len(estimators)} models")
        return voting
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with logistic regression as meta-model"""
        if 'RandomForest' not in self.models or 'XGBoost' not in self.models:
            logger.warning("Cannot create ensemble - base models not trained")
            return None
        
        estimators = []
        
        if 'RandomForest' in self.models:
            estimators.append(('rf', self.models['RandomForest']))
        
        if 'XGBoost' in self.models:
            estimators.append(('xgb', self.models['XGBoost']))
        
        if 'LightGBM' in self.models:
            estimators.append(('lgbm', self.models['LightGBM']))
        
        # Use logistic regression as meta-learner
        meta_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5
        )
        
        logger.info(f"Created stacking ensemble with {len(estimators)} base models")
        return stacking

    def train_all(self, X_train, y_train, tune: bool | None = None) -> None:
        """
        Train all models with optional hyperparameter tuning
        
        Parameters:
        -----------
        X_train : sparse matrix
            Training features
        y_train : array-like
            Training labels
        tune : bool
            Whether to perform hyperparameter tuning
        """
        if tune is None:
            tune = TUNE_HYPERPARAMS

        n_malicious = int((y_train == 1).sum())
        n_benign    = int((y_train == 0).sum())
        pos_weight  = n_benign / max(n_malicious, 1)

        self.models = self._make_models(pos_weight)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logger.info("Initiating model training phase...")
        logger.info(f"  Training set: {len(y_train)} samples "
                   f"(benign={n_benign}, malicious={n_malicious})")
        
        for name, model in list(self.models.items()):
            t0 = time.time()

            if name == 'IsolationForest':
                benign_mask = (y_train == 0).to_numpy()
                X_benign    = X_train[benign_mask]
                model.fit(X_benign)

                train_scores       = model.decision_function(X_benign)
                self.if_score_min  = float(train_scores.min())
                self.if_score_max  = float(train_scores.max())
                logger.info(
                    f"  {name:18s} trained on {X_benign.shape[0]} benign samples. "
                    f"Score range: [{self.if_score_min:.4f}, {self.if_score_max:.4f}]"
                )

            elif tune and name in _PARAM_GRIDS:
                logger.info(f"  {name:18s} hyperparameter search (n_iter=15)...")
                search = RandomizedSearchCV(
                    model, _PARAM_GRIDS[name],
                    n_iter=15, scoring='f1', cv=cv,
                    random_state=42, n_jobs=-1, verbose=0,
                    refit=True,
                )
                search.fit(X_train, y_train)
                self.models[name] = search.best_estimator_
                logger.info(f"    Best params: {search.best_params_}")
                logger.info(f"    Best CV F1 : {search.best_score_:.4f}")

            else:
                model.fit(X_train, y_train)

            elapsed = time.time() - t0
            self.training_times[name] = elapsed
            logger.info(f"  {name:18s} -> {elapsed:.2f}s")
        
        # Train ensembles if enabled
        if self.use_ensembles:
            logger.info("\nTraining ensemble models...")
            
            # Voting Ensemble
            voting = self.create_voting_ensemble()
            if voting is not None:
                t0 = time.time()
                voting.fit(X_train, y_train)
                elapsed = time.time() - t0
                self.models['VotingEnsemble'] = voting
                self.training_times['VotingEnsemble'] = elapsed
                logger.info(f"  VotingEnsemble     -> {elapsed:.2f}s")
            
            # Stacking Ensemble
            stacking = self.create_stacking_ensemble()
            if stacking is not None:
                t0 = time.time()
                stacking.fit(X_train, y_train)
                elapsed = time.time() - t0
                self.models['StackingEnsemble'] = stacking
                self.training_times['StackingEnsemble'] = elapsed
                logger.info(f"  StackingEnsemble   -> {elapsed:.2f}s")

    def get_probs(self, name: str, X) -> np.ndarray:
        """Get probability estimates for positive class"""
        model = self.models[name]

        if name == 'IsolationForest':
            scores    = model.decision_function(X)
            s_min     = self.if_score_min if self.if_score_min is not None else scores.min()
            s_max     = self.if_score_max if self.if_score_max is not None else scores.max()
            denom     = s_max - s_min
            if denom == 0:
                return np.full(len(scores), 0.5)
            return (s_max - scores) / denom

        return model.predict_proba(X)[:, 1]

    def predict(self, model_name: str, X) -> np.ndarray:
        """Make predictions with specified model"""
        model  = self.models[model_name]
        t0     = time.time()
        preds  = model.predict(X)
        elapsed = time.time() - t0
        self.inference_times[model_name] = elapsed / X.shape[0] * 1000 

        if model_name == 'IsolationForest':
            preds = np.where(preds == -1, 1, 0)

        return preds

    def get_feature_importances(self, model_name: str) -> np.ndarray | None:
        """Get feature importances if available"""
        model = self.models.get(model_name)
        if model is not None and hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def get_training_summary(self) -> dict:
        """Get summary of training times and model counts"""
        return {
            'n_models': len(self.models),
            'total_training_time': sum(self.training_times.values()),
            'training_times': self.training_times,
            'models_trained': list(self.models.keys())
        }
