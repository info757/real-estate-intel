"""
Fast Seller Model
XGBoost models to predict fast sellers and DOM to pending.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastSellerModel:
    """
    XGBoost models for predicting fast sellers and DOM to pending.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize fast seller model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.fast_seller_classifier = None  # Binary classifier: is_fast_seller
        self.dom_regressor = None  # Regression: dom_to_pending
        self.feature_names = None
        self.model_metadata = {}
        
    def train(
        self,
        X: pd.DataFrame,
        y_fast_seller: pd.Series,  # Binary: 1 if DOM <= threshold, 0 otherwise
        y_dom: pd.Series,  # Continuous: DOM to pending
        test_size: float = 0.2,
        validation_size: float = 0.2,
        hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Train both fast seller classifier and DOM regressor.
        
        Args:
            X: Feature DataFrame
            y_fast_seller: Binary target (1 = fast seller)
            y_dom: Continuous target (DOM to pending)
            test_size: Fraction for test set
            validation_size: Fraction of training for validation
            fast_seller_threshold: DOM threshold for fast seller
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training fast seller models on {len(X)} samples")
        
        self.feature_names = X.columns.tolist()
        
        # Apply log transform to DOM target for stability
        y_dom = y_dom.clip(lower=0)
        y_dom_log = np.log1p(y_dom)
        
        # Train/validation/test split with stratification when possible
        try:
            X_train, X_temp, y_fs_train, y_fs_temp, y_dom_log_train, y_dom_log_temp, y_dom_train_orig, y_dom_temp_orig = train_test_split(
                X, y_fast_seller, y_dom_log, y_dom, test_size=test_size, random_state=42, stratify=y_fast_seller
            )
        except ValueError:
            X_train, X_temp, y_fs_train, y_fs_temp, y_dom_log_train, y_dom_log_temp, y_dom_train_orig, y_dom_temp_orig = train_test_split(
                X, y_fast_seller, y_dom_log, y_dom, test_size=test_size, random_state=42
            )
        
        split_ratio = validation_size / (test_size + validation_size)
        try:
            X_val, X_test, y_fs_val, y_fs_test, y_dom_log_val, y_dom_log_test, y_dom_val_orig, y_dom_test_orig = train_test_split(
                X_temp, y_fs_temp, y_dom_log_temp, y_dom_temp_orig, test_size=split_ratio, random_state=42, stratify=y_fs_temp
            )
        except ValueError:
            X_val, X_test, y_fs_val, y_fs_test, y_dom_log_val, y_dom_log_test, y_dom_val_orig, y_dom_test_orig = train_test_split(
                X_temp, y_fs_temp, y_dom_log_temp, y_dom_temp_orig, test_size=split_ratio, random_state=42
            )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train fast seller classifier
        logger.info("Training fast seller classifier...")
        classifier_metrics = self._train_fast_seller_classifier(
            X_train, y_fs_train, X_val, y_fs_val, X_test, y_fs_test,
            hyperparameter_tuning
        )
        
        # Train DOM regressor
        logger.info("Training DOM regressor...")
        regressor_metrics = self._train_dom_regressor(
            X_train, y_dom_log_train, y_dom_train_orig,
            X_val, y_dom_log_val, y_dom_val_orig,
            X_test, y_dom_log_test, y_dom_test_orig,
            hyperparameter_tuning
        )
        
        # Store metadata (thresholds_by_zip should be set before calling train())
        if 'thresholds_by_zip' not in self.model_metadata:
            self.model_metadata['thresholds_by_zip'] = {}
        self.model_metadata.update({
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'classifier_metrics': classifier_metrics,
            'regressor_metrics': regressor_metrics,
            'target_transform': 'log1p_dom'
        })
        
        return {
            'classifier': classifier_metrics,
            'regressor': regressor_metrics
        }
    
    def _train_fast_seller_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameter_tuning: bool
    ) -> Dict[str, Any]:
        """Train the binary classifier for fast sellers."""
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 400,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'use_label_encoder': False,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        best_params = base_params.copy()
        if hyperparameter_tuning:
            param_grid = [
                {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.8},
                {'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.8},
                {'max_depth': 6, 'learning_rate': 0.12, 'subsample': 0.8, 'colsample_bytree': 0.8},
                {'max_depth': 8, 'learning_rate': 0.10, 'subsample': 0.9, 'colsample_bytree': 0.9}
            ]
            X_cv = pd.concat([X_train, X_val])
            y_cv = pd.concat([y_train, y_val])
            class_counts = y_cv.value_counts()
            min_class = int(class_counts.min()) if not class_counts.empty else 0
            if min_class < 2 or len(X_cv) < 10:
                logger.warning("Insufficient data for stratified CV; skipping classifier tuning.")
                hyperparameter_tuning = False
            else:
                n_splits = min(5, min_class)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                best_score = -np.inf
                
                for grid_params in param_grid:
                    params = base_params.copy()
                    params.update(grid_params)
                    scores = []
                    for train_idx, val_idx in skf.split(X_cv, y_cv):
                        model = xgb.XGBClassifier(**params)
                        model.fit(
                            X_cv.iloc[train_idx], y_cv.iloc[train_idx],
                            eval_set=[(X_cv.iloc[val_idx], y_cv.iloc[val_idx])],
                            verbose=False
                        )
                        y_pred_proba = model.predict_proba(X_cv.iloc[val_idx])[:, 1]
                        try:
                            score = roc_auc_score(y_cv.iloc[val_idx], y_pred_proba)
                        except ValueError:
                            score = 0.5
                        scores.append(score)
                    mean_score = float(np.mean(scores))
                    logger.info(f"Classifier CV params={grid_params} -> AUC={mean_score:.3f}")
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                logger.info(f"Selected classifier params: {best_params}")
        
        if not hyperparameter_tuning:
            best_params = base_params.copy()
        
        # Train final model on train+val
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        self.fast_seller_classifier = xgb.XGBClassifier(**best_params)
        self.fast_seller_classifier.fit(
            X_train_full, y_train_full,
            eval_set=[(X_train_full, y_train_full), (X_test, y_test)],
            verbose=False
        )
        
        # Evaluate on test
        y_pred_proba = self.fast_seller_classifier.predict_proba(X_test)[:, 1]
        y_pred = self.fast_seller_classifier.predict(X_test)
        
        metrics = {
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"Classifier Test AUC: {metrics['test_auc']:.3f}")
        return metrics
    
    def _train_dom_regressor(
        self,
        X_train: pd.DataFrame,
        y_train_log: pd.Series,
        y_train_orig: pd.Series,
        X_val: pd.DataFrame,
        y_val_log: pd.Series,
        y_val_orig: pd.Series,
        X_test: pd.DataFrame,
        y_test_log: pd.Series,
        y_test_orig: pd.Series,
        hyperparameter_tuning: bool
    ) -> Dict[str, Any]:
        """Train the DOM regressor."""
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        best_params = base_params.copy()
        if hyperparameter_tuning:
            param_grid = [
                {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9},
                {'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.9},
                {'max_depth': 6, 'learning_rate': 0.12, 'subsample': 0.8},
                {'max_depth': 8, 'learning_rate': 0.10, 'subsample': 0.9}
            ]
            X_cv = pd.concat([X_train, X_val])
            y_cv_log = pd.concat([y_train_log, y_val_log])
            y_cv_orig = pd.concat([y_train_orig, y_val_orig])
            if len(X_cv) < 20:
                logger.warning("Insufficient samples for K-fold CV; skipping regressor tuning.")
                hyperparameter_tuning = False
            else:
                kf = KFold(n_splits=min(5, len(X_cv)), shuffle=True, random_state=42)
                best_score = float('inf')
                
                for grid_params in param_grid:
                    params = base_params.copy()
                    params.update(grid_params)
                    scores = []
                    for train_idx, val_idx in kf.split(X_cv):
                        model = xgb.XGBRegressor(**params)
                        model.fit(
                            X_cv.iloc[train_idx], y_cv_log.iloc[train_idx],
                            eval_set=[(X_cv.iloc[val_idx], y_cv_log.iloc[val_idx])],
                            verbose=False
                        )
                        y_pred_log = model.predict(X_cv.iloc[val_idx])
                        y_pred = np.expm1(y_pred_log)
                        y_true = np.maximum(y_cv_orig.iloc[val_idx], 0)
                        score = mean_absolute_error(y_true, y_pred)
                        scores.append(score)
                    mean_score = float(np.mean(scores))
                    logger.info(f"Regressor CV params={grid_params} -> MAE={mean_score:.2f}")
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = params
                logger.info(f"Selected regressor params: {best_params}")
        
        if not hyperparameter_tuning:
            best_params = base_params.copy()
        
        # Train final model on train+val data
        X_train_full = pd.concat([X_train, X_val])
        y_train_full_log = pd.concat([y_train_log, y_val_log])
        self.dom_regressor = xgb.XGBRegressor(**best_params)
        self.dom_regressor.fit(
            X_train_full, y_train_full_log,
            eval_set=[(X_train_full, y_train_full_log)],
            verbose=False
        )
        
        # Evaluate on held-out test set
        y_pred_log = self.dom_regressor.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 0, None)
        y_test = np.maximum(y_test_orig, 0)
        
        mae = mean_absolute_error(y_test, y_pred)
        y_safe = np.where(y_test == 0, 1e-6, y_test)
        mape = float(np.mean(np.abs((y_test - y_pred) / y_safe)) * 100)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics = {
            'test_mae': mae,
            'test_mape': mape,
            'test_r2': r2,
            'test_rmse': rmse
        }
        
        logger.info(f"Regressor Test MAPE: {mape:.2f}%")
        return metrics
    
    def predict_fast_seller_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of being a fast seller."""
        if self.fast_seller_classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = X.fillna(0)
        return self.fast_seller_classifier.predict_proba(X)[:, 1]
    
    def predict_dom(self, X: pd.DataFrame) -> np.ndarray:
        """Predict DOM to pending."""
        if self.dom_regressor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = X.fillna(0)
        preds_log = self.dom_regressor.predict(X)
        preds = np.expm1(preds_log)
        return np.clip(preds, 0, None)
    
    def get_feature_importance(self, model_type: str = 'classifier') -> Dict[str, float]:
        """Get feature importance rankings."""
        if model_type == 'classifier':
            model = self.fast_seller_classifier
        else:
            model = self.dom_regressor
        
        if model is None:
            return {}
        
        importance = model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self):
        """Save models and metadata."""
        if self.fast_seller_classifier is not None:
            joblib.dump(self.fast_seller_classifier, self.model_dir / 'fast_seller_classifier.pkl')
        if self.dom_regressor is not None:
            joblib.dump(self.dom_regressor, self.model_dir / 'dom_regressor.pkl')
        
        if self.feature_names:
            with open(self.model_dir / 'fast_seller_feature_names.json', 'w') as f:
                json.dump(self.feature_names, f)
        
        with open(self.model_dir / 'fast_seller_metadata.json', 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Saved fast seller models to {self.model_dir}")
    
    def load(self):
        """Load models and metadata."""
        classifier_path = self.model_dir / 'fast_seller_classifier.pkl'
        regressor_path = self.model_dir / 'dom_regressor.pkl'
        
        if classifier_path.exists():
            self.fast_seller_classifier = joblib.load(classifier_path)
        if regressor_path.exists():
            self.dom_regressor = joblib.load(regressor_path)
        
        feature_names_path = self.model_dir / 'fast_seller_feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        metadata_path = self.model_dir / 'fast_seller_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        logger.info("Loaded fast seller models")


# Singleton instance
fast_seller_model = FastSellerModel()
