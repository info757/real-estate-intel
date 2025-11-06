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
from sklearn.model_selection import train_test_split
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
        
        # Train/validation/test split
        X_train, X_temp, y_fs_train, y_fs_temp, y_dom_train, y_dom_temp = train_test_split(
            X, y_fast_seller, y_dom, test_size=test_size, random_state=42
        )
        
        split_ratio = validation_size / (test_size + validation_size)
        X_val, X_test, y_fs_val, y_fs_test, y_dom_val, y_dom_test = train_test_split(
            X_temp, y_fs_temp, y_dom_temp, test_size=split_ratio, random_state=42
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
            X_train, y_dom_train, X_val, y_dom_val, X_test, y_dom_test,
            hyperparameter_tuning
        )
        
        # Store metadata (thresholds_by_zip should be set before calling train())
        if 'thresholds_by_zip' not in self.model_metadata:
            self.model_metadata['thresholds_by_zip'] = {}
        self.model_metadata.update({
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'classifier_metrics': classifier_metrics,
            'regressor_metrics': regressor_metrics
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
        
        # Base parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if hyperparameter_tuning:
            # Simple grid search (can be expanded)
            best_score = 0
            best_params = params.copy()
            
            for max_depth in [4, 6, 8]:
                for lr in [0.05, 0.1, 0.15]:
                    test_params = params.copy()
                    test_params['max_depth'] = max_depth
                    test_params['learning_rate'] = lr
                    
                    model = xgb.XGBClassifier(**test_params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                    
                    if score > best_score:
                        best_score = score
                        best_params = test_params
            
            params = best_params
            logger.info(f"Best classifier params: max_depth={params['max_depth']}, lr={params['learning_rate']}")
        
        # Train final model
        self.fast_seller_classifier = xgb.XGBClassifier(**params)
        self.fast_seller_classifier.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
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
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameter_tuning: bool
    ) -> Dict[str, Any]:
        """Train the DOM regressor."""
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        # Base parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if hyperparameter_tuning:
            best_score = float('inf')
            best_params = params.copy()
            
            for max_depth in [4, 6, 8]:
                for lr in [0.05, 0.1, 0.15]:
                    test_params = params.copy()
                    test_params['max_depth'] = max_depth
                    test_params['learning_rate'] = lr
                    
                    model = xgb.XGBRegressor(**test_params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    
                    y_pred = model.predict(X_val)
                    score = mean_absolute_error(y_val, y_pred)
                    
                    if score < best_score:
                        best_score = score
                        best_params = test_params
            
            params = best_params
            logger.info(f"Best regressor params: max_depth={params['max_depth']}, lr={params['learning_rate']}")
        
        # Train final model
        self.dom_regressor = xgb.XGBRegressor(**params)
        self.dom_regressor.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.dom_regressor.predict(X_test)
        
        metrics = {
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_mape': mean_absolute_percentage_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_absolute_error(y_test, y_pred))
        }
        
        logger.info(f"Regressor Test MAPE: {metrics['test_mape']:.2f}%")
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
        return self.dom_regressor.predict(X)
    
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
