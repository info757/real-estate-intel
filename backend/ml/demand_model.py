"""
Demand Model
XGBoost models to predict sales probability and days on market (DOM).
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_percentage_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandModel:
    """
    XGBoost models for predicting property demand:
    - Classification: Probability of selling within X days
    - Regression: Expected days on market (DOM)
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize demand model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Two models: one for classification (sell probability), one for regression (DOM)
        self.sell_probability_model = None
        self.dom_model = None
        
        self.feature_names = None
        self.model_metadata = {}
        self.sell_within_days = 90  # Default: probability of selling within 90 days
        
    def train(
        self,
        X: pd.DataFrame,
        y_sold_fast: pd.Series,  # Binary: 1 if sold within threshold days, 0 otherwise
        y_dom: pd.Series,  # Continuous: actual days on market
        sell_within_days: int = 90,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Train both demand models.
        
        Args:
            X: Feature DataFrame
            y_sold_fast: Binary target (1 = sold within threshold, 0 = otherwise)
            y_dom: Continuous target (days on market)
            sell_within_days: Threshold for "fast sale" classification
            test_size: Fraction of data for test set
            validation_size: Fraction of training data for validation
            hyperparameter_tuning: Whether to perform grid search
            
        Returns:
            Training metrics dictionary for both models
        """
        logger.info(f"Training demand models on {len(X)} samples")
        
        self.sell_within_days = sell_within_days
        self.feature_names = X.columns.tolist()
        
        # Train/validation/test split
        X_train, X_temp, y_prob_train, y_prob_temp, y_dom_train, y_dom_temp = train_test_split(
            X, y_sold_fast, y_dom, test_size=test_size, random_state=42
        )
        
        X_val, X_test, y_prob_val, y_prob_test, y_dom_val, y_dom_test = train_test_split(
            X_temp, y_prob_temp, y_dom_temp,
            test_size=validation_size / (test_size + validation_size),
            random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Train sell probability model (classification)
        logger.info("Training sell probability model...")
        prob_metrics = self._train_sell_probability_model(
            X_train, y_prob_train, X_val, y_prob_val, X_test, y_prob_test,
            hyperparameter_tuning
        )
        
        # Train DOM model (regression)
        logger.info("Training DOM model...")
        dom_metrics = self._train_dom_model(
            X_train, y_dom_train, X_val, y_dom_val, X_test, y_dom_test,
            hyperparameter_tuning
        )
        
        # Combine metrics
        metrics = {
            'sell_probability': prob_metrics,
            'dom': dom_metrics,
            'sell_within_days': sell_within_days,
            'feature_count': len(self.feature_names),
            'sample_count': len(X),
            'training_date': datetime.now().isoformat()
        }
        
        # Store metadata
        self.model_metadata = {
            'feature_names': self.feature_names,
            'training_metrics': metrics,
            'trained_date': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Training complete! Sell probability AUC: {prob_metrics['test_auc']:.3f}, DOM MAPE: {dom_metrics['test_mape']:.2f}%")
        
        return metrics
    
    def _train_sell_probability_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameter_tuning: bool
    ) -> Dict[str, Any]:
        """Train classification model for sell probability."""
        # Default parameters
        default_params = {
            'objective': 'binary:logistic',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc'
        }
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
            }
            
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )
            
            logger.info("Tuning sell probability model hyperparameters...")
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            params = {**default_params, **grid_search.best_params_}
        else:
            params = default_params
        
        # Train model
        # XGBoost 3.x: early_stopping_rounds is a constructor parameter
        self.sell_probability_model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=10
        )
        self.sell_probability_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.sell_probability_model.predict_proba(X_train)[:, 1]
        val_pred = self.sell_probability_model.predict_proba(X_val)[:, 1]
        test_pred = self.sell_probability_model.predict_proba(X_test)[:, 1]
        
        train_pred_binary = self.sell_probability_model.predict(X_train)
        val_pred_binary = self.sell_probability_model.predict(X_val)
        test_pred_binary = self.sell_probability_model.predict(X_test)
        
        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred),
            'train_accuracy': accuracy_score(y_train, train_pred_binary),
            'train_precision': precision_score(y_train, train_pred_binary, zero_division=0),
            'train_recall': recall_score(y_train, train_pred_binary, zero_division=0),
            'train_f1': f1_score(y_train, train_pred_binary, zero_division=0),
            
            'val_auc': roc_auc_score(y_val, val_pred),
            'val_accuracy': accuracy_score(y_val, val_pred_binary),
            'val_precision': precision_score(y_val, val_pred_binary, zero_division=0),
            'val_recall': recall_score(y_val, val_pred_binary, zero_division=0),
            'val_f1': f1_score(y_val, val_pred_binary, zero_division=0),
            
            'test_auc': roc_auc_score(y_test, test_pred),
            'test_accuracy': accuracy_score(y_test, test_pred_binary),
            'test_precision': precision_score(y_test, test_pred_binary, zero_division=0),
            'test_recall': recall_score(y_test, test_pred_binary, zero_division=0),
            'test_f1': f1_score(y_test, test_pred_binary, zero_division=0),
        }
        
        return metrics
    
    def _train_dom_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameter_tuning: bool
    ) -> Dict[str, Any]:
        """Train regression model for days on market."""
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
            }
            
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("Tuning DOM model hyperparameters...")
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=5, scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            params = {**default_params, **grid_search.best_params_}
        else:
            params = default_params
        
        # Train model
        # XGBoost 3.x: early_stopping_rounds is a constructor parameter
        self.dom_model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=10
        )
        self.dom_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.dom_model.predict(X_train)
        val_pred = self.dom_model.predict(X_val)
        test_pred = self.dom_model.predict(X_test)
        
        # Handle zeros/very small values for MAPE
        def safe_mape(y_true, y_pred):
            mask = y_true > 0
            if mask.sum() == 0:
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        metrics = {
            'train_mape': safe_mape(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            
            'val_mape': safe_mape(y_val, val_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
            
            'test_mape': safe_mape(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
        }
        
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Predict both sell probability and expected DOM.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with 'sell_probability' and 'expected_dom' arrays
        """
        if self.sell_probability_model is None or self.dom_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Ensure feature order matches training
        X = X[self.feature_names].copy()
        X = X.fillna(X.median())
        
        # Predict
        sell_probability = self.sell_probability_model.predict_proba(X)[:, 1]
        expected_dom = self.dom_model.predict(X)
        
        return {
            'sell_probability': sell_probability,
            'expected_dom': expected_dom
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from both models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with importance DataFrames for both models
        """
        if self.sell_probability_model is None or self.dom_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        importance_prob = self.sell_probability_model.feature_importances_
        importance_dom = self.dom_model.feature_importances_
        
        prob_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_prob
        }).sort_values('importance', ascending=False)
        prob_df['importance_pct'] = (prob_df['importance'] / prob_df['importance'].sum() * 100)
        
        dom_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_dom
        }).sort_values('importance', ascending=False)
        dom_df['importance_pct'] = (dom_df['importance'] / dom_df['importance'].sum() * 100)
        
        return {
            'sell_probability': prob_df.head(top_n),
            'dom': dom_df.head(top_n)
        }
    
    def save(self, model_name: str = "demand_model") -> str:
        """Save both models to disk."""
        if self.sell_probability_model is None or self.dom_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        model_path_prob = self.model_dir / f"{model_name}_probability.pkl"
        model_path_dom = self.model_dir / f"{model_name}_dom.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        joblib.dump(self.sell_probability_model, model_path_prob)
        joblib.dump(self.dom_model, model_path_dom)
        logger.info(f"Saved models to {self.model_dir}")
        
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return str(self.model_dir)
    
    def load(self, model_name: str = "demand_model") -> 'DemandModel':
        """Load both models from disk."""
        model_path_prob = self.model_dir / f"{model_name}_probability.pkl"
        model_path_dom = self.model_dir / f"{model_name}_dom.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        if not model_path_prob.exists() or not model_path_dom.exists():
            raise FileNotFoundError(f"Models not found in {self.model_dir}")
        
        self.sell_probability_model = joblib.load(model_path_prob)
        self.dom_model = joblib.load(model_path_dom)
        logger.info(f"Loaded models from {self.model_dir}")
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            self.feature_names = self.model_metadata.get('feature_names', [])
            self.sell_within_days = self.model_metadata.get('training_metrics', {}).get('sell_within_days', 90)
            logger.info(f"Loaded metadata from {metadata_path}")
        
        return self


# Singleton instance
demand_model = DemandModel()
