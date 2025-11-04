"""
Pricing Model
XGBoost regression model to predict sale prices for properties.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricingModel:
    """
    XGBoost regression model for predicting property sale prices.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize pricing model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.model_metadata = {}
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        hyperparameter_tuning: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the pricing model.
        
        Args:
            X: Feature DataFrame
            y: Target series (sale prices)
            test_size: Fraction of data for test set
            validation_size: Fraction of training data for validation
            hyperparameter_tuning: Whether to perform grid search
            cv_folds: Number of CV folds for tuning
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training pricing model on {len(X)} samples with {len(X.columns)} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train/validation/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=validation_size / (test_size + validation_size), random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Default hyperparameters
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
            logger.info("Performing hyperparameter tuning...")
            params = self._tune_hyperparameters(
                X_train, y_train, X_val, y_val, cv_folds=cv_folds
            )
        else:
            params = default_params
        
        # Train final model
        logger.info("Training final model...")
        # XGBoost 3.x: early_stopping_rounds is a constructor parameter
        self.model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=10
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on test set
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mape': mean_absolute_percentage_error(y_train, train_pred) * 100,
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            
            'val_mape': mean_absolute_percentage_error(y_val, val_pred) * 100,
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
            
            'test_mape': mean_absolute_percentage_error(y_test, test_pred) * 100,
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
            
            'feature_count': len(self.feature_names),
            'sample_count': len(X),
            'training_date': datetime.now().isoformat(),
            'hyperparameters': params
        }
        
        # Store metadata
        self.model_metadata = {
            'feature_names': self.feature_names,
            'training_metrics': metrics,
            'trained_date': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Training complete! Test MAPE: {metrics['test_mape']:.2f}%")
        
        return metrics
    
    def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            cv_folds: Number of CV folds
            
        Returns:
            Best hyperparameters
        """
        # Grid of parameters to search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search (limited search for speed)
        logger.info("Running grid search (this may take a few minutes)...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        # Add default parameters
        best_params['objective'] = 'reg:squarederror'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
        return best_params
    
    def predict(
        self,
        X: pd.DataFrame,
        return_intervals: bool = False,
        confidence_level: float = 0.8
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict sale prices.
        
        Args:
            X: Feature DataFrame
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals (0-1)
            
        Returns:
            Predictions (and optionally intervals)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches training
        X = X[self.feature_names].copy()
        X = X.fillna(X.median())
        
        # Predict
        predictions = self.model.predict(X)
        
        if return_intervals:
            # Approximate prediction intervals using tree variance
            # (XGBoost doesn't provide uncertainty directly)
            leaf_indices = self.model.apply(X)
            # Simple heuristic: use prediction ± percentage based on model confidence
            # For production, consider using quantile regression or conformal prediction
            uncertainty = np.abs(predictions) * 0.10  # 10% uncertainty estimate
            
            alpha = 1 - confidence_level
            z_score = 1.28  # For 80% confidence (1.96 for 95%)
            
            lower = predictions - z_score * uncertainty
            upper = predictions + z_score * uncertainty
            
            return predictions, lower, upper
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importance (gain)
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Calculate percentage
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df.head(top_n)
    
    def save(self, model_name: str = "pricing_model") -> str:
        """
        Save model to disk.
        
        Args:
            model_name: Name for saved model files
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path = self.model_dir / f"{model_name}.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return str(model_path)
    
    def load(self, model_name: str = "pricing_model") -> 'PricingModel':
        """
        Load model from disk.
        
        Args:
            model_name: Name of saved model files
            
        Returns:
            Self (for chaining)
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            self.feature_names = self.model_metadata.get('feature_names', [])
            logger.info(f"Loaded metadata from {metadata_path}")
        
        return self
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance on new data.
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X)
        
        mape = mean_absolute_percentage_error(y, predictions) * 100
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_absolute_error(y, predictions) ** 2)
        r2 = r2_score(y, predictions)
        
        return {
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_prediction': float(np.mean(predictions)),
            'mean_actual': float(np.mean(y)),
            'median_error': float(np.median(np.abs(predictions - y)))
        }


# Singleton instance
pricing_model = PricingModel()
