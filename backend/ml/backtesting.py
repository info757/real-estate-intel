"""
Backtesting Module
Evaluates model performance on historical data by simulating predictions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from backend.ml.pricing_model import pricing_model
from backend.ml.demand_model import demand_model
from backend.ml.feature_engineering import feature_engineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for evaluating model performance on historical data.
    """
    
    def __init__(
        self,
        pricing_model=pricing_model,
        demand_model=demand_model
    ):
        """
        Initialize backtester.
        
        Args:
            pricing_model: Trained pricing model instance
            demand_model: Trained demand model instance
        """
        self.pricing_model = pricing_model
        self.demand_model = demand_model
        
    def backtest_pricing(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        split_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backtest pricing model on historical data.
        
        Args:
            X: Feature DataFrame
            y_true: True sale prices
            split_date: Optional date string to split train/test (YYYY-MM-DD)
                       If None, uses all data for testing
            
        Returns:
            Dictionary with backtesting metrics
        """
        logger.info(f"Backtesting pricing model on {len(X)} samples")
        
        # Predict prices
        try:
            y_pred = self.pricing_model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting prices: {e}")
            return {'error': str(e)}
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_absolute_error(y_true, y_pred) ** 2)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage of predictions within thresholds
        pct_error = np.abs((y_pred - y_true) / y_true * 100)
        within_5pct = (pct_error <= 5).mean() * 100
        within_10pct = (pct_error <= 10).mean() * 100
        within_20pct = (pct_error <= 20).mean() * 100
        
        # Calculate bias (systematic over/under-prediction)
        bias_pct = ((y_pred - y_true) / y_true * 100).mean()
        
        # Error distribution
        error_distribution = {
            'mean_error': float(np.mean(y_pred - y_true)),
            'median_error': float(np.median(y_pred - y_true)),
            'std_error': float(np.std(y_pred - y_true)),
            'min_error': float(np.min(y_pred - y_true)),
            'max_error': float(np.max(y_pred - y_true)),
            'pct_overpredicting': float((y_pred > y_true).mean() * 100),
            'pct_underpredicting': float((y_pred < y_true).mean() * 100)
        }
        
        return {
            'metrics': {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'bias_pct': float(bias_pct),
            },
            'accuracy_thresholds': {
                'within_5pct': float(within_5pct),
                'within_10pct': float(within_10pct),
                'within_20pct': float(within_20pct),
            },
            'error_distribution': error_distribution,
            'n_samples': len(X),
            'mean_true_price': float(y_true.mean()),
            'mean_predicted_price': float(y_pred.mean()),
            'backtest_date': datetime.now().isoformat()
        }
    
    def backtest_demand(
        self,
        X: pd.DataFrame,
        y_sold_fast_true: pd.Series,
        y_dom_true: pd.Series,
        sell_within_days: int = 90
    ) -> Dict[str, Any]:
        """
        Backtest demand models on historical data.
        
        Args:
            X: Feature DataFrame
            y_sold_fast_true: True binary labels (1 = sold fast, 0 = otherwise)
            y_dom_true: True days on market
            sell_within_days: Threshold for fast sale (must match training)
            
        Returns:
            Dictionary with backtesting metrics for both models
        """
        logger.info(f"Backtesting demand models on {len(X)} samples")
        
        try:
            # Predict demand
            predictions = self.demand_model.predict(X)
            y_prob_pred = predictions['sell_probability']
            y_dom_pred = predictions['expected_dom']
            
            # Binary predictions for classification
            y_sold_fast_pred = (y_prob_pred >= 0.5).astype(int)
            
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            return {'error': str(e)}
        
        # Classification metrics (sell probability)
        auc = roc_auc_score(y_sold_fast_true, y_prob_pred)
        accuracy = accuracy_score(y_sold_fast_true, y_sold_fast_pred)
        precision = precision_score(y_sold_fast_true, y_sold_fast_pred, zero_division=0)
        recall = recall_score(y_sold_fast_true, y_sold_fast_pred, zero_division=0)
        f1 = f1_score(y_sold_fast_true, y_sold_fast_pred, zero_division=0)
        
        # Regression metrics (DOM)
        def safe_mape(y_true, y_pred):
            mask = y_true > 0
            if mask.sum() == 0:
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        dom_mape = safe_mape(y_dom_true, y_dom_pred)
        dom_mae = mean_absolute_error(y_dom_true, y_dom_pred)
        dom_rmse = np.sqrt(mean_absolute_error(y_dom_true, y_dom_pred) ** 2)
        dom_r2 = r2_score(y_dom_true, y_dom_pred)
        
        # Calculate percentage of predictions within thresholds
        dom_pct_error = np.abs((y_dom_pred - y_dom_true) / np.maximum(y_dom_true, 1) * 100)
        dom_within_10pct = (dom_pct_error <= 10).mean() * 100
        dom_within_20pct = (dom_pct_error <= 20).mean() * 100
        dom_within_30pct = (dom_pct_error <= 30).mean() * 100
        
        # Bias analysis
        prob_bias = (y_prob_pred.mean() - y_sold_fast_true.mean()) * 100
        dom_bias = (y_dom_pred - y_dom_true).mean()
        
        return {
            'sell_probability': {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'bias_pct': float(prob_bias),
                'mean_true_prob': float(y_sold_fast_true.mean()),
                'mean_predicted_prob': float(y_prob_pred.mean())
            },
            'dom': {
                'mape': float(dom_mape),
                'mae': float(dom_mae),
                'rmse': float(dom_rmse),
                'r2': float(dom_r2),
                'bias_days': float(dom_bias),
                'accuracy_thresholds': {
                    'within_10pct': float(dom_within_10pct),
                    'within_20pct': float(dom_within_20pct),
                    'within_30pct': float(dom_within_30pct),
                },
                'mean_true_dom': float(y_dom_true.mean()),
                'mean_predicted_dom': float(y_dom_pred.mean())
            },
            'n_samples': len(X),
            'backtest_date': datetime.now().isoformat()
        }
    
    def backtest_recommendations(
        self,
        historical_configs: List[Dict[str, Any]],
        lot_features: Dict[str, Any],
        actual_outcomes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backtest recommendation engine by evaluating what it would have recommended
        versus what actually happened.
        
        Args:
            historical_configs: List of configurations that were actually built
            lot_features: Lot features for the backtest
            actual_outcomes: Dictionary with actual outcomes:
                - sale_price: float
                - days_on_market: int
                - sold_fast: bool
                - actual_cost: float (optional)
        
        Returns:
            Dictionary with backtesting results
        """
        logger.info(f"Backtesting recommendations on {len(historical_configs)} configurations")
        
        # This would require the recommendation engine
        # For now, return placeholder structure
        return {
            'n_configurations': len(historical_configs),
            'note': 'Full recommendation backtesting requires recommendation engine integration',
            'backtest_date': datetime.now().isoformat()
        }
    
    def generate_backtest_report(
        self,
        pricing_results: Dict[str, Any],
        demand_results: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable backtesting report.
        
        Args:
            pricing_results: Results from backtest_pricing
            demand_results: Results from backtest_demand
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MODEL BACKTESTING REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Pricing results
        if 'error' not in pricing_results:
            report_lines.append("PRICING MODEL PERFORMANCE")
            report_lines.append("-"*80)
            metrics = pricing_results['metrics']
            report_lines.append(f"MAPE: {metrics['mape']:.2f}%")
            report_lines.append(f"MAE: ${metrics['mae']:,.0f}")
            report_lines.append(f"R²: {metrics['r2']:.3f}")
            report_lines.append(f"Bias: {metrics['bias_pct']:+.2f}%")
            
            thresholds = pricing_results['accuracy_thresholds']
            report_lines.append("")
            report_lines.append("Accuracy Thresholds:")
            report_lines.append(f"  Within 5%: {thresholds['within_5pct']:.1f}%")
            report_lines.append(f"  Within 10%: {thresholds['within_10pct']:.1f}%")
            report_lines.append(f"  Within 20%: {thresholds['within_20pct']:.1f}%")
            
            error_dist = pricing_results['error_distribution']
            report_lines.append("")
            report_lines.append("Error Distribution:")
            report_lines.append(f"  Mean Error: ${error_dist['mean_error']:,.0f}")
            report_lines.append(f"  Median Error: ${error_dist['median_error']:,.0f}")
            report_lines.append(f"  Over-predicting: {error_dist['pct_overpredicting']:.1f}%")
            report_lines.append(f"  Under-predicting: {error_dist['pct_underpredicting']:.1f}%")
        else:
            report_lines.append("PRICING MODEL: ERROR")
            report_lines.append(f"  {pricing_results['error']}")
        
        report_lines.append("")
        report_lines.append("")
        
        # Demand results
        if 'error' not in demand_results:
            report_lines.append("DEMAND MODEL PERFORMANCE")
            report_lines.append("-"*80)
            
            # Sell probability
            prob_metrics = demand_results['sell_probability']
            report_lines.append("Sell Probability (Classification):")
            report_lines.append(f"  AUC: {prob_metrics['auc']:.3f}")
            report_lines.append(f"  Accuracy: {prob_metrics['accuracy']:.3f}")
            report_lines.append(f"  Precision: {prob_metrics['precision']:.3f}")
            report_lines.append(f"  Recall: {prob_metrics['recall']:.3f}")
            report_lines.append(f"  F1: {prob_metrics['f1']:.3f}")
            
            # DOM
            dom_metrics = demand_results['dom']
            report_lines.append("")
            report_lines.append("Days on Market (Regression):")
            report_lines.append(f"  MAPE: {dom_metrics['mape']:.2f}%")
            report_lines.append(f"  MAE: {dom_metrics['mae']:.1f} days")
            report_lines.append(f"  R²: {dom_metrics['r2']:.3f}")
            report_lines.append(f"  Bias: {dom_metrics['bias_days']:+.1f} days")
            
            thresholds = dom_metrics['accuracy_thresholds']
            report_lines.append("")
            report_lines.append("Accuracy Thresholds:")
            report_lines.append(f"  Within 10%: {thresholds['within_10pct']:.1f}%")
            report_lines.append(f"  Within 20%: {thresholds['within_20pct']:.1f}%")
            report_lines.append(f"  Within 30%: {thresholds['within_30pct']:.1f}%")
        else:
            report_lines.append("DEMAND MODEL: ERROR")
            report_lines.append(f"  {demand_results['error']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)


# Singleton instance
backtester = Backtester()
