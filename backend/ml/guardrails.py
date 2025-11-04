"""
Guardrails Module
Flags low-confidence predictions and out-of-distribution data.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Guardrails:
    """
    Guardrails system for model predictions.
    
    Flags:
    - Low-confidence predictions
    - Out-of-distribution (OOD) data
    - Unusual configurations
    - Insufficient training data scenarios
    """
    
    def __init__(
        self,
        min_training_samples: int = 50,
        confidence_threshold: float = 0.65,
        ood_threshold: float = 3.0  # Standard deviations from training mean
    ):
        """
        Initialize guardrails.
        
        Args:
            min_training_samples: Minimum samples required for reliable predictions
            confidence_threshold: Minimum confidence score (0-1)
            ood_threshold: Z-score threshold for OOD detection
        """
        self.min_training_samples = min_training_samples
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        
        # Training data statistics (set after training)
        self.training_stats = None
        
    def set_training_stats(self, X_train: pd.DataFrame):
        """
        Set training data statistics for OOD detection.
        
        Args:
            X_train: Training feature DataFrame
        """
        self.training_stats = {
            'mean': X_train.mean().to_dict(),
            'std': X_train.std().to_dict(),
            'min': X_train.min().to_dict(),
            'max': X_train.max().to_dict(),
            'n_samples': len(X_train)
        }
        logger.info(f"Training stats set: {len(X_train)} samples, {len(X_train.columns)} features")
    
    def check_prediction_confidence(
        self,
        prediction: Dict[str, Any],
        n_training_samples: int
    ) -> Dict[str, Any]:
        """
        Check if prediction has sufficient confidence.
        
        Args:
            prediction: Dictionary with prediction results (price, probability, etc.)
            n_training_samples: Number of samples used for training
            
        Returns:
            Dictionary with confidence flags and warnings
        """
        flags = []
        warnings = []
        confidence_level = 'HIGH'
        
        # Check training sample size
        if n_training_samples < self.min_training_samples:
            flags.append('LOW_TRAINING_SAMPLES')
            warnings.append(f"Insufficient training data: {n_training_samples} samples (need {self.min_training_samples})")
            confidence_level = 'LOW'
        
        # Check sell probability confidence
        sell_prob = prediction.get('demand', {}).get('sell_probability', 0.5)
        if sell_prob < 0.3 or sell_prob > 0.9:
            flags.append('EXTREME_PROBABILITY')
            warnings.append(f"Extreme sell probability: {sell_prob:.2f}")
            if sell_prob < 0.3:
                confidence_level = 'MEDIUM'
        
        # Check predicted price reasonableness (if bounds available)
        predicted_price = prediction.get('predicted_price', 0)
        if predicted_price <= 0:
            flags.append('INVALID_PRICE')
            warnings.append(f"Invalid predicted price: ${predicted_price:,.0f}")
            confidence_level = 'LOW'
        
        # Check DOM reasonableness
        expected_dom = prediction.get('demand', {}).get('expected_dom', 0)
        if expected_dom < 0 or expected_dom > 365:
            flags.append('UNREALISTIC_DOM')
            warnings.append(f"Unrealistic DOM: {expected_dom:.0f} days")
            confidence_level = 'MEDIUM'
        
        # Check margin reasonableness
        margin_pct = prediction.get('margin', {}).get('gross_margin_pct', 0)
        if margin_pct < 0:
            flags.append('NEGATIVE_MARGIN')
            warnings.append(f"Negative margin: {margin_pct:.1f}%")
            confidence_level = 'LOW'
        elif margin_pct > 50:
            flags.append('UNUSUALLY_HIGH_MARGIN')
            warnings.append(f"Unusually high margin: {margin_pct:.1f}%")
            confidence_level = 'MEDIUM'
        
        return {
            'confidence_level': confidence_level,
            'flags': flags,
            'warnings': warnings,
            'is_reliable': confidence_level == 'HIGH' and len(flags) == 0,
            'recommendation': self._get_recommendation(flags, confidence_level)
        }
    
    def check_out_of_distribution(
        self,
        feature_vector: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check if input features are out-of-distribution.
        
        Args:
            feature_vector: Single-row DataFrame with features
            
        Returns:
            Dictionary with OOD flags
        """
        if self.training_stats is None:
            return {
                'is_ood': False,
                'flags': [],
                'warnings': ['Training stats not set - cannot check OOD'],
                'note': 'Call set_training_stats() first'
            }
        
        flags = []
        warnings = []
        ood_features = []
        
        # Check each feature
        for feature_name in feature_vector.columns:
            if feature_name not in self.training_stats['mean']:
                continue
            
            value = feature_vector[feature_name].iloc[0]
            mean = self.training_stats['mean'][feature_name]
            std = self.training_stats['std'][feature_name]
            
            # Skip if std is 0 (constant feature) or value is NaN
            if std == 0 or pd.isna(value) or pd.isna(mean):
                continue
            
            # Calculate z-score
            z_score = abs((value - mean) / std)
            
            if z_score > self.ood_threshold:
                flags.append(f'OOD_{feature_name}')
                ood_features.append({
                    'feature': feature_name,
                    'value': float(value),
                    'training_mean': float(mean),
                    'z_score': float(z_score)
                })
                warnings.append(
                    f"Feature '{feature_name}' is OOD: value={value:.2f}, "
                    f"training_mean={mean:.2f}, z_score={z_score:.2f}"
                )
        
        return {
            'is_ood': len(flags) > 0,
            'flags': flags,
            'warnings': warnings,
            'ood_features': ood_features,
            'n_ood_features': len(ood_features)
        }
    
    def check_configuration_validity(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if configuration is valid and realistic.
        
        Args:
            config: Configuration dictionary (beds, baths, sqft, etc.)
            
        Returns:
            Dictionary with validity flags
        """
        flags = []
        warnings = []
        
        beds = config.get('beds', config.get('bedrooms', 0))
        baths = config.get('baths', config.get('bathrooms', 0))
        sqft = config.get('sqft', config.get('square_feet', 0))
        
        # Check beds/baths/sqft validity
        if beds <= 0 or beds > 10:
            flags.append('INVALID_BEDS')
            warnings.append(f"Invalid bedrooms: {beds}")
        
        if baths <= 0 or baths > 10:
            flags.append('INVALID_BATHS')
            warnings.append(f"Invalid bathrooms: {baths}")
        
        if sqft <= 0 or sqft > 10000:
            flags.append('INVALID_SQFT')
            warnings.append(f"Invalid square footage: {sqft}")
        
        # Check realistic ratios
        if beds > 0:
            sqft_per_bed = sqft / beds
            if sqft_per_bed < 200:
                flags.append('TOO_SMALL_PER_BED')
                warnings.append(f"Very small sqft per bed: {sqft_per_bed:.0f}")
            elif sqft_per_bed > 2000:
                flags.append('TOO_LARGE_PER_BED')
                warnings.append(f"Very large sqft per bed: {sqft_per_bed:.0f}")
        
        if beds > 0:
            baths_per_bed = baths / beds
            if baths_per_bed > 2:
                flags.append('TOO_MANY_BATHS')
                warnings.append(f"Unusual bath-to-bed ratio: {baths_per_bed:.2f}")
        
        return {
            'is_valid': len(flags) == 0,
            'flags': flags,
            'warnings': warnings
        }
    
    def assess_recommendation_confidence(
        self,
        recommendation: Dict[str, Any],
        n_training_samples: int,
        feature_vector: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive confidence assessment for a recommendation.
        
        Args:
            recommendation: Recommendation dictionary from recommendation engine
            n_training_samples: Number of training samples
            feature_vector: Optional feature vector for OOD check
            
        Returns:
            Dictionary with comprehensive confidence assessment
        """
        # Check prediction confidence
        pred_confidence = self.check_prediction_confidence(recommendation, n_training_samples)
        
        # Check configuration validity
        config = recommendation.get('configuration', {})
        config_validity = self.check_configuration_validity(config)
        
        # Check OOD if feature vector provided
        ood_check = {}
        if feature_vector is not None:
            ood_check = self.check_out_of_distribution(feature_vector)
        
        # Combine all flags and warnings
        all_flags = pred_confidence['flags'] + config_validity['flags'] + ood_check.get('flags', [])
        all_warnings = pred_confidence['warnings'] + config_validity['warnings'] + ood_check.get('warnings', [])
        
        # Overall confidence level
        if pred_confidence['confidence_level'] == 'LOW' or not config_validity['is_valid'] or ood_check.get('is_ood', False):
            overall_confidence = 'LOW'
        elif pred_confidence['confidence_level'] == 'MEDIUM' or len(all_flags) > 0:
            overall_confidence = 'MEDIUM'
        else:
            overall_confidence = 'HIGH'
        
        return {
            'overall_confidence': overall_confidence,
            'is_reliable': overall_confidence == 'HIGH' and len(all_flags) == 0,
            'prediction_confidence': pred_confidence,
            'configuration_validity': config_validity,
            'ood_check': ood_check,
            'all_flags': all_flags,
            'all_warnings': all_warnings,
            'recommendation': self._get_recommendation(all_flags, overall_confidence)
        }
    
    def _get_recommendation(
        self,
        flags: List[str],
        confidence_level: str
    ) -> str:
        """Get human-readable recommendation based on flags."""
        if confidence_level == 'LOW':
            return "⚠️ LOW CONFIDENCE: Manual review recommended. Consider using historical heuristics or collecting more training data."
        elif confidence_level == 'MEDIUM' or len(flags) > 0:
            return "⚡ MEDIUM CONFIDENCE: Use with caution. Verify with market expertise."
        else:
            return "✅ HIGH CONFIDENCE: Prediction is reliable."
    
    def should_fallback_to_heuristics(
        self,
        confidence_assessment: Dict[str, Any],
        min_samples_threshold: int = 30
    ) -> bool:
        """
        Determine if system should fall back to heuristics instead of ML predictions.
        
        Args:
            confidence_assessment: Result from assess_recommendation_confidence
            min_samples_threshold: Minimum samples for ML (below this, use heuristics)
            
        Returns:
            True if should fallback
        """
        flags = confidence_assessment.get('all_flags', [])
        
        # Fallback if very low training samples
        if 'LOW_TRAINING_SAMPLES' in flags:
            return True
        
        # Fallback if multiple major flags
        critical_flags = ['INVALID_PRICE', 'NEGATIVE_MARGIN', 'INVALID_SQFT']
        if sum(1 for flag in flags if any(cf in flag for cf in critical_flags)) >= 2:
            return True
        
        # Fallback if OOD with many features
        ood_check = confidence_assessment.get('ood_check', {})
        if ood_check.get('n_ood_features', 0) > 5:
            return True
        
        return False


# Singleton instance
guardrails = Guardrails()
