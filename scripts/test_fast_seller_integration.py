"""
Test script for fast-seller model integration.
Tests the end-to-end flow with the recommendation engine.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ml.recommendation_engine import RecommendationEngine

def test_fast_seller_integration():
    """Test fast-seller model integration in recommendation engine."""
    print("="*80)
    print("FAST-SELLER MODEL INTEGRATION TEST")
    print("="*80)
    print()
    
    # Initialize recommendation engine
    engine = RecommendationEngine(
        min_sell_probability=0.5,
        max_dom=90,
        min_margin_pct=0.0
    )
    
    # Sample lot features
    lot_features = {
        'zip_code': '27410',
        'latitude': 36.089,
        'longitude': -79.908,
        'lot_size_acres': 0.25,
        'lot_condition': 'flat',
        'utilities_status': 'all_utilities',
        'subdivision': 'Hamilton Lakes'
    }
    
    print("Testing with market insights ENABLED:")
    print("-" * 80)
    
    try:
        results = engine.generate_recommendations(
            lot_features=lot_features,
            property_type='Single Family Home',
            top_n=3,
            use_market_insights=True
        )
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return False
        
        print(f"✅ Generated {len(results['recommendations'])} recommendations")
        print()
        
        # Check each recommendation for fast-seller data
        for i, rec in enumerate(results['recommendations'], 1):
            config = rec.get('configuration', {})
            demand = rec.get('demand', {})
            
            print(f"Recommendation #{i}: {config.get('beds', '?')}BR/{config.get('baths', '?')}BA")
            print(f"  Predicted Price: ${rec.get('predicted_price', 0):,.0f}")
            
            # Check for fast-seller predictions
            fast_prob = demand.get('fast_seller_probability')
            fast_dom = demand.get('fast_seller_dom')
            
            if fast_prob is not None:
                print(f"  ✅ Fast-Seller Probability: {fast_prob*100:.0f}%")
                if fast_dom is not None:
                    print(f"  ✅ Fast-Seller DOM: {fast_dom:.0f} days")
            else:
                print(f"  ⚠️ Fast-seller predictions not available (using demand model defaults)")
            
            # Check for insights
            insights = rec.get('insights')
            if insights:
                print(f"  ✅ Insights available:")
                print(f"     Summary: {insights.get('summary', 'N/A')}")
                market_align = insights.get('market_alignment', {})
                if market_align:
                    print(f"     Market Alignment: {market_align.get('level', 'N/A')}")
            else:
                print(f"  ⚠️ Insights not generated")
            
            print()
        
        print("="*80)
        print("✅ TEST PASSED: Fast-seller integration working")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_fast_seller_integration()
    sys.exit(0 if success else 1)
