"""
Identify ZIP codes similar to 27410 for training data collection.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Similar ZIPs to 27410 (mid-to-upper market areas in Greensboro)
SIMILAR_ZIPS = ['27408', '27410', '27411', '27412']

# All Greensboro ZIPs (fallback)
ALL_GREENSBORO_ZIPS = [
    '27401', '27402', '27403', '27404', '27405', '27406', '27407', 
    '27408', '27409', '27410', '27411', '27412', '27429', '27435', 
    '27438', '27455', '27495', '27497', '27498', '27499'
]

def get_training_zips(use_similar_only: bool = True):
    """
    Get ZIP codes for training.
    
    Args:
        use_similar_only: If True, return similar ZIPs. If False, return all.
        
    Returns:
        List of ZIP codes
    """
    if use_similar_only:
        return SIMILAR_ZIPS
    else:
        return ALL_GREENSBORO_ZIPS

if __name__ == '__main__':
    print("Similar ZIPs to 27410:", ', '.join(SIMILAR_ZIPS))
    print("All Greensboro ZIPs:", ', '.join(ALL_GREENSBORO_ZIPS))
