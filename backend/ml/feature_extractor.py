"""
LLM Feature Extractor
Uses OpenAI GPT-4 to extract structured features from property descriptions.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from property descriptions using LLM."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            openai_api_key: OpenAI API key (optional, will check settings)
        """
        self.openai_api_key = openai_api_key or self._get_openai_key()
        self._client = None
    
    def _get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key from settings."""
        try:
            from config.settings import settings
            if settings.openai_api_key:
                return settings.openai_api_key
        except:
            pass
        import os
        return os.getenv('OPENAI_API_KEY', None)
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key not available")
                self._client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client
    
    def extract_features_from_description(
        self,
        description: str
    ) -> Dict[str, Any]:
        """
        Extract structured features from property description using GPT-4.
        
        Args:
            description: Property description text
            
        Returns:
            Dictionary with interior, exterior, upgrades, and condition
        """
        if not description:
            return {
                'interior': [],
                'exterior': [],
                'upgrades': [],
                'condition': ''
            }
        
        client = self._get_client()
        
        prompt = f"""Analyze this property description and extract structured features.

Property Description:
{description}

Extract and categorize the following:
1. Interior features: flooring (hardwood, tile, carpet), countertops (granite, quartz, marble), appliances (stainless steel, smart appliances), lighting, finishes, etc.
2. Exterior features: deck, patio, pool, fencing, landscaping, siding, roof type, outdoor spaces, etc.
3. Recent upgrades/renovations: renovated kitchen, updated bathrooms, new roof, HVAC, windows, electrical, plumbing, etc.
4. Overall condition: move-in ready, needs work, turnkey, updated, well-maintained, etc.

Return as JSON with this exact structure:
{{
    "interior": ["feature1", "feature2", ...],
    "exterior": ["feature1", "feature2", ...],
    "upgrades": ["upgrade1", "upgrade2", ...],
    "condition": "brief condition description"
}}

Only include features explicitly mentioned or strongly implied. Be specific and consistent.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a real estate feature extraction expert. Extract only factual features from property descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent extraction
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            content = response.choices[0].message.content
            features = json.loads(content)
            
            # Ensure structure
            result = {
                'interior': features.get('interior', []),
                'exterior': features.get('exterior', []),
                'upgrades': features.get('upgrades', []),
                'condition': features.get('condition', '')
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response content: {content}")
            return {
                'interior': [],
                'exterior': [],
                'upgrades': [],
                'condition': ''
            }
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {
                'interior': [],
                'exterior': [],
                'upgrades': [],
                'condition': ''
            }
    
    def batch_extract_features(
        self,
        listings: List[Dict[str, Any]],
        batch_size: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Process multiple listings efficiently.
        
        Args:
            listings: List of listing dictionaries with 'description' field
            
        Returns:
            List of listings with added 'extracted_features' field
        """
        enriched_listings = []
        
        for i, listing in enumerate(listings):
            description = listing.get('description', '')
            
            if description:
                try:
                    features = self.extract_features_from_description(description)
                    listing['extracted_features'] = features
                    logger.info(f"Extracted features for listing {i+1}/{len(listings)}")
                except Exception as e:
                    logger.warning(f"Error extracting features for listing {i+1}: {e}")
                    listing['extracted_features'] = {
                        'interior': [],
                        'exterior': [],
                        'upgrades': [],
                        'condition': ''
                    }
            else:
                listing['extracted_features'] = {
                    'interior': [],
                    'exterior': [],
                    'upgrades': [],
                    'condition': ''
                }
            
            enriched_listings.append(listing)
        
        return enriched_listings
    
    def get_all_features(
        self,
        listings: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Get all unique features across all listings.
        
        Args:
            listings: List of listings with extracted_features
            
        Returns:
            Dictionary with 'interior', 'exterior', 'upgrades' lists of unique features
        """
        all_interior = set()
        all_exterior = set()
        all_upgrades = set()
        
        for listing in listings:
            features = listing.get('extracted_features', {})
            all_interior.update(features.get('interior', []))
            all_exterior.update(features.get('exterior', []))
            all_upgrades.update(features.get('upgrades', []))
        
        return {
            'interior': sorted(list(all_interior)),
            'exterior': sorted(list(all_exterior)),
            'upgrades': sorted(list(all_upgrades))
        }


# Singleton instance
feature_extractor = FeatureExtractor()
