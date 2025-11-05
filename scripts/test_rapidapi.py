#!/usr/bin/env python3
"""Quick script to test RapidAPI connection."""

import os
import sys
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings

def test_rapidapi():
    api_key = getattr(settings, 'rapidapi_key', None) or os.getenv('RAPIDAPI_KEY')
    
    if not api_key:
        print("❌ RAPIDAPI_KEY not found!")
        print("\n💡 To get your key:")
        print("   1. Go to https://rapidapi.com")
        print("   2. Sign up / Log in")
        print("   3. Subscribe to a real estate API (Realtor.com recommended)")
        print("   4. Copy your API key from https://rapidapi.com/developer/billing")
        print("   5. Add to .env file: RAPIDAPI_KEY=your_key_here")
        return False
    
    print(f"✅ Found RAPIDAPI_KEY: {api_key[:10]}...")
    print("\n🔍 Testing Realtor.com API...")
    
    # Test Realtor.com API
    url = "https://realtor.p.rapidapi.com/properties/v3/list"
    headers = {
        'X-RapidAPI-Key': api_key,
        'X-RapidAPI-Host': 'realtor.p.rapidapi.com'
    }
    params = {
        'postal_code': '27410',
        'limit': 5,
        'offset': 0,
        'status': 'for_sale'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('data', {}).get('home_search', {}).get('results', []))
            print(f"✅ Success! Found {count} listings")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed - check your API key")
            return False
        elif response.status_code == 403:
            print("❌ Access denied - make sure you're subscribed to the API")
            return False
        else:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_rapidapi()
    sys.exit(0 if success else 1)

