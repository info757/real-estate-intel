"""
Sold Listings Analyzer
Analyzes recently sold listings to extract timeline data and calculate DOM metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Iterable
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoldListingsAnalyzer:
    """Analyzes sold listings to extract timeline and DOM metrics."""

    LISTED_STATUSES = {"listed", "active", "coming soon", "comingsoon"}
    PENDING_STATUSES = {"pending", "contingent", "under contract", "undercontract"}
    SOLD_STATUSES = {"sold", "closed", "offmarket", "off market"}

    def __init__(self):
        """Initialize the analyzer."""
        pass

    @staticmethod
    def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def extract_timeline(self, listing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse MLS history / price history to find listing→pending→sold dates.

        Args:
            listing: Listing dictionary with priceHistory / property_detail_raw metadata

        Returns:
            Dictionary with listing_date, pending_date, sold_date, and DOM metrics
        """
        timeline = {
            'listing_date': None,
            'pending_date': None,
            'sold_date': None,
            'dom_to_pending': None,
            'dom_to_sold': None,
            'pending_to_sold': None,
        }

        detail = listing.get('property_detail_raw') or {}

        timeline['listing_date'] = self._parse_date(
            listing.get('listing_date')
            or listing.get('datePosted')
            or (detail or {}).get('mlsListingDate')
            or (detail or {}).get('listDate')
        )
        timeline['pending_date'] = self._parse_date(
            listing.get('pending_date')
            or listing.get('pendingDate')
            or (detail or {}).get('mlsPendingDate')
            or (detail or {}).get('pendingDate')
        )
        timeline['sold_date'] = self._parse_date(
            listing.get('sold_date')
            or listing.get('dateSold')
            or (detail or {}).get('mlsLastSaleDate')
            or (detail or {}).get('lastSaleDate')
        )

        # Inspect MLS history for status transitions and DOM info
        mls_history = detail.get('mlsHistory')
        if isinstance(mls_history, list):
            listing_event = self._find_status_event(mls_history, self.LISTED_STATUSES)
            pending_event = self._find_status_event(mls_history, self.PENDING_STATUSES)
            sold_event = self._find_status_event(mls_history, self.SOLD_STATUSES)

            if listing_event and not timeline['listing_date']:
                timeline['listing_date'] = self._parse_date(
                    listing_event.get('statusDate') or listing_event.get('status_date')
                )
            if pending_event and not timeline['pending_date']:
                timeline['pending_date'] = self._parse_date(
                    pending_event.get('statusDate') or pending_event.get('status_date')
                )
            if sold_event and not timeline['sold_date']:
                timeline['sold_date'] = self._parse_date(
                    sold_event.get('statusDate') or sold_event.get('status_date')
                )

            pending_dom = self._parse_dom(pending_event)
            if pending_dom is not None:
                timeline['dom_to_pending'] = pending_dom

            sold_dom = self._parse_dom(sold_event)
            if sold_dom is not None:
                timeline['dom_to_sold'] = sold_dom

        # Fallback to priceHistory for pending/sold timestamps or DOM
        price_history = listing.get('priceHistory', [])
        if isinstance(price_history, list):
            for event in price_history:
                if not isinstance(event, dict):
                    continue
                event_label = (event.get('event') or '').lower()
                event_status = (event.get('status') or '').lower()
                event_date = self._parse_date(event.get('date') or event.get('statusDate'))

                if (
                    timeline['pending_date'] is None
                    and event_date
                    and (
                        'pending' in event_label
                        or event_status in self.PENDING_STATUSES
                    )
                ):
                    timeline['pending_date'] = event_date
                    dom_value = self._parse_dom(event)
                    if dom_value is not None:
                        timeline['dom_to_pending'] = dom_value

                if (
                    timeline['sold_date'] is None
                    and event_date
                    and (
                        'sold' in event_label
                        or 'closed' in event_label
                        or event_status in self.SOLD_STATUSES
                    )
                ):
                    timeline['sold_date'] = event_date
                    dom_value = self._parse_dom(event)
                    if dom_value is not None:
                        timeline['dom_to_sold'] = dom_value

        timeline['listing_date'] = self._normalize_datetime(timeline['listing_date'])
        timeline['pending_date'] = self._normalize_datetime(timeline['pending_date'])
        timeline['sold_date'] = self._normalize_datetime(timeline['sold_date'])

        # Calculate DOM via dates when no explicit DOM provided
        if (
            timeline['listing_date']
            and timeline['pending_date']
            and timeline['dom_to_pending'] is None
        ):
            dom_to_pending = (timeline['pending_date'] - timeline['listing_date']).days
            if dom_to_pending >= 0:
                timeline['dom_to_pending'] = dom_to_pending

        if (
            timeline['listing_date']
            and timeline['sold_date']
            and timeline['dom_to_sold'] is None
        ):
            dom_to_sold = (timeline['sold_date'] - timeline['listing_date']).days
            if dom_to_sold >= 0:
                timeline['dom_to_sold'] = dom_to_sold

        if (
            timeline['pending_date']
            and timeline['sold_date']
            and timeline['pending_to_sold'] is None
        ):
            pending_to_sold = (timeline['sold_date'] - timeline['pending_date']).days
            if pending_to_sold >= 0:
                timeline['pending_to_sold'] = pending_to_sold

        if timeline['dom_to_pending'] is None and timeline['dom_to_sold'] is not None:
            timeline['dom_to_pending'] = timeline['dom_to_sold']

        if (
            timeline['pending_date'] is None
            and timeline['sold_date']
            and timeline['listing_date']
            and timeline['dom_to_pending'] is not None
        ):
            timeline['pending_date'] = timeline['listing_date'] + timedelta(
                days=timeline['dom_to_pending']
            )

        return timeline

    def _find_status_event(
        self,
        history: Iterable[Dict[str, Any]],
        statuses: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        statuses_lower = {status.lower() for status in statuses}
        for event in history:
            if not isinstance(event, dict):
                continue
            status_value = str(event.get('status') or '').strip().lower()
            if status_value in statuses_lower:
                return event
        return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None

        text = str(date_str).strip()
        if not text:
            return None

        candidates = [text]
        if text.endswith('Z'):
            candidates.append(text[:-1] + '+00:00')
        if text.endswith(' UTC'):
            candidates.append(text[:-4] + '+00:00')

        for candidate in candidates:
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                continue

        try:
            parsed = datetime.strptime(text.split('T')[0], '%Y-%m-%d')
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

        for fmt in ('%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y'):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        logger.debug("Could not parse date: %s", date_str)
        return None

    def _parse_dom(self, event: Optional[Dict[str, Any]]) -> Optional[int]:
        if not isinstance(event, dict):
            return None
        dom_value = (
            event.get('dom')
            or event.get('daysOnMarket')
            or event.get('days_on_market')
        )
        if dom_value is None:
            return None
        try:
            return int(float(dom_value))
        except (ValueError, TypeError):
            return None

    def calculate_dom_metrics(
        self,
        listings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate DOM metrics for a list of listings.

        Args:
            listings: List of listing dictionaries

        Returns:
            List of listings with added DOM metrics
        """
        enriched_listings = []

        for listing in listings:
            timeline = self.extract_timeline(listing)

            enriched_listing = {**listing, **timeline}
            enriched_listings.append(enriched_listing)

        return enriched_listings

    def filter_fast_sellers(
        self,
        listings: List[Dict[str, Any]],
        dom_threshold: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Identify properties with DOM to pending < threshold.

        Args:
            listings: List of listings with DOM metrics
            dom_threshold: Maximum DOM to pending for "fast seller"

        Returns:
            List of fast-selling properties
        """
        fast_sellers = []

        for listing in listings:
            dom_to_pending = listing.get('dom_to_pending')

            if dom_to_pending is not None and dom_to_pending <= dom_threshold:
                fast_sellers.append(listing)

        return fast_sellers

    def analyze_fast_seller_characteristics(
        self,
        listings: List[Dict[str, Any]],
        dom_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Statistical analysis of fast sellers vs slow sellers.

        Args:
            listings: List of listings with DOM metrics
            dom_threshold: DOM threshold for fast vs slow

        Returns:
            Dictionary with comparison statistics
        """
        fast_sellers = self.filter_fast_sellers(listings, dom_threshold)
        slow_sellers = [
            l for l in listings
            if l.get('dom_to_pending') is not None and l.get('dom_to_pending', 999) > dom_threshold
        ]

        analysis = {
            'total_listings': len(listings),
            'fast_sellers': len(fast_sellers),
            'slow_sellers': len(slow_sellers),
            'fast_seller_pct': (len(fast_sellers) / len(listings) * 100) if listings else 0,
        }

        if fast_sellers:
            analysis['fast_seller_avg_price'] = self._avg([l.get('price') for l in fast_sellers if l.get('price')])
            analysis['fast_seller_avg_beds'] = self._avg([l.get('beds') for l in fast_sellers if l.get('beds')])
            analysis['fast_seller_avg_baths'] = self._avg([l.get('baths') for l in fast_sellers if l.get('baths')])
            analysis['fast_seller_avg_sqft'] = self._avg([l.get('sqft') for l in fast_sellers if l.get('sqft')])
            analysis['fast_seller_avg_dom_to_pending'] = self._avg([l.get('dom_to_pending') for l in fast_sellers if l.get('dom_to_pending') is not None])

        if slow_sellers:
            analysis['slow_seller_avg_price'] = self._avg([l.get('price') for l in slow_sellers if l.get('price')])
            analysis['slow_seller_avg_beds'] = self._avg([l.get('beds') for l in slow_sellers if l.get('beds')])
            analysis['slow_seller_avg_baths'] = self._avg([l.get('baths') for l in slow_sellers if l.get('baths')])
            analysis['slow_seller_avg_sqft'] = self._avg([l.get('sqft') for l in slow_sellers if l.get('sqft')])
            analysis['slow_seller_avg_dom_to_pending'] = self._avg([l.get('dom_to_pending') for l in slow_sellers if l.get('dom_to_pending') is not None])

        return analysis

    def _avg(self, values: List[float]) -> float:
        """Calculate average of numeric values."""
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return 0.0
        return sum(valid_values) / len(valid_values)


# Singleton instance
sold_listings_analyzer = SoldListingsAnalyzer()
