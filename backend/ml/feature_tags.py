"""
Utility helpers to extract human-readable feature flags from listing payloads.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


FEATURE_LIBRARY: Dict[str, Dict[str, Any]] = {
    # Outdoor living & site appeal
    "covered_outdoor_living": {
        "label": "Covered outdoor living",
        "keywords": [
            r"covered\s+deck",
            r"covered\s+porch",
            r"covered\s+patio",
            r"screen\w*\s+porch",
            r"screen\w*\s+patio",
            r"screen\w*\s+lanai",
            r"screen\w*\s+room",
            r"outdoor\s+living\s+room",
            r"extended\s+patio",
            r"lanai",
        ],
    },
    "screened_porch": {
        "label": "Screened porch",
        "keywords": [
            r"screened[-\s]*in\s+porch",
            r"screened\s+deck",
            r"screened\s+patio",
            r"screened\s+lanai",
        ],
        "detail_contains": [
            {"path": "propertyInfo.porchType", "contains": ["screen", "enclos"]}
        ],
    },
    "deck_patio": {
        "label": "Deck or patio",
        "summary_flags": ["deck", "patio"],
        "keywords": [
            r"\bdeck\b",
            r"\bpatio\b",
            r"paver\s+patio",
            r"flagstone\s+patio",
        ],
    },
    "outdoor_kitchen": {
        "label": "Outdoor kitchen / grill",
        "keywords": [
            r"outdoor\s+kitchen",
            r"summer\s+kitchen",
            r"built[-\s]?in\s+grill",
            r"grilling\s+station",
            r"bbq\s+station",
            r"bbq\s+island",
            r"pizza\s+oven",
        ],
    },
    "outdoor_fireplace": {
        "label": "Outdoor fireplace",
        "keywords": [
            r"outdoor\s+fireplace",
            r"exterior\s+fireplace",
        ],
    },
    "fire_pit": {
        "label": "Fire pit",
        "keywords": [
            r"fire\s+pit",
            r"firepit",
        ],
    },
    "pergola": {
        "label": "Pergola or arbor",
        "keywords": [
            r"\bpergola\b",
            r"\barbor\b",
            r"gazebo",
        ],
    },
    "fenced_yard": {
        "label": "Fenced yard",
        "keywords": [
            r"fenced\s+yard",
            r"privacy\s+fence",
            r"fenced\s+back\s?yard",
            r"fully\s+fenced",
            r"fenced\s+lot",
        ],
    },
    "irrigation_system": {
        "label": "Irrigation system",
        "keywords": [
            r"irrigation\s+system",
            r"sprinkler\s+system",
            r"sprinklers?\s+installed",
            r"smart\s+irrigation",
        ],
    },
    "landscaping": {
        "label": "Professional landscaping",
        "keywords": [
            r"professional\s+landscap",
            r"landscape\s+lighting",
            r"outdoor\s+lighting",
            r"upgraded\s+landscap",
        ],
    },
    "pool": {
        "label": "Private pool",
        "summary_flags": ["pool"],
        "keywords": [
            r"in-?ground\s+pool",
            r"heated\s+pool",
            r"saltwater\s+pool",
            r"gunite\s+pool",
        ],
    },
    "hot_tub": {
        "label": "Hot tub or spa",
        "keywords": [
            r"hot\s+tub",
            r"spa\s+tub",
            r"jacuzzi",
        ],
    },
    "cul_de_sac": {
        "label": "Cul-de-sac lot",
        "keywords": [
            r"cul[-\s]?de[-\s]?sac",
        ],
    },
    "golf_course_lot": {
        "label": "Golf course lot",
        "keywords": [
            r"golf\s+course\s+lot",
            r"on\s+the\s+golf\s+course",
            r"golf\s+course\s+view",
            r"fairway\s+view",
        ],
    },
    "gated_community": {
        "label": "Gated community",
        "keywords": [
            r"gated\s+community",
            r"gated\s+entry",
            r"guard\s+gated",
        ],
    },
    "community_pool": {
        "label": "Community pool access",
        "keywords": [
            r"community\s+pool",
            r"neighborhood\s+pool",
            r"club\s+pool",
        ],
    },
    "clubhouse": {
        "label": "Clubhouse access",
        "keywords": [
            r"community\s+clubhouse",
            r"neighborhood\s+clubhouse",
            r"clubhouse\s+amenit",
        ],
    },
    "tennis_courts": {
        "label": "Tennis courts",
        "keywords": [
            r"tennis\s+courts?",
            r"community\s+tennis",
        ],
    },
    "pickleball_courts": {
        "label": "Pickleball courts",
        "keywords": [
            r"pickleball\s+courts?",
        ],
    },

    # Kitchen & entertaining
    "chef_kitchen": {
        "label": "Chef / gourmet kitchen",
        "keywords": [
            r"chef'?s?\s+kitchen",
            r"gourmet\s+kitchen",
            r"designer\s+kitchen",
            r"professional\s+range",
            r"sub-?zero",
            r"wolf\s+appliance",
            r"thermador",
            r"double\s+island",
            r"entertainer'?s?\s+kitchen",
        ],
    },
    "butlers_pantry": {
        "label": "Butler's pantry",
        "keywords": [
            r"butler'?s?\s+pantr",
        ],
    },
    "walk_in_pantry": {
        "label": "Walk-in pantry",
        "keywords": [
            r"walk[-\s]?in\s+pantr",
            r"scullery",
        ],
    },
    "double_oven": {
        "label": "Double oven",
        "keywords": [
            r"double\s+ovens?",
            r"dual\s+ovens?",
        ],
    },
    "gas_cooking": {
        "label": "Gas cooking",
        "keywords": [
            r"gas\s+cooktop",
            r"gas\s+range",
            r"six\s+burner\s+range",
        ],
    },
    "farmhouse_sink": {
        "label": "Farmhouse sink",
        "keywords": [
            r"farmhouse\s+sink",
            r"apron\s+front\s+sink",
        ],
    },
    "soft_close_cabinets": {
        "label": "Soft-close cabinetry",
        "keywords": [
            r"soft[-\s]?close\s+cabin",
            r"soft[-\s]?close\s+drawers?",
        ],
    },
    "quartz_counters": {
        "label": "Quartz countertops",
        "keywords": [
            r"quartz\s+counter(?:tops?)?",
        ],
    },
    "granite_counters": {
        "label": "Granite countertops",
        "keywords": [
            r"granite\s+counter(?:tops?)?",
        ],
    },
    "wet_bar": {
        "label": "Wet bar",
        "keywords": [
            r"wet\s+bar",
            r"bar\s+area",
            r"beverage\s+center",
        ],
    },
    "wine_cellar": {
        "label": "Wine storage",
        "keywords": [
            r"wine\s+cellar",
            r"wine\s+room",
            r"wine\s+fridge",
        ],
    },

    # Primary bath & spa
    "luxury_primary_bath": {
        "label": "Luxury primary bath",
        "keywords": [
            r"spa[-\s]?like\s+bath",
            r"owner'?s?\s+spa",
            r"luxury\s+ensuite",
            r"retreat\s+bath",
        ],
    },
    "freestanding_tub": {
        "label": "Freestanding tub",
        "keywords": [
            r"freestanding\s+tub",
            r"standalone\s+tub",
        ],
    },
    "soaking_tub": {
        "label": "Soaking tub",
        "keywords": [
            r"soaking\s+tub",
            r"garden\s+tub",
        ],
    },
    "rain_shower": {
        "label": "Rain shower",
        "keywords": [
            r"rain\s+shower",
            r"rainfall\s+shower",
        ],
    },
    "steam_shower": {
        "label": "Steam shower",
        "keywords": [
            r"steam\s+shower",
        ],
    },
    "heated_bath_floor": {
        "label": "Heated bathroom floors",
        "keywords": [
            r"heated\s+bath\s+floors?",
            r"heated\s+flooring?\s+in\s+the\s+bath",
        ],
    },

    # Interior finishes
    "fireplace": {
        "label": "Fireplace (indoor/outdoor)",
        "keywords": [
            r"\bfireplace\b",
            r"gas\s+logs?",
            r"stone\s+fireplace",
            r"double\s+fireplace",
            r"two\s+way\s+fireplace",
        ],
        "detail_flags": ["propertyInfo.fireplace"],
    },
    "hardwood_floors": {
        "label": "Hardwood floors",
        "keywords": [
            r"hardwood\s+floors?",
            r"hardwoods\s+throughout",
            r"site[-\s]?finished\s+hardwoods?",
            r"engineered\s+hardwoods?",
        ],
    },
    "lvp_flooring": {
        "label": "Luxury vinyl plank",
        "keywords": [
            r"luxury\s+vinyl\s+plank",
            r"\bLVP\b",
            r"vinyl\s+plank\s+floor",
        ],
    },
    "tile_bath": {
        "label": "Tile baths/showers",
        "keywords": [
            r"tiled?\s+shower",
            r"tile\s+bath(rooms?)?",
        ],
    },
    "built_ins": {
        "label": "Custom built-ins",
        "keywords": [
            r"custom\s+built[-\s]?ins?",
            r"built[-\s]?in\s+bookcases?",
            r"built[-\s]?in\s+storage",
        ],
    },
    "wainscoting": {
        "label": "Wainscoting / trim detail",
        "keywords": [
            r"wainscoting",
            r"judge'?s?\s+paneling",
        ],
    },
    "shiplap": {
        "label": "Shiplap accents",
        "keywords": [
            r"ship\s*lap",
        ],
    },
    "coffered_ceiling": {
        "label": "Coffered ceiling",
        "keywords": [
            r"coffered\s+ceiling",
        ],
    },
    "tray_ceiling": {
        "label": "Tray ceiling",
        "keywords": [
            r"tray\s+ceiling",
            r"trey\s+ceiling",
        ],
    },
    "vaulted_ceiling": {
        "label": "Vaulted or cathedral ceiling",
        "keywords": [
            r"vaulted\s+ceiling",
            r"cathedral\s+ceiling",
            r"two\s+story\s+ceiling",
        ],
    },
    "walk_up_attic": {
        "label": "Walk-up attic storage",
        "detail_flags": ["propertyInfo.attic"],
    },
    "barn_door": {
        "label": "Barn door detail",
        "keywords": [
            r"barn\s+door",
        ],
    },
    "keeping_room": {
        "label": "Keeping room",
        "keywords": [
            r"keeping\s+room",
        ],
    },

    # Flex rooms & lifestyle spaces
    "home_office": {
        "label": "Dedicated home office",
        "keywords": [
            r"home\s+office",
            r"private\s+study",
            r"executive\s+office",
            r"dual\s+offices?",
        ],
    },
    "bonus_flex_room": {
        "label": "Bonus / flex room",
        "keywords": [
            r"bonus\s+room",
            r"flex\s+space",
            r"loft\s+area",
            r"game\s+room",
            r"rec\s+room",
        ],
    },
    "media_room": {
        "label": "Media / theater room",
        "keywords": [
            r"media\s+room",
            r"home\s+theater",
            r"cinema\s+room",
        ],
    },
    "home_gym": {
        "label": "Home gym",
        "keywords": [
            r"home\s+gym",
            r"exercise\s+room",
            r"fitness\s+room",
        ],
    },
    "sunroom": {
        "label": "Sunroom / four seasons room",
        "keywords": [
            r"sun\s*room",
            r"four\s+season[s']?\s+room",
            r"all[-\s]season\s+room",
        ],
    },
    "mudroom": {
        "label": "Mudroom / drop zone",
        "keywords": [
            r"mud\s*room",
            r"drop\s+zone",
            r"built[-\s]?in\s+lockers",
        ],
    },

    # Technology & efficiency
    "smart_home": {
        "label": "Smart home package",
        "keywords": [
            r"smart\s+home",
            r"smart\s+thermostat",
            r"home\s+automation",
            r"control4",
            r"wired\s+network",
            r"whole\s+home\s+audio",
        ],
    },
    "ev_ready_garage": {
        "label": "EV-ready garage",
        "keywords": [
            r"ev\s+charg",
            r"electric\s+vehicle\s+charger",
            r"car\s+charger",
            r"tesla\s+charger",
            r"240v\s+outlet\s+in\s+garage",
        ],
    },
    "energy_package": {
        "label": "Energy-efficient upgrades",
        "keywords": [
            r"energy\s+star",
            r"hers\s+rating",
            r"high\s+efficiency\s+hvac",
            r"foam\s+insulated",
        ],
    },
    "tankless_water_heater": {
        "label": "Tankless water heater",
        "keywords": [
            r"tankless\s+water\s+heater",
            r"on[-\s]?demand\s+water",
        ],
    },
    "spray_foam_insulation": {
        "label": "Spray foam insulation",
        "keywords": [
            r"spray\s+foam\s+insulation",
            r"spray\s+foamed\s+attic",
        ],
    },
    "solar_ready": {
        "label": "Solar ready / equipped",
        "keywords": [
            r"solar\s+panel",
            r"pre-?wired\s+for\s+solar",
        ],
    },
    "whole_house_generator": {
        "label": "Whole-house generator",
        "keywords": [
            r"whole\s+house\s+generator",
            r"backup\s+generator",
        ],
    },
    "built_in_speakers": {
        "label": "Built-in speakers",
        "keywords": [
            r"built[-\s]?in\s+speakers",
            r"surround\s+sound",
            r"pre[-\s]?wired\s+audio",
        ],
    },
    "central_vacuum": {
        "label": "Central vacuum",
        "keywords": [
            r"central\s+vac",
            r"central\s+vacuum",
        ],
    },

    # Structure & storage
    "basement": {
        "label": "Basement",
        "summary_flags": ["basement"],
        "keywords": [
            r"basement",
        ],
    },
    "finished_basement": {
        "label": "Finished basement",
        "keywords": [
            r"finished\s+basement",
            r"daylight\s+basement",
        ],
    },
    "walkout_basement": {
        "label": "Walk-out basement",
        "keywords": [
            r"walk[-\s]?out\s+basement",
            r"walkout\s+basement",
        ],
    },
    "three_car_garage": {
        "label": "Three-car garage",
        "keywords": [
            r"three\s+car\s+garage",
            r"3\s*-?\s*car\s+garage",
            r"oversized\s+garage",
        ],
    },
    "garage_parking": {
        "label": "Garage parking",
        "summary_flags": ["garage"],
        "keywords": [
            r"\bgarage\b",
        ],
    },
    "side_entry_garage": {
        "label": "Side-entry garage",
        "keywords": [
            r"side[-\s]?entry\s+garage",
            r"side[-\s]?load\s+garage",
        ],
    },
}

_COMPILED_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    key: [re.compile(pattern, re.IGNORECASE) for pattern in meta.get("keywords", [])]
    for key, meta in FEATURE_LIBRARY.items()
}


def _truthy(value: Any) -> bool:
    """Interpret a value as truthy, accounting for string representations."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"y", "yes", "true", "1"}:
            return True
        if lowered in {"n", "no", "false", "0", ""}:
            return False
    return bool(value)


def _ensure_dict(value: Any) -> Dict[str, Any]:
    """Return a dictionary representation when possible."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Retrieve a nested value from a dictionary using dot-separated paths."""
    if not isinstance(data, dict):
        return None
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _build_text_blob(listing: Dict) -> str:
    """Collect as much free text as possible from a listing payload."""
    parts: list[str] = []

    description = listing.get("description")
    if isinstance(description, str):
        parts.append(description)

    summary = listing.get("summary") or {}
    if isinstance(summary, dict):
        for key in (
            "description",
            "remarks",
            "publicRemarks",
            "interiorFeatures",
            "exteriorFeatures",
            "cooling",
            "heating",
            "fireplace",
            "laundry",
        ):
            value = summary.get(key)
            if isinstance(value, str):
                parts.append(value)

    property_detail = _ensure_dict(listing.get("property_detail_raw"))
    for field in (
        "interiorFeatures",
        "exteriorFeatures",
        "remarks",
        "fireplace",
        "kitchen",
        "bath",
        "garage",
    ):
        value = property_detail.get(field)
        if isinstance(value, str):
            parts.append(value)

    mls_detail = listing.get("mls_detail_raw")
    if isinstance(mls_detail, str):
        parts.append(mls_detail)
    elif isinstance(mls_detail, dict):
        for val in mls_detail.values():
            if isinstance(val, str):
                parts.append(val)

    return "\n".join(parts).lower()


def tag_listing_features(listing: Dict) -> Dict[str, int]:
    """
    Return a dictionary of feature flags (1/0) for a listing.
    """
    text_blob = _build_text_blob(listing)
    summary = _ensure_dict(listing.get("summary"))
    detail = _ensure_dict(listing.get("property_detail_raw"))
    metadata = _ensure_dict(listing.get("metadata"))

    flags: Dict[str, int] = {}

    for feature_key, meta in FEATURE_LIBRARY.items():
        patterns = _COMPILED_PATTERNS.get(feature_key, [])
        matched = any(pattern.search(text_blob) for pattern in patterns)

        if not matched:
            for flag_key in meta.get("summary_flags", []):
                value = summary.get(flag_key)
                if _truthy(value):
                    matched = True
                    break

        if not matched:
            for flag_key in meta.get("detail_flags", []):
                value = _get_nested_value(detail, flag_key)
                if _truthy(value):
                    matched = True
                    break

        if not matched:
            for flag_key in meta.get("metadata_flags", []):
                value = _get_nested_value(metadata, flag_key)
                if _truthy(value):
                    matched = True
                    break

        if not matched:
            for clause in meta.get("detail_contains", []):
                path = clause.get("path")
                needles = clause.get("contains") or clause.get("any") or []
                if not path or not needles:
                    continue
                value = _get_nested_value(detail, path)
                if isinstance(value, str):
                    value_lower = value.lower()
                    if any(needle.lower() in value_lower for needle in needles):
                        matched = True
                        break
                elif isinstance(value, (list, tuple, set)):
                    joined = " ".join(str(item).lower() for item in value)
                    if any(needle.lower() in joined for needle in needles):
                        matched = True
                        break

        if not matched:
            for clause in meta.get("detail_predicates", []):
                path = clause.get("path")
                comparator = clause.get("comparator")
                threshold = clause.get("threshold")
                if not path or comparator not in {"gte", "gt", "lte", "lt"}:
                    continue
                value = _get_nested_value(detail, path)
                if value is None:
                    continue
                try:
                    numeric_val = float(value)
                except (TypeError, ValueError):
                    continue
                try:
                    threshold_val = float(threshold)
                except (TypeError, ValueError):
                    continue
                if comparator == "gte" and numeric_val >= threshold_val:
                    matched = True
                    break
                if comparator == "gt" and numeric_val > threshold_val:
                    matched = True
                    break
                if comparator == "lte" and numeric_val <= threshold_val:
                    matched = True
                    break
                if comparator == "lt" and numeric_val < threshold_val:
                    matched = True
                    break

        flags[feature_key] = 1 if matched else 0

    return flags


__all__ = ["tag_listing_features", "FEATURE_LIBRARY"]


