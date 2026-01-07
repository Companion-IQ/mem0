"""
Fast entity matching for search-time processing.

This module provides entity, relationship, and date matching from user queries
WITHOUT using LLM calls - enabling fast search filtering using pre-extracted metadata.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EntityMatcher:
    """
    Matches entities, relationships, and dates in queries using local methods (no LLM).

    Strategies:
    1. Self-reference resolution (I, me, my -> user_id)
    2. Keyword matching against known entity names
    3. Relationship type detection from query words
    4. Date extraction and normalization
    """

    # Self-reference pronouns
    SELF_REFS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}

    # Map query words to relationship types
    # When user asks "what do I like?", we search for memories with these relationship types
    RELATIONSHIP_KEYWORDS = {
        # Likes/preferences
        "like": ["likes", "loves", "enjoys", "prefers", "interested_in", "favorite"],
        "love": ["loves", "likes", "enjoys", "adores"],
        "enjoy": ["enjoys", "likes", "loves"],
        "prefer": ["prefers", "likes", "favors"],
        "favorite": ["likes", "loves", "prefers", "favorite"],
        "interest": ["interested_in", "likes", "curious_about"],
        # Dislikes
        "dislike": ["dislikes", "hates", "avoids", "doesnt_like"],
        "hate": ["hates", "dislikes", "despises"],
        "avoid": ["avoids", "dislikes", "stays_away_from"],
        # Work/professional
        "work": ["works_at", "employed_at", "works_for", "employed_by"],
        "job": ["works_at", "employed_at", "job_at"],
        "employ": ["employed_at", "works_at", "hired_by"],
        # Location
        "live": ["lives_in", "located_in", "resides_in"],
        "from": ["from", "born_in", "originates_from"],
        "located": ["located_in", "based_in", "situated_in"],
        # Relationships
        "know": ["knows", "met", "friends_with", "acquainted_with"],
        "friend": ["friends_with", "knows", "befriended"],
        "met": ["met", "knows", "encountered"],
        "married": ["married_to", "spouse_of", "wed_to"],
        "dating": ["dating", "in_relationship_with", "seeing"],
        # Activities
        "do": ["does", "performs", "engages_in"],
        "play": ["plays", "enjoys", "practices"],
        "study": ["studies", "learning", "studying"],
        "learn": ["learning", "studies", "studying"],
        # Time-based
        "happen": ["happened_on", "occurred_on", "took_place_on"],
        "visit": ["visited", "went_to", "traveled_to"],
        "went": ["went_to", "visited", "traveled_to"],
    }

    # Date patterns for extraction from queries
    DATE_PATTERNS = [
        # ISO format: 2024-01-15
        (r"\b(\d{4}-\d{2}-\d{2})\b", None),
        # US format: 01/15/2024 or 1/15/24
        (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", None),
        # Month + Year only (no day): December 2025, Jan 2024
        # MUST come before full month patterns to avoid "December 2025" matching as "December 20, 25"
        (
            r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b",
            "month_year",
        ),
        (r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b", "month_year"),
        # Full month names: January 15, 2024 or January 15th 2024
        (
            r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})\b",
            None,
        ),
        # Abbreviated months: Jan 15, 2024
        (r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})\b", None),
        # Relative dates
        (r"\b(yesterday)\b", "relative"),
        (r"\b(today)\b", "relative"),
        (r"\b(tomorrow)\b", "relative"),
        # Relative periods
        (r"\b(last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "relative"),
        (r"\b(this\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "relative"),
        (r"\b(next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "relative"),
    ]

    def __init__(self):
        """Initialize the EntityMatcher."""
        # Try to import dateutil for date parsing
        try:
            from dateutil import parser as date_parser
            from dateutil.relativedelta import relativedelta

            self._date_parser = date_parser
            self._relativedelta = relativedelta
            self._dateutil_available = True
        except ImportError:
            logger.warning("dateutil not installed, date parsing will be limited")
            self._date_parser = None
            self._relativedelta = None
            self._dateutil_available = False

    def match(
        self,
        query: str,
        user_id: str,
        known_entities: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities, relationship types, and dates from a query without LLM.

        Args:
            query: The user's search query
            user_id: The user ID for self-reference resolution
            known_entities: Optional set of known entity names for keyword matching

        Returns:
            Dict containing:
                - entities: List of matched entity names
                - relationship_types: List of detected relationship types
                - date_filter: Dict with "start" and "end" ISO dates, or None
        """
        query_lower = query.lower()
        result = {
            "entities": [],
            "relationship_types": [],
            "date_filter": None,
        }

        # 1. Self-reference resolution
        if self._has_self_reference(query_lower):
            result["entities"].append(user_id)

        # 2. Entity keyword matching
        if known_entities:
            matched = self._match_known_entities(query_lower, known_entities)
            for entity in matched:
                if entity not in result["entities"]:
                    result["entities"].append(entity)

        # 3. Relationship type detection
        relationship_types = self._detect_relationship_types(query_lower)
        result["relationship_types"] = relationship_types

        # 4. Date extraction
        result["date_filter"] = self._extract_date_filter(query_lower)

        return result

    def _has_self_reference(self, query: str) -> bool:
        """Check if query contains self-reference pronouns."""
        words = set(re.findall(r"\b\w+\b", query))
        return bool(words & self.SELF_REFS)

    def _match_known_entities(self, query: str, known_entities: Set[str]) -> List[str]:
        """Match query against known entity names."""
        matched = []

        for entity in known_entities:
            # Normalize entity for matching (replace underscores with spaces)
            entity_normalized = entity.replace("_", " ")

            # Check for exact match or partial match
            if entity_normalized in query or entity in query:
                matched.append(entity)
            else:
                # Check if any word in entity appears in query (for partial matches)
                entity_words = entity_normalized.split()
                if len(entity_words) > 1:
                    # Multi-word entity - check if all words appear
                    if all(word in query for word in entity_words):
                        matched.append(entity)

        return matched

    def _detect_relationship_types(self, query: str) -> List[str]:
        """Detect relationship types from query keywords."""
        relationship_types = set()

        # Extract words from query
        words = set(re.findall(r"\b\w+\b", query))

        for keyword, rel_types in self.RELATIONSHIP_KEYWORDS.items():
            if keyword in words:
                relationship_types.update(rel_types)

        return list(relationship_types)

    def _extract_date_filter(self, query: str) -> Optional[Dict[str, str]]:
        """Extract date or date range from query."""
        # First, check for range queries
        range_filter = self._extract_date_range(query)
        if range_filter:
            return range_filter

        # Then, check for single date references
        for pattern, pattern_type in self.DATE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                date_str = match.group(1)

                if pattern_type == "relative":
                    iso_date = self._parse_relative_date(date_str)
                    if iso_date:
                        return {"start": iso_date, "end": iso_date}
                elif pattern_type == "month_year":
                    # Month + Year pattern - return full month range
                    month_range = self._parse_month_year(date_str)
                    if month_range:
                        return month_range
                else:
                    iso_date = self._parse_date(date_str)
                    if iso_date:
                        # Single date - use same date for start and end
                        return {"start": iso_date, "end": iso_date}

        return None

    def _parse_month_year(self, date_str: str) -> Optional[Dict[str, str]]:
        """Parse a month+year string and return the full month range."""
        import calendar

        if not self._dateutil_available:
            return None

        try:
            # Parse to get the month and year
            cleaned = date_str.strip().replace("_", " ")
            parsed = self._date_parser.parse(cleaned, fuzzy=True)
            year = parsed.year
            month = parsed.month

            # Get the last day of the month
            _, last_day = calendar.monthrange(year, month)

            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{last_day:02d}"

            return {"start": start_date, "end": end_date}
        except Exception:
            return None

    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract date range from queries like 'between X and Y' or 'from X to Y'."""
        # Pattern for "between X and Y"
        between_pattern = r"between\s+(.+?)\s+and\s+(.+?)(?:\s|$|,|\?)"
        match = re.search(between_pattern, query, re.IGNORECASE)
        if match:
            start_str = match.group(1).strip()
            end_str = match.group(2).strip()
            start_date = self._parse_date(start_str)
            end_date = self._parse_date(end_str)
            if start_date and end_date:
                return {"start": start_date, "end": end_date}

        # Pattern for "from X to Y"
        from_to_pattern = r"from\s+(.+?)\s+to\s+(.+?)(?:\s|$|,|\?)"
        match = re.search(from_to_pattern, query, re.IGNORECASE)
        if match:
            start_str = match.group(1).strip()
            end_str = match.group(2).strip()
            start_date = self._parse_date(start_str)
            end_date = self._parse_date(end_str)
            if start_date and end_date:
                return {"start": start_date, "end": end_date}

        return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse a date string to ISO format (YYYY-MM-DD)."""
        if not self._dateutil_available:
            return self._parse_date_fallback(date_str)

        try:
            # Clean up the string
            cleaned = date_str.strip().replace("_", " ")
            parsed = self._date_parser.parse(cleaned, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _parse_date_fallback(self, date_str: str) -> Optional[str]:
        """Fallback date parsing without dateutil."""
        # Try ISO format
        iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if iso_match:
            return date_str

        # Try US format MM/DD/YYYY
        us_match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", date_str)
        if us_match:
            month, day, year = us_match.groups()
            if len(year) == 2:
                year = "20" + year if int(year) < 50 else "19" + year
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        return None

    def _parse_relative_date(self, date_str: str) -> Optional[str]:
        """Parse relative date references like 'yesterday', 'last week'."""
        today = datetime.now()
        date_str_lower = date_str.lower().strip()

        if date_str_lower == "today":
            return today.strftime("%Y-%m-%d")
        elif date_str_lower == "yesterday":
            return (today - self._timedelta(days=1)).strftime("%Y-%m-%d") if self._dateutil_available else None
        elif date_str_lower == "tomorrow":
            return (today + self._timedelta(days=1)).strftime("%Y-%m-%d") if self._dateutil_available else None

        # Handle "last week", "last month", etc.
        if "last" in date_str_lower:
            if "week" in date_str_lower:
                return (today - self._timedelta(days=7)).strftime("%Y-%m-%d") if self._dateutil_available else None
            elif "month" in date_str_lower:
                return (today - self._timedelta(days=30)).strftime("%Y-%m-%d") if self._dateutil_available else None
            elif "year" in date_str_lower:
                return (today - self._timedelta(days=365)).strftime("%Y-%m-%d") if self._dateutil_available else None

        return None

    def _timedelta(self, days: int):
        """Create a timedelta-like object."""
        from datetime import timedelta

        return timedelta(days=days)
