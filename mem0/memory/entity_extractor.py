"""
Entity and relationship extraction for write-time processing.

This module extracts entities, relationships, and dates from text using LLM tool calling,
storing them as metadata for fast search without requiring LLM calls at search time.
"""

import logging
from typing import Any, Dict, List, Optional

from mem0.graphs.tools import (
    EXTRACT_ENTITIES_TOOL,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    RELATIONS_TOOL,
    RELATIONS_STRUCT_TOOL,
)

logger = logging.getLogger(__name__)

# Enhanced system prompt for entity extraction with date support
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a smart assistant who extracts entities and their types from text.

Extract ALL entities including:
- People (names, roles)
- Organizations (companies, teams, institutions)
- Locations (cities, countries, places)
- Dates/Times (specific dates, time periods - use format like "january_15_2024" for dates)
- Objects (items, products, things)
- Preferences (things liked/disliked, interests)
- Activities (hobbies, sports, actions)
- Food (meals, cuisines, dishes)

If the text contains self-references like 'I', 'me', 'my', 'myself', use '{user_id}' as the entity.

Extract entities as they appear - do not infer or add entities not mentioned.
***DO NOT*** answer any questions in the text - only extract entities."""

# System prompt for relationship extraction
RELATIONS_EXTRACTION_SYSTEM_PROMPT = """You are a smart assistant who identifies relationships between entities in text.

Given a list of entities and the original text, establish relationships between them.

Guidelines:
1. Use clear, concise relationship names (e.g., "likes", "works_at", "lives_in", "met_on")
2. If the text contains self-references like 'I', 'me', 'my', use '{user_id}' as the source entity
3. Only create relationships that are explicitly stated or clearly implied in the text
4. For temporal events, create relationships like "happened_on", "met_on", "visited_on"

Example relationships:
- likes, loves, enjoys, prefers, dislikes, hates, avoids
- works_at, employed_at, lives_in, located_in, from
- knows, met, friends_with, related_to
- happened_on, visited_on, started_on, ended_on"""


class EntityExtractor:
    """
    Extracts entities, relationships, and dates from text using LLM tool calling.

    This extractor is designed for write-time processing, storing extracted data
    as metadata for fast search without requiring LLM calls during search.
    """

    def __init__(self, llm, llm_provider: str = "openai"):
        """
        Initialize the EntityExtractor.

        Args:
            llm: The LLM instance to use for extraction
            llm_provider: The LLM provider name (for selecting structured vs regular tools)
        """
        self.llm = llm
        self.llm_provider = llm_provider

        # Select appropriate tools based on provider
        self._use_structured = llm_provider in ["azure_openai_structured", "openai_structured"]

    def extract(self, text: str, user_id: str = "user") -> Dict[str, Any]:
        """
        Extract entities and relationships from text.

        Args:
            text: The text to extract entities from
            user_id: The user ID for self-reference resolution

        Returns:
            Dict containing:
                - entities: List of {"name": str, "type": str}
                - relationships: List of {"source": str, "relationship": str, "destination": str}
                - entity_names: Flat list of entity names for filtering
                - relationship_types: Flat list of relationship types for filtering
                - event_dates: List of ISO date strings for date filtering
        """
        result = {
            "entities": [],
            "relationships": [],
            "entity_names": [],
            "relationship_types": [],
            "event_dates": [],
        }

        try:
            # Step 1: Extract entities
            entities = self._extract_entities(text, user_id)
            result["entities"] = entities
            result["entity_names"] = [e["name"] for e in entities]

            # Extract dates from entities
            result["event_dates"] = self._extract_dates_from_entities(entities)

            # Step 2: Extract relationships (only if we have entities)
            if entities:
                entity_names = [e["name"] for e in entities]
                relationships = self._extract_relationships(text, entity_names, user_id)
                result["relationships"] = relationships
                result["relationship_types"] = list(set(r["relationship"] for r in relationships))

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        return result

    def _extract_entities(self, text: str, user_id: str) -> List[Dict[str, str]]:
        """Extract entities from text using LLM."""
        tools = [EXTRACT_ENTITIES_STRUCT_TOOL] if self._use_structured else [EXTRACT_ENTITIES_TOOL]

        system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(user_id=user_id)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            tools=tools,
        )

        entities = []
        try:
            for tool_call in response.get("tool_calls", []):
                if tool_call.get("name") != "extract_entities":
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    entity_name = item.get("entity", "").lower().replace(" ", "_")
                    entity_type = item.get("entity_type", "unknown").lower().replace(" ", "_")
                    if entity_name:
                        entities.append({"name": entity_name, "type": entity_type})
        except Exception as e:
            logger.warning(f"Error parsing entity extraction response: {e}")

        return entities

    def _extract_relationships(
        self, text: str, entity_names: List[str], user_id: str
    ) -> List[Dict[str, str]]:
        """Extract relationships between entities using LLM."""
        tools = [RELATIONS_STRUCT_TOOL] if self._use_structured else [RELATIONS_TOOL]

        system_prompt = RELATIONS_EXTRACTION_SYSTEM_PROMPT.format(user_id=user_id)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Entities: {entity_names}\n\nText: {text}"},
            ],
            tools=tools,
        )

        relationships = []
        try:
            for tool_call in response.get("tool_calls", []):
                # Handle both "establish_relationships" and "establish_relations" function names
                if tool_call.get("name") not in ["establish_relationships", "establish_relations"]:
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    source = item.get("source", "").lower().replace(" ", "_")
                    relationship = item.get("relationship", "").lower().replace(" ", "_")
                    destination = item.get("destination", "").lower().replace(" ", "_")
                    if source and relationship and destination:
                        relationships.append({
                            "source": source,
                            "relationship": relationship,
                            "destination": destination,
                        })
        except Exception as e:
            logger.warning(f"Error parsing relationship extraction response: {e}")

        return relationships

    def _extract_dates_from_entities(self, entities: List[Dict[str, str]]) -> List[str]:
        """
        Extract and normalize dates from entity list.

        Looks for entities with type "date" or date-like patterns in names,
        and normalizes them to ISO format (YYYY-MM-DD) where possible.
        """
        dates = []

        try:
            from dateutil import parser as date_parser
        except ImportError:
            logger.warning("dateutil not installed, date normalization disabled")
            return dates

        for entity in entities:
            entity_type = entity.get("type", "").lower()
            entity_name = entity.get("name", "")

            # Check if entity is a date type
            if entity_type in ["date", "time", "datetime", "day", "month", "year"]:
                normalized = self._normalize_date(entity_name, date_parser)
                if normalized:
                    dates.append(normalized)
            # Also try to parse date-like entity names
            elif self._looks_like_date(entity_name):
                normalized = self._normalize_date(entity_name, date_parser)
                if normalized:
                    dates.append(normalized)

        return list(set(dates))  # Remove duplicates

    def _normalize_date(self, date_str: str, date_parser) -> Optional[str]:
        """Convert various date formats to ISO (YYYY-MM-DD)."""
        # Clean up entity name format (e.g., "january_15_2024" -> "january 15 2024")
        cleaned = date_str.replace("_", " ")

        try:
            parsed = date_parser.parse(cleaned, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like it might contain a date."""
        import re

        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'january|february|march|april|may|june|july|august|september|october|november|december',
            r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',
        ]

        text_lower = text.lower().replace("_", " ")
        for pattern in date_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
